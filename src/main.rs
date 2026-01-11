mod model;
mod compiler;
mod codegen_c;
mod manifest;
mod graph_ext;

use model::{Op, ComputationalGraph, Node, DataType, TensorShape, KernelRegistry};
use compiler::Compiler;
use codegen_c::CodegenC;
use manifest::Manifest;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::path::Path;

use graph_ext::LogicalGraph;

pub struct CompiledProgram {
    pub compiler: Compiler,
    pub execution_order: Vec<NodeIndex>,
}

fn load_logical_graph(path: &str) -> anyhow::Result<LogicalGraph<Op>> {
    let path_with_ext = if path.ends_with(".json") { path.to_string() } else { format!("{}.json", path) };
    let content = fs::read_to_string(&path_with_ext)
        .map_err(|e| anyhow::anyhow!("Failed to read graph file {}: {}", path_with_ext, e))?;
    
    LogicalGraph::from_json(
        &content, 
        |sub_path| {
            let resolved_path = if Path::new(sub_path).exists() || sub_path.starts_with("assets/") {
                sub_path.to_string()
            } else {
                format!("assets/lib/{}", sub_path)
            };
            load_logical_graph(&resolved_path)
        },
        |op| KernelRegistry::get_interface(op)
    )
}

fn main() -> anyhow::Result<()> {
    let manifest_path = "assets/programs/painter/manifest.json";
    let gen_dir = "generated";
    let out_dir = "out";

    fs::create_dir_all(gen_dir)?;
    fs::create_dir_all(out_dir)?;

    let manifest = Manifest::from_json(&fs::read_to_string(manifest_path)?)?;
    let mut programs = HashMap::new();

    for prog_entry in &manifest.programs {
        let logical_graph = load_logical_graph(&prog_entry.path)?;
        let flat_records = logical_graph.flatten()?;
        
        let mut nodes = Vec::new();
        for rec in flat_records {
            if let Some(op) = rec.payload {
                let shape: TensorShape = serde_json::from_value(rec.interface.outputs[0].shape.clone())?;
                nodes.push(Node {
                    id: rec.id,
                    op,
                    inputs: rec.inputs,
                    shape,
                    dtype: DataType::F32, // TODO: map rec.interface.outputs[0].dtype_id
                    strides: None,
                });
            } else if rec.is_input {
                let shape: TensorShape = serde_json::from_value(rec.interface.outputs[0].shape.clone())?;
                nodes.push(Node {
                    id: rec.id.clone(),
                    op: Op::Input { name: rec.id },
                    inputs: vec![],
                    shape,
                    dtype: DataType::F32,
                    strides: None,
                });
            } else if rec.is_output {
                let shape: TensorShape = serde_json::from_value(rec.interface.inputs[0].shape.clone())?;
                nodes.push(Node {
                    id: rec.id.clone(),
                    op: Op::Output { name: rec.id },
                    inputs: rec.inputs,
                    shape,
                    dtype: DataType::F32,
                    strides: None,
                });
            }
        }

        let flat_graph = ComputationalGraph { imports: None, nodes };
        
        let mut compiler = Compiler::new();
        let execution_order = compiler.build(flat_graph)?;
        
        // Разрешаем неопределенные размерности (_) на основе манифеста
        compiler.resolve_shapes(&prog_entry.id, &manifest, &execution_order, &programs)?;
        
        programs.insert(prog_entry.id.clone(), CompiledProgram { compiler, execution_order });
    }

    let gen_code = CodegenC::new(programs, &manifest).generate("sdl2_runtime")?;
    let c_file_path = Path::new(gen_dir).join("generated.c");
    fs::write(&c_file_path, gen_code.module)?;
    
    let runtime_file_path = Path::new(gen_dir).join("runtime.c");
    fs::write(&runtime_file_path, gen_code.runtime)?;

    let bin_path = Path::new(out_dir).join("generated_bin");
    let status = Command::new("gcc")
        .arg("-O3")
        .arg("-fopenmp")
        .arg(&c_file_path)
        .arg(&runtime_file_path)
        .arg("-o")
        .arg(&bin_path)
        .arg("-lSDL2")
        .arg("-lm")
        .status()?;

    if status.success() {
        println!("Build successful: out/generated_bin");
    } else {
        eprintln!("Compilation failed!");
        std::process::exit(1);
    }

    Ok(())
}
