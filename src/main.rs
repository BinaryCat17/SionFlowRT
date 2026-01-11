mod model;
mod codegen_c;
mod manifest;
mod json_graph;
mod ir_graph;
mod ir_passes;
mod linear_ir;
mod linear_passes;

use manifest::Manifest;
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::path::Path;
use json_graph::LogicalGraph;
use ir_graph::IRGraph;
use linear_ir::LinearIR;
use codegen_c::CodegenC;

fn load_logical_graph(path: &str) -> anyhow::Result<LogicalGraph<serde_json::Value>> {
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
        |op| ir_graph::KernelRegistry::get_interface(&serde_json::from_value(op.clone()).unwrap())
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
    let parameters = manifest.parameters.clone().unwrap_or_default();

    for prog_entry in &manifest.programs {
        println!("Compiling program: {}", prog_entry.id);
        
        let logical_graph = load_logical_graph(&prog_entry.path)?;
        let inline_res = logical_graph.inline();
        
        let mut ir_graph = IRGraph::from_inline_result(inline_res)?;
        ir_passes::run_dce(&mut ir_graph);
        
        let mut linear_ir = LinearIR::from_ir_graph(ir_graph)?;
        linear_passes::run_shape_inference(&mut linear_ir, &parameters, &manifest.mappings, &prog_entry.id)?;
        
        programs.insert(prog_entry.id.clone(), linear_ir);
    }

    let codegen = CodegenC::new(programs, &manifest);
    let gen_code = codegen.generate("sdl2_runtime")?;
    
    fs::write(Path::new(gen_dir).join("generated.c"), gen_code.module)?;
    fs::write(Path::new(gen_dir).join("runtime.c"), gen_code.runtime)?;

    println!("Running GCC...");
    let status = Command::new("gcc")
        .args(["-O3", "-fopenmp", "generated/generated.c", "generated/runtime.c", "-o", "out/generated_bin", "-lSDL2", "-lm"])
        .status()?;

    if status.success() {
        println!("Build successful: out/generated_bin");
    } else {
        eprintln!("Compilation failed!");
        std::process::exit(1);
    }

    Ok(())
}
