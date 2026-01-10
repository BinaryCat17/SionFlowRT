mod model;
mod compiler;
mod codegen_c;
mod manifest;

use model::ComputationalGraph;
use compiler::Compiler;
use codegen_c::CodegenC;
use manifest::Manifest;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::path::Path;

pub struct CompiledProgram {
    pub compiler: Compiler,
    pub execution_order: Vec<NodeIndex>,
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
        let graph = ComputationalGraph::from_json(&fs::read_to_string(&prog_entry.path)?)?;
        let mut compiler = Compiler::new();
        let execution_order = compiler.build(graph)?;
        
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
