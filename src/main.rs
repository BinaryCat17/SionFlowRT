mod core;
mod manifest;
mod analyzer;
mod inliner;
mod resolver;
mod linearizer;
mod codegen;
mod linker;

use std::path::Path;

fn main() -> anyhow::Result<()> {
    let manifest_path = "assets/programs/painter/manifest.json";
    println!("SionFlowRT 2.0 - Starting Compilation...");

    // 1. Manifest Stage
    let manifest_json = std::fs::read_to_string(manifest_path)?;
    let manifest = manifest::Manifest::from_json(&manifest_json)?;
    println!("  [1/6] Manifest loaded: {}", manifest_path);

    // 2. Analyzer Stage (Global Planning)
    let plan = analyzer::analyze_project(&manifest)?;
    println!("  [2/6] Project analysis complete. {} programs found.", plan.execution_order.len());

    // 3. Module Compilation (Per Program)
    for prog_id in &plan.execution_order {
        println!("  [3/6] Compiling module: {}", prog_id);
        
        let prog_def = manifest.programs.iter().find(|p| &p.id == prog_id).unwrap();
        let prog_interface = plan.programs.get(prog_id).ok_or_else(|| anyhow::anyhow!("Interface for {} not found", prog_id))?;
        let prog_path = if prog_def.path.ends_with(".json") { 
            prog_def.path.clone() 
        } else { 
            format!("{}.json", prog_def.path) 
        };
        
        // Inliner
        let raw_ir = inliner::load_and_inline(Path::new(&prog_path))?;
        println!("    - Inlining complete (nodes: {})", raw_ir.graph.node_count());

        // Resolver (using real inputs from the plan!)
        let resolved_ir = resolver::resolve_module(raw_ir, prog_interface.inputs.clone())?;
        println!("    - Type & Shape resolution complete");

        // Linearizer
        let linear_ir = linearizer::linearize(resolved_ir)?;
        println!("    - Linearization complete");

        // Codegen
        let c_code = codegen::generate_module_source(prog_id, &linear_ir);
        let h_code = codegen::generate_module_header(prog_id, &linear_ir);
        
        std::fs::create_dir_all("generated")?;
        std::fs::write(format!("generated/{}.c", prog_id), c_code)?;
        std::fs::write(format!("generated/{}.h", prog_id), h_code)?;
        println!("    - C code generated");
    }

    // 4. Linker Stage (Global Glue)
    let runtime_c = linker::generate_runtime_c(&plan);
    std::fs::write("generated/runtime.c", runtime_c)?;
    println!("  [4/6] Linker generated runtime.c");

    println!("SionFlowRT 2.0 - Compilation Finished Successfully.");
    Ok(())
}