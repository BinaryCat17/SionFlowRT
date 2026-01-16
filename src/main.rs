use anyhow::{Context};
use std::path::Path;

mod manifest;
mod analyzer;
mod inliner;
mod resolver;
mod linearizer;
mod codegen;
mod linker;
mod core;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: SionFlowRT <manifest.json> [--test] [--run]");
        return Ok(());
    }

    let manifest_path = &args[1];
    let is_test = args.contains(&"--test".to_string());
    let is_run = args.contains(&"--run".to_string());

    println!("SionFlowRT 2.0 - Starting Compilation...");

    // 1. Load Manifest
    let manifest_content = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("Failed to read manifest at {}", manifest_path))?;
    let manifest = manifest::Manifest::from_json(&manifest_content)?;
    println!("  [1/6] Manifest loaded: {}", manifest_path);

    // 2. Project Analysis
    let manifest_dir = Path::new(manifest_path).parent().unwrap_or(Path::new("."));
    let mut plan = analyzer::analyze_project(&manifest, manifest_dir)?;
    println!("  [2/6] Project analysis complete. {} programs found.", plan.programs.len());

    // 3. Module Compilation (Per Program)
    for prog_id in &plan.execution_order {
        println!("  [3/6] Compiling module: {}", prog_id);
        
        let prog_def = manifest.programs.iter().find(|p| &p.id == prog_id).unwrap();
        let prog_interface = plan.programs.get(prog_id).ok_or_else(|| anyhow::anyhow!("Interface for {} not found", prog_id))?;
        let prog_graph = plan.program_graphs.get(prog_id).cloned().ok_or_else(|| anyhow::anyhow!("Graph for {} not found", prog_id))?;
        let prog_path = if prog_def.path.ends_with(".json") { 
            prog_def.path.clone() 
        } else { 
            format!("{}.json", prog_def.path) 
        };
        
        let raw_ir = inliner::load_and_inline(prog_graph, Path::new(&prog_path), &manifest, &mut plan.synthetic_vars)?;
        println!("    - Inlining complete (nodes: {})", raw_ir.graph.node_count());

        let resolved_ir = resolver::resolve_module(raw_ir, prog_interface.inputs.clone())?;
        println!("    - Type & Shape resolution complete");

        let linear_ir = linearizer::linearize(resolved_ir)?;
        println!("    - Linearization complete");

        plan.workspace_info.insert(prog_id.clone(), linear_ir.get_workspace_slots());

        let c_code = codegen::generate_module_source(prog_id, &linear_ir);
        let h_code = codegen::generate_module_header(prog_id, &linear_ir);
        
        std::fs::create_dir_all("generated")?;
        std::fs::write(format!("generated/{}.c", prog_id), c_code)?;
        std::fs::write(format!("generated/{}.h", prog_id), h_code)?;
        println!("    - C code generated");
    }

    // 4. Linker (Generate top-level runtime)
    let runtime_c = linker::generate_runtime_c(&plan);
    std::fs::write("generated/runtime.c", runtime_c)?;
    println!("  [4/6] Linker generated runtime.c");

    // 5. Test Runner Generation
    if is_test || is_run {
        let runner_c = linker::generate_test_runner(&plan, &manifest.tests);
        std::fs::write("generated/test_runner.c", runner_c)?;
        println!("  [5/6] Generated test_runner.c");

        println!("  [6/6] Compiling and running...");
        std::fs::create_dir_all("out")?;
        
        let output_name = if cfg!(windows) { "out/test_runner.exe" } else { "out/test_runner" };
        
        let status = std::process::Command::new("gcc")
            .arg("generated/test_runner.c")
            .arg("-Igenerated")
            .arg("-o")
            .arg(output_name)
            .arg("-lm")
            .status()
            .context("Failed to execute gcc. Is it installed?")?;

        if !status.success() {
            anyhow::bail!("C compilation failed");
        }

        if is_test || is_run {
            let mut run_cmd = if cfg!(windows) {
                 std::process::Command::new(format!("{}.exe", output_name.strip_suffix(".exe").unwrap_or(output_name)))
            } else {
                 std::process::Command::new(format!("./{}", output_name))
            };

            let run_status = run_cmd
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .status()
                .context("Failed to run the compiled test runner")?;
            
            if is_test && !run_status.success() {
                anyhow::bail!("Tests failed");
            }
        }
    } else {
        println!("  [5/6] Skipping test generation (use --test to enable)");
        println!("  [6/6] Done.");
    }

    println!("SionFlowRT 2.0 - Compilation Finished Successfully.");
    Ok(())
}
