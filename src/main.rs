mod model;
mod codegen_c;
mod manifest;
mod json_graph;
mod ir_graph;
mod ir_passes;
mod orchestration_passes;
mod linear_ir;
mod linear_passes;
mod shape_engine;
mod orchestrator;
mod pipeline;
mod builder;

use pipeline::{Pipeline, CompilerContext};
use manifest::{LoadManifestStage, ResolveParametersStage};
use ir_graph::IngestionStage;
use ir_passes::IRPasses;
use orchestrator::{OrchestrationStage};
use orchestration_passes::{OrchestrationPasses};
use linear_ir::{LoweringStage};
use codegen_c::CodegenStage;
use builder::BuildStage;

fn main() -> anyhow::Result<()> {
    let manifest_path = "assets/programs/painter/manifest.json";
    let gen_dir = "generated";
    let out_dir = "out";

    println!("SionFlowRT Compiler Starting...");
    
    let mut ctx = CompilerContext::new(manifest_path, gen_dir, out_dir);
    let mut pipeline = Pipeline::new();

    // 1. Stage: Load & Config
    pipeline.add_stage(LoadManifestStage);
    pipeline.add_stage(ResolveParametersStage);
    
    // 2. Stage: Ingestion & Local Hygiene
    pipeline.add_stage(IngestionStage::new()
        .with_pass(IRPasses::run_dce)
    );

    // 3. Stage: Global Orchestration & Analysis
    pipeline.add_stage(OrchestrationStage::new()
        .with_pass(OrchestrationPasses::run_dce)
        .with_pass(OrchestrationPasses::run_shape_inference)
    );

    // 4. Stage: Lowering & Backend Optimization
    pipeline.add_stage(LoweringStage::new()
        .with_pass(linear_passes::run_loop_fusion)
    );

    // 5. Stage: Codegen & GCC
    pipeline.add_stage(CodegenStage);
    pipeline.add_stage(BuildStage);

    pipeline.execute(&mut ctx)?;

    println!("SionFlowRT Compilation Finished Successfully.");
    Ok(())
}