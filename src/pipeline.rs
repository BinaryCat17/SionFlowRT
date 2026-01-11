use crate::manifest::Manifest;
use crate::ir_graph::IRGraph;
use crate::orchestrator::ProjectOrchestration;
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use std::path::PathBuf;

pub type Parameters = HashMap<String, usize>;
pub type UnifiedGraph = DiGraph<crate::ir_graph::IRNode, usize>;

// Типы функций-пассов для разных стадий
pub type IRPassFn = fn(&mut IRGraph);
pub type GlobalPassFn = fn(&mut UnifiedGraph, &Parameters) -> anyhow::Result<()>;
pub type LinearPassFn = fn(&mut crate::linear_ir::LinearIR) -> anyhow::Result<()>;

pub struct CompilerContext {
    pub manifest_path: PathBuf,
    pub gen_dir: PathBuf,
    pub out_dir: PathBuf,
    
    pub manifest: Option<Manifest>,
    pub parameters: Parameters,
    pub ir_graphs: HashMap<String, IRGraph>,
    pub unified_graph: Option<UnifiedGraph>,
    pub orchestration: Option<ProjectOrchestration>,
    
    pub generated_module: Option<String>,
    pub generated_runtime: Option<String>,
}

impl CompilerContext {
    pub fn new(manifest_path: &str, gen_dir: &str, out_dir: &str) -> Self {
        Self {
            manifest_path: PathBuf::from(manifest_path),
            gen_dir: PathBuf::from(gen_dir),
            out_dir: PathBuf::from(out_dir),
            manifest: None,
            parameters: HashMap::new(),
            ir_graphs: HashMap::new(),
            unified_graph: None,
            orchestration: None,
            generated_module: None,
            generated_runtime: None,
        }
    }
}

pub trait Stage {
    fn name(&self) -> &str;
    fn run(&self, ctx: &mut CompilerContext) -> anyhow::Result<()>;
}

pub struct Pipeline {
    stages: Vec<Box<dyn Stage>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn add_stage<S: Stage + 'static>(&mut self, stage: S) {
        self.stages.push(Box::new(stage));
    }

    pub fn execute(&self, ctx: &mut CompilerContext) -> anyhow::Result<()> {
        for stage in &self.stages {
            println!("[Stage: {}]", stage.name());
            stage.run(ctx)?;
        }
        Ok(())
    }
}
