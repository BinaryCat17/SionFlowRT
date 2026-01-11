use crate::model::{Op, TensorShape, DataType};
use crate::pipeline::{Stage, CompilerContext, LinearPassFn};
use crate::orchestrator::Orchestrator;
use std::collections::HashMap;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct LinearIR {
    pub nodes: Vec<LinearNode>,
    pub outputs: HashMap<String, String>,
    pub groups: Vec<Vec<usize>>, // Индексы узлов, сгруппированные для Fusion
}

#[derive(Debug, Clone, Serialize)]
pub struct LinearNode {
    pub id: String,
    pub op: Op,
    pub inputs: Vec<String>,
    pub shape: TensorShape,
    pub dtype: DataType,
}

pub struct LoweringStage {
    passes: Vec<LinearPassFn>,
}

impl LoweringStage {
    pub fn new() -> Self { Self { passes: Vec::new() } }
    pub fn with_pass(mut self, pass: LinearPassFn) -> Self {
        self.passes.push(pass);
        self
    }
}

impl Stage for LoweringStage {
    fn name(&self) -> &str { "Lowering: Linearization & Local Optimizations" }
    fn run(&self, ctx: &mut CompilerContext) -> anyhow::Result<()> {
        let manifest = ctx.manifest.as_ref().unwrap();
        let unified_graph = ctx.unified_graph.as_ref().ok_or_else(|| anyhow::anyhow!("No unified graph"))?;
        
        // 1. Линеаризация
        let mut orchestration = Orchestrator::compile_to_orchestration(manifest, unified_graph)?;

        // 2. Запуск локальных Backend-пассов
        for ir in orchestration.programs.values_mut() {
            for pass in &self.passes {
                pass(ir)?;
            }
        }
        
        ctx.orchestration = Some(orchestration);
        Ok(())
    }
}

impl LinearIR {
}