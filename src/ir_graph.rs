use crate::model::{Op, TensorShape};
use crate::json_graph::{InlinedPayload, InlineResult, LogicalGraph};
use crate::pipeline::{Stage, CompilerContext, IRPassFn};
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct IngestionStage {
    passes: Vec<IRPassFn>,
}

impl IngestionStage {
    pub fn new() -> Self { Self { passes: Vec::new() } }
    pub fn with_pass(mut self, pass: IRPassFn) -> Self {
        self.passes.push(pass);
        self
    }
}

impl Stage for IngestionStage {
    fn name(&self) -> &str { "Ingestion & Local Optimization" }
    fn run(&self, ctx: &mut CompilerContext) -> anyhow::Result<()> {
        let manifest = ctx.manifest.as_ref().unwrap();
        let mut ir_graphs = HashMap::new();

        for prog_entry in &manifest.programs {
            let logical = load_logical_graph(&prog_entry.path)?;
            let mut ir = IRGraph::from_inline_result(logical.inline(), Some(prog_entry.id.clone()))?;
            
            for pass in &self.passes {
                pass(&mut ir);
            }
            
            ir_graphs.insert(prog_entry.id.clone(), ir);
        }
        ctx.ir_graphs = ir_graphs;
        Ok(())
    }
}

fn load_logical_graph(path: &str) -> anyhow::Result<LogicalGraph<serde_json::Value>> {
    let path_with_ext = if path.ends_with(".json") { path.to_string() } else { format!("{}.json", path) };
    let content = fs::read_to_string(&path_with_ext)?;
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
        |op| KernelRegistry::get_interface(&serde_json::from_value(op.clone()).unwrap())
    )
}

#[derive(Debug, Clone)]
pub struct IRGraph {
    pub graph: DiGraph<IRNode, usize>,
    pub outputs: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct IRNode {
    pub id: String,
    pub op: Op,
    pub shape: Option<TensorShape>,
    pub dtype: Option<String>,
    pub program_id: Option<String>,
}

impl IRGraph {
    pub fn from_inline_result(res: InlineResult<serde_json::Value>, program_id: Option<String>) -> anyhow::Result<Self> {
        let mut ir_graph = DiGraph::new();
        let mut index_map = HashMap::new();

        for idx in res.graph.node_indices() {
            let inlined_node = &res.graph[idx];
            let op = match &inlined_node.payload {
                InlinedPayload::Primitive(val) => serde_json::from_value::<Op>(val.clone())?,
                InlinedPayload::Input => Op::Input { name: inlined_node.id.clone() },
            };

            let parsed_shape = if let Some(s_val) = &inlined_node.shape {
                serde_json::from_value::<crate::model::TensorShape>(s_val.clone()).ok()
            } else {
                None
            };

            let ir_node = IRNode {
                id: inlined_node.id.clone(),
                op,
                shape: parsed_shape,
                dtype: inlined_node.dtype.clone(),
                program_id: program_id.clone(),
            };
            let new_idx = ir_graph.add_node(ir_node);
            index_map.insert(idx, new_idx);
        }

        for edge in res.graph.edge_references() {
            ir_graph.add_edge(index_map[&edge.source()], index_map[&edge.target()], *edge.weight());
        }

        Ok(IRGraph { graph: ir_graph, outputs: res.outputs })
    }
}

pub struct KernelRegistry;

impl KernelRegistry {
    pub fn get_interface(op: &Op) -> crate::json_graph::NodeInterface {
        use crate::json_graph::Port;
        let unknown_shape = serde_json::json!(["_"]);
        let float_type = "float".to_string();

        match op {
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Min | Op::Max | Op::Pow | Op::MatMul | Op::Conv => {
                crate::json_graph::NodeInterface {
                    inputs: vec![
                        Port { name: "left".into(), dtype: float_type.clone(), shape: unknown_shape.clone() },
                        Port { name: "right".into(), dtype: float_type.clone(), shape: unknown_shape.clone() }
                    ],
                    outputs: vec![Port { name: "output".into(), dtype: float_type, shape: unknown_shape }]
                }
            }
            Op::Sin | Op::Abs | Op::Sqrt | Op::Square | Op::Exp | Op::Log | Op::ReduceSum { .. } | Op::Reshape { .. } | Op::Transpose { .. } | Op::Broadcast => {
                crate::json_graph::NodeInterface {
                    inputs: vec![Port { name: "input".into(), dtype: float_type.clone(), shape: unknown_shape.clone() }],
                    outputs: vec![Port { name: "output".into(), dtype: float_type, shape: unknown_shape }]
                }
            }
            Op::Constant { values } => {
                let shape = serde_json::json!([values.len()]);
                crate::json_graph::NodeInterface {
                    inputs: vec![],
                    outputs: vec![Port { name: "output".into(), dtype: float_type, shape }]
                }
            }
            Op::Input { name } => crate::json_graph::NodeInterface {
                inputs: vec![],
                outputs: vec![Port { name: name.clone(), dtype: float_type, shape: unknown_shape }]
            },
            Op::Output { name } => crate::json_graph::NodeInterface {
                inputs: vec![Port { name: name.clone(), dtype: float_type, shape: unknown_shape }],
                outputs: vec![]
            },
        }
    }
}