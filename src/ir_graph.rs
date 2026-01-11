use crate::model::{Op, TensorShape, DataType};
use crate::json_graph::{InlinedPayload, InlineResult};
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

pub struct IRGraph {
    pub graph: DiGraph<IRNode, usize>,
    pub outputs: HashMap<String, String>,
}

pub struct IRNode {
    pub id: String,
    pub op: Op,
    pub shape: Option<TensorShape>, 
    pub dtype: Option<DataType>,   
}

impl IRGraph {
    pub fn from_inline_result(res: InlineResult<serde_json::Value>) -> anyhow::Result<Self> {
        let mut ir_graph = DiGraph::new();
        let mut index_map = HashMap::new();

        for idx in res.graph.node_indices() {
            let inlined_node = &res.graph[idx];
            let op = match &inlined_node.payload {
                InlinedPayload::Primitive(val) => serde_json::from_value::<Op>(val.clone())?,
                InlinedPayload::Input => Op::Input { name: inlined_node.id.clone() },
            };

            let ir_node = IRNode {
                id: inlined_node.id.clone(),
                op,
                shape: None,
                dtype: None,
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
        let unknown_shape = serde_json::json!([{"Symbol": "_"}]);
        let f32 = "F32".to_string();

        match op {
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Min | Op::Max | Op::Pow | Op::MatMul | Op::Conv => crate::json_graph::NodeInterface {
                inputs: vec![
                    Port { name: "left".into(), dtype: f32.clone(), shape: unknown_shape.clone() },
                    Port { name: "right".into(), dtype: f32.clone(), shape: unknown_shape.clone() }
                ],
                outputs: vec![Port { name: "output".into(), dtype: f32, shape: unknown_shape }]
            },
            Op::Sin | Op::Abs | Op::Sqrt | Op::Square | Op::Exp | Op::Log | Op::ReduceSum { .. } | Op::Reshape { .. } | Op::Transpose { .. } | Op::Broadcast => crate::json_graph::NodeInterface {
                inputs: vec![Port { name: "input".into(), dtype: f32.clone(), shape: unknown_shape.clone() }],
                outputs: vec![Port { name: "output".into(), dtype: f32, shape: unknown_shape }]
            },
            Op::Clamp => crate::json_graph::NodeInterface {
                inputs: vec![
                    Port { name: "input".into(), dtype: f32.clone(), shape: unknown_shape.clone() },
                    Port { name: "min".into(), dtype: f32.clone(), shape: unknown_shape.clone() },
                    Port { name: "max".into(), dtype: f32.clone(), shape: unknown_shape.clone() }
                ],
                outputs: vec![Port { name: "output".into(), dtype: f32, shape: unknown_shape }]
            },
            Op::Constant { values } => {
                let shape = serde_json::json!([{"Value": values.len()}]);
                crate::json_graph::NodeInterface {
                    inputs: vec![],
                    outputs: vec![Port { name: "output".into(), dtype: f32, shape }]
                }
            },
            Op::Input { name } => crate::json_graph::NodeInterface {
                inputs: vec![],
                outputs: vec![Port { name: name.clone(), dtype: f32, shape: unknown_shape }]
            },
            Op::Output { name } => crate::json_graph::NodeInterface {
                inputs: vec![Port { name: name.clone(), dtype: f32, shape: unknown_shape }],
                outputs: vec![]
            },
        }
    }
}