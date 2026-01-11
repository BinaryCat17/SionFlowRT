use crate::model::{Op, TensorShape};
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
    pub dtype: Option<String>,   
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
                dtype: inlined_node.dtype.clone(), // Теперь берем строку типа из инлайнера
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



                            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Min | Op::Max | Op::Pow | Op::MatMul | Op::Conv => crate::json_graph::NodeInterface {



                                inputs: vec![



                                    Port { name: "left".into(), dtype: float_type.clone(), shape: unknown_shape.clone() },



                                    Port { name: "right".into(), dtype: float_type.clone(), shape: unknown_shape.clone() }



                                ],



                                outputs: vec![Port { name: "output".into(), dtype: float_type, shape: unknown_shape }]



                            },



                                        Op::Sin | Op::Abs | Op::Sqrt | Op::Square | Op::Exp | Op::Log | Op::ReduceSum { .. } | Op::Reshape { .. } | Op::Transpose { .. } | Op::Broadcast => crate::json_graph::NodeInterface {



                                            inputs: vec![Port { name: "input".into(), dtype: float_type.clone(), shape: unknown_shape.clone() }],



                                            outputs: vec![Port { name: "output".into(), dtype: float_type, shape: unknown_shape }]



                                        },



                                        Op::Constant { values } => {



                            



                                let shape = serde_json::json!([values.len()]);



                                crate::json_graph::NodeInterface {



                                    inputs: vec![],



                                    outputs: vec![Port { name: "output".into(), dtype: float_type, shape }]



                                }



                            },



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
