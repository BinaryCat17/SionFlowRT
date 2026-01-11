use crate::ir_graph::{IRGraph};
use crate::model::{Op, TensorShape, DataType};
use petgraph::algo::toposort;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

pub struct LinearIR {
    pub nodes: Vec<LinearNode>,
    pub outputs: HashMap<String, String>,
}

pub struct LinearNode {
    pub id: String,
    pub op: Op,
    pub inputs: Vec<String>, 
    pub shape: TensorShape,
    pub dtype: DataType,
}

impl LinearIR {

    pub fn from_ir_graph(ir: IRGraph, type_map: &HashMap<String, DataType>) -> anyhow::Result<Self> {

        let order = toposort(&ir.graph, None)

            .map_err(|_| anyhow::anyhow!("Cycle detected in IR Graph during linearization"))?;



        let mut nodes = Vec::new();

        for idx in order {

            let ir_node = &ir.graph[idx];

            

            let mut incoming_edges: Vec<_> = ir.graph.edges_directed(idx, petgraph::Direction::Incoming).collect();

            incoming_edges.sort_by_key(|e| *e.weight());

            

            let mut inputs = Vec::new();

            for edge in incoming_edges {

                inputs.push(ir.graph[edge.source()].id.clone());

            }



            // Маппинг типа: "float" -> DataType::F32

            let resolved_dtype = ir_node.dtype.as_ref()

                .and_then(|t| type_map.get(t))

                .cloned()

                .unwrap_or(DataType::F32);



            nodes.push(LinearNode {

                id: ir_node.id.clone(),

                op: ir_node.op.clone(),

                inputs,

                shape: ir_node.shape.clone().unwrap_or_else(|| TensorShape { dims: vec![] }),

                dtype: resolved_dtype,

            });

        }



        Ok(LinearIR { nodes, outputs: ir.outputs })

    }

}
