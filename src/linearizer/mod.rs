pub mod ir;

use crate::resolver::ir::ResolvedIR;
use crate::linearizer::ir::{LinearIR, LinearNode, InputConnection};
use petgraph::algo::toposort;
use petgraph::visit::EdgeRef;

pub fn linearize(resolved: ResolvedIR) -> anyhow::Result<LinearIR> {
    let mut nodes = Vec::new();
    let mut current_offset = 0;
    
    let order = toposort(&resolved.graph, None)
        .map_err(|_| anyhow::anyhow!("Cycle detected during linearization"))?;

    for idx in order {
        let node = &resolved.graph[idx];
        
        let mut inputs = Vec::new();
        let mut incoming: Vec<_> = resolved.graph.edges_directed(idx, petgraph::Direction::Incoming).collect();
        incoming.sort_by(|a, b| a.weight().dst_port.cmp(&b.weight().dst_port));
        
        for edge in incoming {
            let src_node = &resolved.graph[edge.source()];
            inputs.push(InputConnection {
                node_id: src_node.id.clone(),
                src_port: edge.weight().src_port.clone(),
                shape: src_node.shape.clone(),
            });
        }

        // Calculate offset for intermediate nodes (those that aren't pure inputs)
        let offset = if matches!(node.op, crate::core::op::Op::Input { .. }) {
            0
        } else {
            let start = current_offset;
            if !matches!(node.op, crate::core::op::Op::Output { .. }) {
                match &node.op {
                    crate::core::op::Op::Split { parts, .. } => {
                        current_offset += parts;
                    }
                    _ => {
                        current_offset += 1;
                    }
                }
            }
            start
        };

        nodes.push(LinearNode {
            id: node.id.clone(),
            op: node.op.clone(),
            inputs,
            shape: node.shape.clone(),
            dtype: node.dtype,
            offset,
        });
    }

    Ok(LinearIR {
        nodes,
        inputs: resolved.inputs,
        outputs: resolved.outputs,
    })
}
