pub mod ir;

use crate::resolver::ir::ResolvedIR;
use crate::linearizer::ir::{LinearIR, LinearNode};
use petgraph::algo::toposort;
use petgraph::visit::EdgeRef;

pub fn linearize(resolved: ResolvedIR) -> anyhow::Result<LinearIR> {
    let mut nodes = Vec::new();
    
    let order = toposort(&resolved.graph, None)
        .map_err(|_| anyhow::anyhow!("Cycle detected during linearization"))?;

    for idx in order {
        let node = &resolved.graph[idx];
        
        let mut inputs = Vec::new();
        let mut incoming: Vec<_> = resolved.graph.edges_directed(idx, petgraph::Direction::Incoming).collect();
        incoming.sort_by(|a, b| a.weight().dst_port.cmp(&b.weight().dst_port));
        
        for edge in incoming {
            let src_node = &resolved.graph[edge.source()];
            inputs.push((src_node.id.clone(), edge.weight().src_port.clone()));
        }

        nodes.push(LinearNode {
            id: node.id.clone(),
            op: node.op.clone(),
            inputs,
            shape: node.shape.clone(),
            dtype: node.dtype,
        });
    }

    Ok(LinearIR {
        nodes,
        inputs: resolved.inputs,
        outputs: resolved.outputs,
    })
}
