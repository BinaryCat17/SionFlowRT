use crate::ir_graph::IRGraph;
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use std::collections::HashSet;

pub struct IRPasses;

impl IRPasses {
    /// Локальный DCE для одной программы перед слиянием
    pub fn run_dce(ir: &mut IRGraph) {
        let mut keep = HashSet::new();
        let mut worklist = Vec::new();

        // Корни — это выходы программы
        for (_name, node_id) in &ir.outputs {
            if let Some(idx) = ir.graph.node_indices().find(|&i| ir.graph[i].id == *node_id) {
                worklist.push(idx);
                keep.insert(idx);
            }
        }

        while let Some(idx) = worklist.pop() {
            for edge in ir.graph.edges_directed(idx, Direction::Incoming) {
                let src = edge.source();
                if keep.insert(src) {
                    worklist.push(src);
                }
            }
        }

        ir.graph.retain_nodes(|_, idx| keep.contains(&idx));
    }
}
