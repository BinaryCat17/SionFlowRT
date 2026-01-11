use crate::ir_graph::IRGraph;
use petgraph::visit::IntoNodeReferences;
use petgraph::Direction;
use std::collections::HashSet;

pub fn run_dce(ir: &mut IRGraph) {
    let mut to_keep = HashSet::new();
    let mut stack = Vec::new();

    // 1. Находим все выходы (Output) и узлы, помеченные как результаты графа
    for (idx, node) in ir.graph.node_references() {
        if matches!(node.op, crate::model::Op::Output { .. }) {
            stack.push(idx);
            to_keep.insert(idx);
        }
        // Если ID узла упомянут как реальный ID для какого-то выхода
        if ir.outputs.values().any(|real_id| real_id == &node.id) {
            stack.push(idx);
            to_keep.insert(idx);
        }
    }

    // 2. Идем вверх по графу от выходов
    while let Some(idx) = stack.pop() {
        for neighbor in ir.graph.neighbors_directed(idx, Direction::Incoming) {
            if to_keep.insert(neighbor) {
                stack.push(neighbor);
            }
        }
    }

    // 3. Удаляем все узлы, которые не попали в to_keep
    ir.graph.retain_nodes(|_, idx| to_keep.contains(&idx));
}
