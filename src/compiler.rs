use crate::model::{ComputationalGraph, Node, Op};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use std::collections::HashMap;
use anyhow::anyhow;

pub struct Compiler {
    pub graph: DiGraph<Node, ()>,
    node_map: HashMap<String, NodeIndex>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn build(&mut self, source: ComputationalGraph) -> anyhow::Result<Vec<NodeIndex>> {
        // 1. Добавляем все узлы в petgraph
        for node in source.nodes {
            let id = node.id.clone();
            let idx = self.graph.add_node(node);
            self.node_map.insert(id, idx);
        }

        // 2. Собираем зависимости во временный вектор, чтобы не блокировать граф
        let mut edges = Vec::new();
        for &idx in self.graph.node_indices().collect::<Vec<_>>().iter() {
            let node = &self.graph[idx];
            let deps = self.get_dependencies(&node.op);
            
            for dep_id in deps {
                let dep_idx = self.node_map.get(&dep_id)
                    .ok_or_else(|| anyhow!("Узел '{}' ссылается на несуществующий узел '{}'", node.id, dep_id))?;
                edges.push((*dep_idx, idx));
            }
        }

        // 3. Добавляем ребра
        for (src, dst) in edges {
            self.graph.add_edge(src, dst, ());
        }

        // 4. Топологическая сортировка
        let sorted = toposort(&self.graph, None)
            .map_err(|_| anyhow!("В графе обнаружен цикл! Вычисления невозможны."))?;

        Ok(sorted)
    }

    fn get_dependencies(&self, op: &Op) -> Vec<String> {
        match op {
            Op::Input { .. } => vec![],
            Op::Add { left, right } => vec![left.clone(), right.clone()],
            Op::Mul { left, right } => vec![left.clone(), right.clone()],
            Op::Sin { input } => vec![input.clone()],
        }
    }
}