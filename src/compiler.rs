use crate::model::{ComputationalGraph, Node};
use crate::manifest::Manifest;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use std::collections::HashMap;
use anyhow::anyhow;

pub struct Compiler {
    pub graph: DiGraph<Node, ()>,
    pub node_map: HashMap<String, NodeIndex>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn build(&mut self, source: ComputationalGraph) -> anyhow::Result<Vec<NodeIndex>> {
        for n in source.nodes {
            let id = n.id.clone();
            let idx = self.graph.add_node(n);
            self.node_map.insert(id, idx);
        }

        let mut edges = Vec::new();
        for &idx in self.graph.node_indices().collect::<Vec<_>>().iter() {
            for dep_id in &self.graph[idx].inputs {
                let dep_idx = self.node_map.get(dep_id)
                    .ok_or_else(|| anyhow!("Node '{}' not found (dependency of '{}')", dep_id, self.graph[idx].id))?;
                edges.push((*dep_idx, idx));
            }
        }

        for (src, dst) in edges {
            self.graph.add_edge(src, dst, ());
        }
        
        let order = toposort(&self.graph, None).map_err(|_| anyhow!("Cycle detected in graph"))?;
        Ok(order)
    }

    pub fn resolve_shapes(
        &mut self, 
        _prog_id: &str, 
        _manifest: &Manifest, 
        _execution_order: &[NodeIndex], 
        _compiled_programs: &HashMap<String, crate::CompiledProgram>
    ) -> anyhow::Result<()> {
        // Placeholder for future symbolic shape inference
        Ok(())
    }

    pub fn get_shape_by_id(&self, id: &str, params: &HashMap<String, usize>) -> Option<Vec<crate::model::Dimension>> {
        let idx = *self.node_map.get(id)?;
        Some(self.graph[idx].shape.dims.iter().map(|d| d.eval(params)).collect())
    }
}