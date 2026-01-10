use crate::model::{ComputationalGraph, Node, Op, Dimension, TensorShape};
use crate::manifest::{Manifest, MappingSource};
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
        for node in source.nodes {
            let id = node.id.clone();
            let idx = self.graph.add_node(node);
            self.node_map.insert(id, idx);
        }

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

        for (src, dst) in edges {
            self.graph.add_edge(src, dst, ());
        }

        let sorted = toposort(&self.graph, None)
            .map_err(|_| anyhow!("В графе обнаружен цикл!"))?;

        Ok(sorted)
    }

    pub fn resolve_shapes(&mut self, prog_id: &str, manifest: &Manifest, execution_order: &[NodeIndex], compiled_programs: &HashMap<String, crate::CompiledProgram>) -> anyhow::Result<()> {
        let mut changed = true;
        let mut passes = 0;

        while changed && passes < 10 {
            changed = false;
            passes += 1;

            for &idx in execution_order {
                let mut new_dims = None;
                
                {
                    let node = &self.graph[idx];
                    
                    // 1. Попытка разрешить из Манифеста
                    let mapping = manifest.mappings.iter().find(|m| m.program == prog_id && m.tensor == node.id);
                    if let Some(m) = mapping {
                        new_dims = match &m.source {
                            MappingSource::ScreenUV => Some(vec![
                                Dimension::Symbol("WIDTH".into()),
                                Dimension::Symbol("HEIGHT".into()),
                                Dimension::Value(2)
                            ]),
                            MappingSource::Display => Some(vec![
                                Dimension::Symbol("WIDTH".into()),
                                Dimension::Symbol("HEIGHT".into()),
                                Dimension::Value(4)
                            ]),
                            MappingSource::Link { program, output } => {
                                // Ищем программу в уже скомпилированных
                                if let Some(source_prog) = compiled_programs.get(program) {
                                    source_prog.compiler.graph.node_indices()
                                        .find(|&i| source_prog.compiler.graph[i].id == *output)
                                        .map(|i| source_prog.compiler.graph[i].shape.dims.clone())
                                } else if program == prog_id {
                                    // Ссылка на самого себя (Feedback loop)
                                    self.graph.node_indices()
                                        .find(|&i| self.graph[i].id == *output)
                                        .and_then(|i| {
                                            let dims = self.graph[i].shape.dims.clone();
                                            if dims.iter().any(|d| matches!(d, Dimension::Symbol(s) if s == "_")) {
                                                None
                                            } else {
                                                Some(dims)
                                            }
                                        })
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                    }

                    if new_dims.is_none() {
                        new_dims = self.infer_shape_from_ops(idx)?;
                    }
                }

                if let Some(dims) = new_dims {
                    if self.apply_shape_update(idx, dims) {
                        changed = true;
                    }
                }
            }
        }

        for &idx in execution_order {
            let node = &self.graph[idx];
            if node.shape.dims.iter().any(|d| matches!(d, Dimension::Symbol(s) if s == "_")) {
                return Err(anyhow!("Не удалось разрешить форму тензора '{}' в программе '{}'.", node.id, prog_id));
            }
        }

        Ok(())
    }

    fn apply_shape_update(&mut self, idx: NodeIndex, dims: Vec<Dimension>) -> bool {
        let node = &mut self.graph[idx];
        let mut updated = false;
        for (i, dim) in dims.into_iter().enumerate() {
            if i < node.shape.dims.len() {
                if matches!(node.shape.dims[i], Dimension::Symbol(ref s) if s == "_") && !matches!(dim, Dimension::Symbol(ref s) if s == "_") {
                    node.shape.dims[i] = dim;
                    updated = true;
                }
            }
        }
        updated
    }

    fn infer_shape_from_ops(&self, idx: NodeIndex) -> anyhow::Result<Option<Vec<Dimension>>> {
        let node = &self.graph[idx];
        match &node.op {
            Op::Add { left, .. } | Op::Mul { left, .. } | Op::Sin { input: left } | Op::Transpose { input: left, .. } => {
                let parent_idx = self.node_map.get(left).unwrap();
                let parent_node = &self.graph[*parent_idx];
                if parent_node.shape.dims.iter().any(|d| matches!(d, Dimension::Symbol(s) if s == "_")) {
                    Ok(None)
                } else {
                    if matches!(node.op, Op::Transpose { .. }) {
                        Ok(None)
                    } else {
                        Ok(Some(parent_node.shape.dims.clone()))
                    }
                }
            }
            Op::ReduceSum { input, axis } => {
                let parent_idx = self.node_map.get(input).unwrap();
                let parent_node = &self.graph[*parent_idx];
                if parent_node.shape.dims.iter().any(|d| matches!(d, Dimension::Symbol(s) if s == "_")) {
                    Ok(None)
                } else {
                    let mut new_dims = parent_node.shape.dims.clone();
                    if *axis < new_dims.len() {
                        new_dims.remove(*axis);
                    }
                    Ok(Some(new_dims))
                }
            }
            _ => Ok(None),
        }
    }

    fn get_dependencies(&self, op: &Op) -> Vec<String> {
        match op {
            Op::Input { .. } => vec![],
            Op::Constant { .. } => vec![],
            Op::Add { left, right } => vec![left.clone(), right.clone()],
            Op::Mul { left, right } => vec![left.clone(), right.clone()],
            Op::Sin { input } => vec![input.clone()],
            Op::Transpose { input, .. } => vec![input.clone()],
            Op::ReduceSum { input, .. } => vec![input.clone()],
            Op::MatMul { left, right } => vec![left.clone(), right.clone()],
            Op::Conv { input, kernel } => vec![input.clone(), kernel.clone()],
        }
    }
}