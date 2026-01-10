use crate::model::{ComputationalGraph, Node, Dimension, Op};
use crate::manifest::{Manifest, MappingSource};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
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

    fn inline_recursive(
        &self,
        graph: ComputationalGraph,
        base_path: &Path,
    ) -> anyhow::Result<Vec<Node>> {
        let mut result_nodes = Vec::new();
        let imports = graph.imports.clone().unwrap_or_default();

        for node in graph.nodes {
            match node.op {
                Op::Call { ref subgraph, ref inputs } => {
                    let first_slash = subgraph.find('/')
                        .ok_or_else(|| anyhow!("Неверный формат вызова подграфа: '{}'", subgraph))?;
                    
                    let prefix = &subgraph[..first_slash];
                    let sub_path_str = &subgraph[first_slash + 1..];
                    let sub_dir = imports.get(prefix)
                        .ok_or_else(|| anyhow!("Импорт '{}' не найден", prefix))?;
                    
                    let sub_path = base_path.join(sub_dir).join(format!("{}.json", sub_path_str));
                    let sub_graph_json = fs::read_to_string(&sub_path)?;
                    let sub_graph = ComputationalGraph::from_json(&sub_graph_json)?;

                    let inlined_sub_nodes = self.inline_recursive(sub_graph, base_path)?;
                    let mut sub_id_map: HashMap<String, String> = HashMap::new();
                    
                    // 1. Мапим входы и пробрасываем формы
                    let mut input_shapes: HashMap<String, Vec<Dimension>> = HashMap::new();
                    for sub_node in &inlined_sub_nodes {
                        if let Op::Input { ref name } = sub_node.op {
                            if let Some(parent_id) = inputs.get(name) {
                                sub_id_map.insert(sub_node.id.clone(), parent_id.clone());
                                // Запоминаем форму родителя, если она не пустая (не _)
                                if !node.shape.dims.is_empty() && !matches!(node.shape.dims[0], Dimension::Symbol(ref s) if s == "_") {
                                     input_shapes.insert(parent_id.clone(), node.shape.dims.clone());
                                }
                            } else {
                                sub_id_map.insert(sub_node.id.clone(), format!("{}/{}", node.id, sub_node.id));
                            }
                        }
                    }

                    // 2. Мапим выходы
                    for sub_node in &inlined_sub_nodes {
                        if let Op::Output { ref input, .. } = sub_node.op {
                            let source_id = sub_id_map.get(input).cloned().unwrap_or_else(|| format!("{}/{}", node.id, input));
                            sub_id_map.insert(sub_node.id.clone(), source_id);
                        }
                    }

                    // 3. Мапим остальные
                    for sub_node in &inlined_sub_nodes {
                        if !sub_id_map.contains_key(&sub_node.id) {
                            sub_id_map.insert(sub_node.id.clone(), format!("{}/{}", node.id, sub_node.id));
                        }
                    }

                    let main_out = inlined_sub_nodes.iter()
                        .filter_map(|n| if let Op::Output { ref name, ref input } = n.op { Some((name, input)) } else { None })
                        .find(|(n, _)| *n == "default" || *n == "output")
                        .map(|(_, i)| i)
                        .unwrap_or_else(|| {
                            inlined_sub_nodes.iter().filter_map(|n| if let Op::Output { ref input, .. } = n.op { Some(input) } else { None }).next().expect("Sub-graph must have output")
                        });

                    let call_result_id = sub_id_map.get(main_out).cloned().unwrap_or_else(|| format!("{}/{}", node.id, main_out));

                    for mut sub_node in inlined_sub_nodes {
                        if matches!(sub_node.op, Op::Input { .. } | Op::Output { .. }) { continue; }
                        sub_node.id = sub_id_map.get(&sub_node.id).unwrap().clone();
                        sub_node.op = sub_node.op.map_dependencies(|id| sub_id_map.get(id).cloned().unwrap_or_else(|| id.to_string()));
                        result_nodes.push(sub_node);
                    }

                    result_nodes.push(Node {
                        id: node.id.clone(),
                        op: Op::Output { name: "bridge".to_string(), input: call_result_id },
                        shape: node.shape.clone(),
                        dtype: node.dtype.clone(),
                        strides: None,
                    });
                }
                _ => result_nodes.push(node),
            }
        }

        let mut final_id_map: HashMap<String, String> = HashMap::new();
        for n in &result_nodes {
            if let Op::Output { ref name, ref input } = n.op {
                if name == "bridge" { final_id_map.insert(n.id.clone(), input.clone()); }
            }
        }

        if final_id_map.is_empty() { return Ok(result_nodes); }

        let mut final_nodes = Vec::new();
        for mut n in result_nodes {
            if let Op::Output { ref name, .. } = n.op {
                if name == "bridge" { continue; }
            }
            n.op = n.op.map_dependencies(|id| {
                let mut curr = id.to_string();
                while let Some(next) = final_id_map.get(&curr) { curr = next.clone(); }
                curr
            });
            final_nodes.push(n);
        }
        Ok(final_nodes)
    }

    pub fn build(&mut self, source: ComputationalGraph) -> anyhow::Result<Vec<NodeIndex>> {
        let inlined = self.inline_recursive(source, Path::new("."))?;
        for n in inlined {
            let id = n.id.clone();
            let idx = self.graph.add_node(n);
            self.node_map.insert(id, idx);
        }
        let mut edges = Vec::new();
        for &idx in self.graph.node_indices().collect::<Vec<_>>().iter() {
            for dep_id in self.graph[idx].op.get_dependencies() {
                let dep_idx = self.node_map.get(&dep_id).ok_or_else(|| anyhow!("Node '{}' not found", dep_id))?;
                edges.push((*dep_idx, idx));
            }
        }
        for (src, dst) in edges { self.graph.add_edge(src, dst, ()); }
        Ok(toposort(&self.graph, None).map_err(|_| anyhow!("Cycle detected"))?)
    }

    pub fn resolve_shapes(&mut self, prog_id: &str, manifest: &Manifest, execution_order: &[NodeIndex], compiled_programs: &HashMap<String, crate::CompiledProgram>) -> anyhow::Result<()> {
        let mut changed = true;
        let mut pass = 0;
        let _params = manifest.parameters.clone().unwrap_or_default();

        while changed && pass < 20 {
            changed = false; pass += 1;
            for &idx in execution_order {
                let node_id = self.graph[idx].id.clone();
                
                // 1. Из манифеста
                let m_shape = manifest.mappings.iter()
                    .find(|m| m.program == prog_id && m.tensor == node_id)
                    .and_then(|m| match &m.source {
                        MappingSource::Link { program, output } => {
                            if program == prog_id { self.get_shape_by_id(output) }
                            else { compiled_programs.get(program).and_then(|p| p.compiler.get_shape_by_id(output)) }
                        }
                        _ => m.shape.clone(),
                    });

                // 2. Из операции
                let op_shape = self.graph[idx].op.infer_shape(|id| self.get_shape_by_id(id));
                
                // Выбираем лучшее (самое полное)
                if let Some(dims) = m_shape.or(op_shape) {
                    /* 
                    // Заменяем символы на параметры из манифеста, если они там есть
                    for d in &mut dims {
                        if let Dimension::Symbol(s) = d {
                            if let Some(&val) = params.get(s) {
                                *d = Dimension::Value(val);
                            }
                        }
                    }
                    */

                    if self.apply_shape_update(idx, dims) { changed = true; }
                }

                // 3. Обратное распространение (Constraint Solving)
                // Если мы знаем форму C в C = A + B, то мы можем обновить A и B
                // Это и есть Advanced Inference
                let current_shape = self.graph[idx].shape.dims.clone();
                if !current_shape.is_empty() && !current_shape.iter().any(|d| matches!(d, Dimension::Symbol(s) if s == "_")) {
                    for dep_id in self.graph[idx].op.get_dependencies() {
                        if let Some(&dep_idx) = self.node_map.get(&dep_id) {
                            if self.apply_shape_update(dep_idx, current_shape.clone()) { changed = true; }
                        }
                    }
                }
            }
        }

        for &idx in execution_order {
            if self.graph[idx].shape.dims.iter().any(|d| matches!(d, Dimension::Symbol(s) if s == "_")) {
                return Err(anyhow!("Тензор '{}': форма {:?} не разрешена.", self.graph[idx].id, self.graph[idx].shape.dims));
            }
        }
        Ok(())
    }

    pub fn get_shape_by_id(&self, id: &str) -> Option<Vec<Dimension>> {
        let idx = *self.node_map.get(id)?;
        let dims = &self.graph[idx].shape.dims;
        if dims.is_empty() || dims.iter().any(|d| matches!(d, Dimension::Symbol(s) if s == "_")) {
            None
        } else {
            Some(dims.clone())
        }
    }

    fn apply_shape_update(&mut self, idx: NodeIndex, new_dims: Vec<Dimension>) -> bool {
        let node = &mut self.graph[idx];
        if node.shape.dims == new_dims { return false; }

        // Если текущая форма - это просто [_], разрешаем полную замену
        if node.shape.dims.len() == 1 && matches!(node.shape.dims[0], Dimension::Symbol(ref s) if s == "_") {
            node.shape.dims = new_dims;
            return true;
        }

        if node.shape.dims.len() != new_dims.len() { return false; }

        let mut updated = false;
        for i in 0..node.shape.dims.len() {
            if matches!(node.shape.dims[i], Dimension::Symbol(ref s) if s == "_") {
                node.shape.dims[i] = new_dims[i].clone();
                updated = true;
            }
        }
        updated
    }
}
