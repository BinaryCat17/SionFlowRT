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
                Op::Call { subgraph, inputs: call_inputs } => {
                    let first_slash = subgraph.find('/')
                        .ok_or_else(|| anyhow!("Неверный формат вызова подграфа: '{}'. Ожидается 'prefix/path/name'", subgraph))?;
                    
                    let prefix = &subgraph[..first_slash];
                    let sub_path_str = &subgraph[first_slash + 1..];

                    let sub_dir = imports.get(prefix)
                        .ok_or_else(|| anyhow!("Импорт '{}' не найден в графе", prefix))?;
                    
                    let sub_path = base_path.join(sub_dir).join(format!("{}.json", sub_path_str));
                    let sub_graph_json = fs::read_to_string(&sub_path)
                        .map_err(|e| anyhow!("Не удалось прочитать подграф '{}': {}", sub_path.display(), e))?;
                    let sub_graph = ComputationalGraph::from_json(&sub_graph_json)?;

                    if !sub_graph.nodes.iter().any(|n| matches!(n.op, Op::Output { .. })) {
                        return Err(anyhow!("Подграф '{}' должен содержать хотя бы один узел Output", sub_path.display()));
                    }

                    // Рекурсивно инлайним подграф (он уже может содержать свои инлайны)
                    let inlined_sub_nodes = self.inline_recursive(sub_graph, base_path)?;

                    // Карта соответствия: старый ID узла в подграфе -> новый ID (или ID родительского узла)
                    let mut sub_id_map: HashMap<String, String> = HashMap::new();

                    // 1. Первый проход: определяем маппинги для растворения входов и выходов
                    // Сначала входы (растворяются в родительские тензоры)
                    for sub_node in &inlined_sub_nodes {
                        if let Op::Input { name: input_name } = &sub_node.op {
                            if let Some(parent_source_id) = call_inputs.get(input_name) {
                                sub_id_map.insert(sub_node.id.clone(), parent_source_id.clone());
                            } else {
                                // Если вход не передан, он остается внутренним (с префиксом)
                                sub_id_map.insert(sub_node.id.clone(), format!("{}/{}", node.id, sub_node.id));
                            }
                        }
                    }

                    // Затем выходы (растворяются в свои входные тензоры внутри подграфа)
                    // Важно: если выход ссылается на вход, который уже растворен, маппинг сработает рекурсивно позже
                    for sub_node in &inlined_sub_nodes {
                        if let Op::Output { input: internal_input, .. } = &sub_node.op {
                            // Идентификатор, к которому ведет этот выход (пока с префиксом)
                            let source_id = sub_id_map.get(internal_input).cloned()
                                .unwrap_or_else(|| format!("{}/{}", node.id, internal_input));
                            sub_id_map.insert(sub_node.id.clone(), source_id);
                        }
                    }

                    // Для всех остальных узлов создаем префиксные ID
                    for sub_node in &inlined_sub_nodes {
                        if !sub_id_map.contains_key(&sub_node.id) {
                            sub_id_map.insert(sub_node.id.clone(), format!("{}/{}", node.id, sub_node.id));
                        }
                    }

                    // 2. Второй проход: определяем основной выход (до того как переместим узлы)
                    let main_output_source_id = inlined_sub_nodes.iter()
                        .filter_map(|n| {
                            if let Op::Output { name, input } = &n.op {
                                Some((name.clone(), input.clone()))
                            } else {
                                None
                            }
                        })
                        .find(|(name, _)| name == "default" || name == "output")
                        .map(|(_, input)| input)
                        .unwrap_or_else(|| {
                            inlined_sub_nodes.iter()
                                .filter_map(|n| if let Op::Output { input, .. } = &n.op { Some(input.clone()) } else { None })
                                .next().unwrap()
                        });

                    // 3. Третий проход: добавляем только вычислительные узлы подграфа
                    for mut sub_node in inlined_sub_nodes {
                        // Пропускаем растворенные узлы (Input и Output)
                        if matches!(sub_node.op, Op::Input { .. } | Op::Output { .. }) {
                            continue;
                        }

                        let old_id = sub_node.id.clone();
                        sub_node.id = sub_id_map.get(&old_id).unwrap().clone();

                        // Рекурсивный маппинг зависимостей
                        let map_dep = |id: &str| -> String {
                            sub_id_map.get(id).cloned().unwrap_or_else(|| id.to_string())
                        };

                        sub_node.op = match sub_node.op {
                            Op::Add { left, right } => Op::Add { left: map_dep(&left), right: map_dep(&right) },
                            Op::Sub { left, right } => Op::Sub { left: map_dep(&left), right: map_dep(&right) },
                            Op::Mul { left, right } => Op::Mul { left: map_dep(&left), right: map_dep(&right) },
                            Op::Div { left, right } => Op::Div { left: map_dep(&left), right: map_dep(&right) },
                            Op::Sin { input } => Op::Sin { input: map_dep(&input) },
                            Op::Transpose { input, permutation } => Op::Transpose { input: map_dep(&input), permutation },
                            Op::ReduceSum { input, axis } => Op::ReduceSum { input: map_dep(&input), axis },
                            Op::MatMul { left, right } => Op::MatMul { left: map_dep(&left), right: map_dep(&right) },
                            Op::Conv { input, kernel } => Op::Conv { input: map_dep(&input), kernel: map_dep(&kernel) },
                            Op::Broadcast { input } => Op::Broadcast { input: map_dep(&input) },
                            Op::Call { subgraph, inputs } => {
                                let mut new_inputs = HashMap::new();
                                for (k, v) in inputs {
                                    new_inputs.insert(k, map_dep(&v));
                                }
                                Op::Call { subgraph, inputs: new_inputs }
                            },
                            op => op,
                        };

                        result_nodes.push(sub_node);
                    }

                    // 4. Итоговый маппинг: ID вызова -> ID тензора внутри (уже с префиксом или растворенного)
                    let call_result_id = sub_id_map.get(&main_output_source_id).cloned().unwrap_or_else(|| format!("{}/{}", node.id, main_output_source_id));
                    
                    // Чтобы все последующие узлы в родительском графе, зависящие от node.id, 
                    // переключились на call_result_id, нам нужно либо добавить Identity, 
                    // либо сделать еще один проход маппинга. 
                    // Проще всего добавить Op::Output (как мост), но вы просили "растворить".
                    // Для растворения на уровне родителя нам нужно запомнить этот маппинг.
                    
                    // Добавим временный узел-заглушку, который мы растворим в самом конце inline_recursive
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

        // Финальный проход: растворяем "bridge" узлы в родительском графе
        let mut final_id_map: HashMap<String, String> = HashMap::new();
        for node in &result_nodes {
            if let Op::Output { name, input } = &node.op {
                if name == "bridge" {
                    final_id_map.insert(node.id.clone(), input.clone());
                }
            }
        }

        if final_id_map.is_empty() {
            return Ok(result_nodes);
        }

        let mut final_nodes = Vec::new();
        for mut node in result_nodes {
            if let Op::Output { name, .. } = &node.op {
                if name == "bridge" { continue; }
            }

            let map_dep = |id: &str| -> String {
                let mut current = id.to_string();
                while let Some(next) = final_id_map.get(&current) {
                    current = next.clone();
                }
                current
            };

            node.op = match node.op {
                Op::Add { left, right } => Op::Add { left: map_dep(&left), right: map_dep(&right) },
                Op::Sub { left, right } => Op::Sub { left: map_dep(&left), right: map_dep(&right) },
                Op::Mul { left, right } => Op::Mul { left: map_dep(&left), right: map_dep(&right) },
                Op::Div { left, right } => Op::Div { left: map_dep(&left), right: map_dep(&right) },
                Op::Sin { input } => Op::Sin { input: map_dep(&input) },
                Op::Transpose { input, permutation } => Op::Transpose { input: map_dep(&input), permutation },
                Op::ReduceSum { input, axis } => Op::ReduceSum { input: map_dep(&input), axis },
                Op::MatMul { left, right } => Op::MatMul { left: map_dep(&left), right: map_dep(&right) },
                Op::Conv { input, kernel } => Op::Conv { input: map_dep(&input), kernel: map_dep(&kernel) },
                Op::Broadcast { input } => Op::Broadcast { input: map_dep(&input) },
                Op::Output { name, input } => Op::Output { name, input: map_dep(&input) },
                Op::Call { subgraph, inputs } => {
                    let mut new_inputs = HashMap::new();
                    for (k, v) in inputs {
                        new_inputs.insert(k, map_dep(&v));
                    }
                    Op::Call { subgraph, inputs: new_inputs }
                },
                op => op,
            };
            final_nodes.push(node);
        }

        Ok(final_nodes)
    }

    pub fn build(&mut self, source: ComputationalGraph) -> anyhow::Result<Vec<NodeIndex>> {
        // Выполняем инлайнинг перед построением графа
        // Путь считаем относительно корня проекта, так как в манифесте пути от корня
        let inlined_nodes = self.inline_recursive(source, Path::new("."))?;

        if !inlined_nodes.iter().any(|n| matches!(n.op, Op::Output { .. })) {
            return Err(anyhow!("Граф не содержит ни одного узла Output. Вычисления бессмысленны."));
        }

        for node in inlined_nodes {
            let id = node.id.clone();
            let idx = self.graph.add_node(node);
            self.node_map.insert(id, idx);
        }

        let mut edges = Vec::new();
        for &idx in self.graph.node_indices().collect::<Vec<_>>().iter() {
            let node = &self.graph[idx];
            for dep_id in node.op.get_dependencies() {
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
                            MappingSource::Link { program, output } => {
                                if let Some(source_prog) = compiled_programs.get(program) {
                                    source_prog.compiler.graph.node_indices()
                                        .find(|&i| source_prog.compiler.graph[i].id == *output)
                                        .map(|i| source_prog.compiler.graph[i].shape.dims.clone())
                                } else if program == prog_id {
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
                            _ => m.shape.clone(),
                        };
                    }

                    if new_dims.is_none() {
                        new_dims = node.op.infer_shape(|dep_id| {
                            let dep_idx = self.node_map.get(dep_id)?;
                            let dims = &self.graph[*dep_idx].shape.dims;
                            if dims.iter().any(|d| matches!(d, Dimension::Symbol(s) if s == "_")) {
                                None
                            } else {
                                Some(dims.clone())
                            }
                        });
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
        
        // Если у нас ранг 1 с символом "_", разрешаем полную замену формы (изменение ранга)
        if node.shape.dims.len() == 1 && matches!(node.shape.dims[0], Dimension::Symbol(ref s) if s == "_") {
            if node.shape.dims != dims {
                node.shape.dims = dims;
                return true;
            }
            return false;
        }

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
}