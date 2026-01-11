use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Port {
    pub name: String,
    pub dtype: String,
    pub shape: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInterface {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

#[derive(Debug, Clone)]
pub enum Component<P> {
    Primitive(P),
    Subgraph(LogicalGraph<P>),
    Input,
    Output,
}

#[derive(Debug, Clone)]
pub struct LogicalNode<P> {
    pub id: String,
    pub component: Component<P>,
    pub interface: NodeInterface,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub src_port: String,
    pub dst_port: String,
}

#[derive(Debug, Clone)]
pub struct LogicalGraph<P> {
    pub graph: DiGraph<LogicalNode<P>, Connection>,
    pub node_map: HashMap<String, NodeIndex>,
}

impl<P> Default for LogicalGraph<P> {
    fn default() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphDef<P> {
    pub imports: Option<HashMap<String, String>>,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    pub nodes: Vec<NodeDef<P>>,
    pub links: Vec<LinkDef>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeDef<P> {
    pub id: String,
    pub subgraph: Option<String>,
    pub op: Option<P>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LinkDef(pub (String, String));

#[derive(Debug, Clone)]
pub struct InlineResult<P> {
    pub graph: DiGraph<InlinedNode<P>, usize>,
    pub outputs: HashMap<String, String>,
}

impl<P: Clone + for<'de> Deserialize<'de> + Serialize> LogicalGraph<P> {
    pub fn from_json(
        json: &str, 
        load_subgraph: impl Fn(&str) -> anyhow::Result<LogicalGraph<P>>,
        resolve_interface: impl Fn(&P) -> NodeInterface
    ) -> anyhow::Result<Self> {
        let def: GraphDef<P> = serde_json::from_str(json)?;
        let mut l_graph = LogicalGraph::default();
        let mut port_addresses = HashMap::new();

        // 1. Регистрируем входы графа под именем inputs.NAME
        for in_p in def.inputs {
            let interface = NodeInterface { inputs: vec![], outputs: vec![in_p.clone()] };
            let idx = l_graph.add_node(&in_p.name, Component::Input, interface);
            port_addresses.insert(format!("inputs.{}", in_p.name), (idx, in_p.name.clone()));
        }

        // 2. Регистрируем выходы графа (пока просто добавляем узлы)
        for out_p in def.outputs {
            let interface = NodeInterface { inputs: vec![Port { name: "value".into(), ..out_p.clone() }], outputs: vec![] };
            l_graph.add_node(&out_p.name, Component::Output, interface);
        }

        // 3. Регистрируем узлы (примитивы и сабграфы)
        for n_def in def.nodes {
            if let Some(sub_path_raw) = n_def.subgraph {
                let mut actual_path = sub_path_raw.clone();
                if let Some(imports) = &def.imports {
                    for (prefix, target) in imports {
                        if sub_path_raw.starts_with(prefix) {
                            actual_path = sub_path_raw.replace(prefix, target);
                            break;
                        }
                    }
                }
                let sub = load_subgraph(&actual_path)?;
                let (sub_in, sub_out) = sub.get_interface();
                let idx = l_graph.add_node(&n_def.id, Component::Subgraph(sub), NodeInterface { inputs: sub_in, outputs: sub_out.clone() });
                for p in sub_out {
                    // Порты сабграфа доступны как NODE.PORT
                    port_addresses.insert(format!("{}.{}", n_def.id, p.name), (idx, p.name));
                }
            } else if let Some(payload) = n_def.op {
                let interface = resolve_interface(&payload);
                let out_ports = interface.outputs.clone();
                let idx = l_graph.add_node(&n_def.id, Component::Primitive(payload), interface);
                for p in out_ports {
                    // Порты примитива доступны как NODE.PORT
                    port_addresses.insert(format!("{}.{}", n_def.id, p.name), (idx, p.name));
                }
            }
        }

        // 4. Разрешаем линки
        for link_def in def.links {
            let (from, to) = &link_def.0;
            let &(src_idx, ref src_port) = port_addresses.get(from).ok_or_else(|| anyhow::anyhow!("Source port not found: {}", from))?;
            
            // Если цель - это outputs.NAME
            if let Some(target_name) = to.strip_prefix("outputs.") {
                if let Some(dst_idx) = l_graph.node_map.get(target_name) {
                    l_graph.graph.add_edge(src_idx, *dst_idx, Connection { src_port: src_port.clone(), dst_port: "value".into() });
                    continue;
                }
            }

            // Иначе цель - это NODE.PORT
            let (dst_id, dst_port) = to.split_once('.').ok_or_else(|| anyhow::anyhow!("Invalid destination port: {}", to))?;
            let dst_idx = *l_graph.node_map.get(dst_id).ok_or_else(|| anyhow::anyhow!("Destination node not found: {}", dst_id))?;
            l_graph.graph.add_edge(src_idx, dst_idx, Connection { src_port: src_port.clone(), dst_port: dst_port.to_string() });
        }

        Ok(l_graph)
    }

    fn add_node(&mut self, id: &str, component: Component<P>, interface: NodeInterface) -> NodeIndex {
        let node = LogicalNode { id: id.to_string(), component, interface };
        let idx = self.graph.add_node(node);
        self.node_map.insert(id.to_string(), idx);
        idx
    }

    fn get_interface(&self) -> (Vec<Port>, Vec<Port>) {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for idx in self.graph.node_indices() {
            let node = &self.graph[idx];
            match &node.component {
                Component::Input => { 
                    for p in &node.interface.outputs { 
                        inputs.push(Port { name: node.id.clone(), ..p.clone() }); 
                    } 
                }
                Component::Output => { 
                    for p in &node.interface.inputs { 
                        outputs.push(Port { name: node.id.clone(), ..p.clone() }); 
                    } 
                }
                _ => {}
            }
        }
        (inputs, outputs)
    }

    pub fn inline(&self) -> InlineResult<P> {
        let mut flat_graph = DiGraph::new();
        let outputs = self.inline_recursive("", &mut flat_graph, &HashMap::new());
        
        let mut final_outputs = HashMap::new();
        for (name, idx) in outputs {
            final_outputs.insert(name, flat_graph[idx].id.clone());
        }

        InlineResult { graph: flat_graph, outputs: final_outputs }
    }

    fn inline_recursive(
        &self, 
        prefix: &str, 
        result_graph: &mut DiGraph<InlinedNode<P>, usize>,
        input_mappings: &HashMap<String, NodeIndex>
    ) -> HashMap<String, NodeIndex> {
        let mut current_node_map = HashMap::new();
        let mut output_mappings = HashMap::new();

        for idx in self.graph.node_indices() {
            let node = &self.graph[idx];
            if let Component::Input = &node.component {
                if let Some(&ext_idx) = input_mappings.get(&node.id) {
                    current_node_map.insert(node.id.clone(), ext_idx);
                    current_node_map.insert(format!("inputs.{}", node.id), ext_idx);
                } else if prefix.is_empty() {
                    let new_idx = result_graph.add_node(InlinedNode {
                        id: node.id.clone(),
                        payload: InlinedPayload::Input,
                        dtype: node.interface.outputs.get(0).map(|p| p.dtype.clone()),
                        shape: node.interface.outputs.get(0).map(|p| p.shape.clone()),
                    });
                    current_node_map.insert(node.id.clone(), new_idx);
                    current_node_map.insert(format!("inputs.{}", node.id), new_idx);
                }
            }
        }

        let order = petgraph::algo::toposort(&self.graph, None).unwrap_or_default();
        for idx in order {
            let node = &self.graph[idx];
            let full_id = if prefix.is_empty() { node.id.clone() } else { format!("{}/{}", prefix, node.id) };

            match &node.component {
                Component::Primitive(p) => {
                    let new_idx = result_graph.add_node(InlinedNode {
                        id: full_id,
                        payload: InlinedPayload::Primitive(p.clone()),
                        dtype: node.interface.outputs.get(0).map(|p| p.dtype.clone()),
                        shape: node.interface.outputs.get(0).map(|p| p.shape.clone()),
                    });
                    current_node_map.insert(node.id.clone(), new_idx);

                    for edge in self.graph.edges_directed(idx, petgraph::Direction::Incoming) {
                        let src_node_idx = edge.source();
                        let src_node_id = &self.graph[src_node_idx].id;
                        let lookup_key = format!("{}.{}", src_node_id, edge.weight().src_port);
                        
                        let src_flat_idx = current_node_map.get(&lookup_key)
                            .or_else(|| current_node_map.get(src_node_id))
                            .or_else(|| current_node_map.get(&format!("inputs.{}", src_node_id)));

                        if let Some(&flat_idx) = src_flat_idx {
                            let port_idx = node.interface.inputs.iter().position(|p| p.name == edge.weight().dst_port).unwrap_or(0);
                            result_graph.add_edge(flat_idx, new_idx, port_idx);
                        }
                    }
                }
                Component::Subgraph(sub) => {
                    let mut sub_inputs = HashMap::new();
                    for edge in self.graph.edges_directed(idx, petgraph::Direction::Incoming) {
                        let src_node_idx = edge.source();
                        let src_node_id = &self.graph[src_node_idx].id;
                        let lookup_key = format!("{}.{}", src_node_id, edge.weight().src_port);
                        let src_flat_idx = current_node_map.get(&lookup_key)
                            .or_else(|| current_node_map.get(src_node_id))
                            .or_else(|| current_node_map.get(&format!("inputs.{}", src_node_id)));

                        if let Some(&flat_idx) = src_flat_idx {
                            sub_inputs.insert(edge.weight().dst_port.clone(), flat_idx);
                        }
                    }
                    
                    let sub_outputs = sub.inline_recursive(&full_id, result_graph, &sub_inputs);
                    for (out_name, out_idx) in sub_outputs {
                        current_node_map.insert(format!("{}.{}", node.id, out_name), out_idx);
                    }
                }
                Component::Output => {
                    for edge in self.graph.edges_directed(idx, petgraph::Direction::Incoming) {
                        let src_node_idx = edge.source();
                        let src_node_id = &self.graph[src_node_idx].id;
                        let lookup_key = format!("{}.{}", src_node_id, edge.weight().src_port);
                        let src_flat_idx = current_node_map.get(&lookup_key)
                            .or_else(|| current_node_map.get(src_node_id))
                            .or_else(|| current_node_map.get(&format!("inputs.{}", src_node_id)));

                        if let Some(&flat_idx) = src_flat_idx {
                            output_mappings.insert(node.id.clone(), flat_idx);
                            output_mappings.insert(format!("outputs.{}", node.id), flat_idx);
                        }
                    }
                }
                _ => {}
            }
        }
        output_mappings
    }
}

#[derive(Debug, Clone)]
pub struct InlinedNode<P> {
    pub id: String,
    pub payload: InlinedPayload<P>,
    pub dtype: Option<String>,
    pub shape: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum InlinedPayload<P> {
    Primitive(P),
    Input,
}
