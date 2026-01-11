use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Port {
    pub name: String,
    #[serde(alias = "dtype")]
    pub dtype_id: String,
    pub shape: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInterface {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogicalGraph<P> {
    #[serde(skip)]
    pub graph: DiGraph<LogicalNode<P>, Connection>,
    #[serde(skip)]
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

impl<P: Clone + for<'de> Deserialize<'de> + Serialize> LogicalGraph<P> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_json(
        json: &str, 
        load_subgraph: impl Fn(&str) -> anyhow::Result<LogicalGraph<P>>,
        resolve_interface: impl Fn(&P) -> NodeInterface
    ) -> anyhow::Result<Self> {
        let def: GraphDef<P> = serde_json::from_str(json)?;
        let mut graph = LogicalGraph::new();
        let mut port_addresses = HashMap::new();

        for in_p in def.inputs {
            let idx = graph.add_node(&in_p.name, Component::Input, NodeInterface { inputs: vec![], outputs: vec![in_p.clone()] });
            port_addresses.insert(in_p.name.clone(), PortAddress { node_idx: idx, port_name: in_p.name.clone() });
        }

        for out_p in &def.outputs {
            graph.add_node(&out_p.name, Component::Output, NodeInterface { inputs: vec![Port { name: "value".into(), ..out_p.clone() }], outputs: vec![] });
        }

        for n_def in def.nodes {
            if let Some(subgraph) = n_def.subgraph {
                let mut actual_path = subgraph.clone();
                if let Some(imports) = &def.imports {
                    for (prefix, target) in imports {
                        if subgraph.starts_with(prefix) {
                            actual_path = subgraph.replace(prefix, target);
                            break;
                        }
                    }
                }
                let sub = load_subgraph(&actual_path)?;
                let (sub_in, sub_out) = Self::get_subgraph_interface(&sub);
                let idx = graph.add_node(&n_def.id, Component::Subgraph(sub), NodeInterface { inputs: sub_in, outputs: sub_out.clone() });
                for p in sub_out {
                    port_addresses.insert(format!("{}.{}", n_def.id, p.name), PortAddress { node_idx: idx, port_name: p.name });
                }
            } else if let Some(payload) = n_def.op {
                let interface = resolve_interface(&payload);
                let out_ports = interface.outputs.clone();
                let idx = graph.add_node(&n_def.id, Component::Primitive(payload), interface);
                for p in out_ports {
                    port_addresses.insert(format!("{}.{}", n_def.id, p.name), PortAddress { node_idx: idx, port_name: p.name });
                }
            }
        }

        for link_def in def.links {
            let (from, to) = &link_def.0;
            let src_addr = port_addresses.get(from).ok_or_else(|| anyhow::anyhow!("Source port not found: {}", from))?;
            if let Some(out_idx) = graph.node_map.get(to).filter(|&&idx| matches!(graph.graph[idx].component, Component::Output)) {
                graph.connect(src_addr.node_idx, &src_addr.port_name, *out_idx, "value")?;
            } else {
                let (dst_id, dst_p) = to.split_once('.').ok_or_else(|| anyhow::anyhow!("Invalid destination port: {}", to))?;
                let dst_idx = *graph.node_map.get(dst_id).ok_or_else(|| anyhow::anyhow!("Destination node not found: {}", dst_id))?;
                graph.connect(src_addr.node_idx, &src_addr.port_name, dst_idx, dst_p)?;
            }
        }
        Ok(graph)
    }

    fn get_subgraph_interface(sub: &LogicalGraph<P>) -> (Vec<Port>, Vec<Port>) {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for idx in sub.graph.node_indices() {
            let node = &sub.graph[idx];
            match &node.component {
                Component::Input => { for p in &node.interface.outputs { inputs.push(Port { name: node.id.clone(), ..p.clone() }); } }
                Component::Output => { for p in &node.interface.inputs { outputs.push(Port { name: node.id.clone(), ..p.clone() }); } }
                _ => {}
            }
        }
        (inputs, outputs)
    }

    pub fn add_node(&mut self, id: &str, component: Component<P>, interface: NodeInterface) -> NodeIndex {
        let node = LogicalNode { id: id.to_string(), component, interface };
        let idx = self.graph.add_node(node);
        self.node_map.insert(id.to_string(), idx);
        idx
    }

    pub fn connect(&mut self, src: NodeIndex, src_p: &str, dst: NodeIndex, dst_p: &str) -> anyhow::Result<()> {
        self.graph.add_edge(src, dst, Connection { src_port: src_p.to_string(), dst_port: dst_p.to_string() });
        Ok(())
    }

    pub fn flatten(&self) -> anyhow::Result<Vec<FlatNodeRecord<P>>> {
        let mut nodes = Vec::new();
        let mut p_map = HashMap::new();
        self.flatten_recursive("", &mut nodes, &mut p_map, &HashMap::new())?;
        Ok(nodes)
    }

    fn flatten_recursive(&self, prefix: &str, result: &mut Vec<FlatNodeRecord<P>>, p_map: &mut HashMap<(String, String, String), String>, ext_ctx: &HashMap<String, String>) -> anyhow::Result<()> {
        let order = petgraph::algo::toposort(&self.graph, None).map_err(|_| anyhow::anyhow!("Cycle detected in logical graph: {}", prefix))?;
        for idx in order {
            let node = &self.graph[idx];
            let full_id = if prefix.is_empty() { node.id.clone() } else { format!("{}/{}", prefix, node.id) };
            match &node.component {
                Component::Input => {
                    let id = ext_ctx.get(&node.id).cloned().unwrap_or_else(|| node.id.clone());
                    for p in &node.interface.outputs { p_map.insert((prefix.to_string(), node.id.clone(), p.name.clone()), id.clone()); }
                    if prefix.is_empty() {
                        result.push(FlatNodeRecord { id: node.id.clone(), payload: None, inputs: vec![], interface: node.interface.clone(), is_input: true, is_output: false });
                    }
                }
                Component::Primitive(payload) => {
                    let mut inputs = Vec::new();
                    for in_p in &node.interface.inputs {
                        let id = self.graph.edges_directed(idx, Direction::Incoming).find(|e| e.weight().dst_port == in_p.name)
                            .map(|e| p_map.get(&(prefix.to_string(), self.graph[e.source()].id.clone(), e.weight().src_port.clone())).cloned().unwrap())
                            .unwrap_or_else(|| in_p.name.clone());
                        inputs.push(id);
                    }
                    result.push(FlatNodeRecord { id: full_id.clone(), payload: Some(payload.clone()), inputs, interface: node.interface.clone(), is_input: false, is_output: false });
                    for p in &node.interface.outputs { p_map.insert((prefix.to_string(), node.id.clone(), p.name.clone()), full_id.clone()); }
                }
                Component::Subgraph(sub) => {
                    let mut sub_ctx = HashMap::new();
                    for in_p in &node.interface.inputs {
                        if let Some(e) = self.graph.edges_directed(idx, Direction::Incoming).find(|e| e.weight().dst_port == in_p.name) {
                            if let Some(id) = p_map.get(&(prefix.to_string(), self.graph[e.source()].id.clone(), e.weight().src_port.clone())) {
                                sub_ctx.insert(in_p.name.clone(), id.clone());
                            }
                        }
                    }
                    sub.flatten_recursive(&full_id, result, p_map, &sub_ctx)?;
                    for p in &node.interface.outputs {
                        let sub_out_id = p_map.get(&(full_id.clone(), p.name.clone(), "value".to_string())).cloned();
                        if let Some(id) = sub_out_id { p_map.insert((prefix.to_string(), node.id.clone(), p.name.clone()), id); }
                    }
                }
                Component::Output => {
                    if let Some(e) = self.graph.edges_directed(idx, Direction::Incoming).next() {
                        let src_id = p_map.get(&(prefix.to_string(), self.graph[e.source()].id.clone(), e.weight().src_port.clone())).cloned();
                        if let Some(id) = src_id { 
                            p_map.insert((prefix.to_string(), node.id.clone(), "value".to_string()), id.clone()); 
                            if prefix.is_empty() {
                                result.push(FlatNodeRecord { id: node.id.clone(), payload: None, inputs: vec![id], interface: node.interface.clone(), is_input: false, is_output: true });
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FlatNodeRecord<P> {
    pub id: String,
    pub payload: Option<P>,
    pub inputs: Vec<String>,
    pub interface: NodeInterface,
    pub is_input: bool,
    pub is_output: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PortAddress {
    pub node_idx: NodeIndex,
    pub port_name: String,
}