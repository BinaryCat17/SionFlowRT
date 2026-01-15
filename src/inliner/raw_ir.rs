use petgraph::graph::DiGraph;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct RawNode {
    pub id: String,
    pub op: Value,
}

#[derive(Debug, Clone)]
pub struct RawEdge {
    pub src_port: String,
    pub dst_port: String,
}

#[derive(Debug, Clone)]
pub struct RawIR {
    pub graph: DiGraph<RawNode, RawEdge>,
    pub inputs: Vec<crate::inliner::json::JsonPort>,
    pub outputs: Vec<crate::inliner::json::JsonPort>,
}

impl RawIR {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}