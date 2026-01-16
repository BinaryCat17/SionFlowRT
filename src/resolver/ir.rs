use crate::core::types::{Shape, DataType, Port};
use crate::core::op::Op;
use petgraph::graph::DiGraph;

#[derive(Debug, Clone)]
pub struct ResolvedNode {
    pub id: String,
    pub op: Op,
    pub shape: Shape,
    pub dtype: DataType,
}

#[derive(Debug, Clone)]
pub struct ResolvedEdge {
    pub src_port: String,
    pub dst_port: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedIR {
    pub graph: DiGraph<ResolvedNode, ResolvedEdge>,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>, // Changed from HashMap for consistency
}