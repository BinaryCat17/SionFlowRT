use crate::core::types::{Shape, DataType, Port, WorkspaceSlot};
use crate::core::op::Op;

// ... (InputConnection and LinearNode structs)

#[derive(Debug, Clone)]
pub struct InputConnection {
    pub node_id: String,
    pub src_port: String,
    pub shape: Shape,
}

#[derive(Debug, Clone)]
pub struct LinearNode {
    pub id: String,
    pub op: Op,
    pub inputs: Vec<InputConnection>,
    pub shape: Shape,
    pub dtype: DataType,
    pub offset: usize, // Offset in elements within the workspace buffer
}

#[derive(Debug, Clone)]
pub struct LinearIR {
    pub nodes: Vec<LinearNode>,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

impl LinearIR {
    pub fn get_workspace_slots(&self) -> Vec<WorkspaceSlot> {
        self.nodes.iter()
            .filter(|n| !matches!(n.op, Op::Input { .. } | Op::Output { .. }))
            .map(|n| WorkspaceSlot { shape: n.shape.clone(), dtype: n.dtype })
            .collect()
    }
}