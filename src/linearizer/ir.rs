use crate::core::types::{Shape, DataType, Port};
use crate::core::op::Op;

#[derive(Debug, Clone)]
pub struct LinearNode {
    pub id: String,
    pub op: Op,
    pub inputs: Vec<(String, String)>, // (ID узла, Имя порта)
    pub shape: Shape,
    pub dtype: DataType,
}

#[derive(Debug, Clone)]
pub struct LinearIR {
    pub nodes: Vec<LinearNode>,
    pub inputs: Vec<Port>,
    pub outputs: std::collections::HashMap<String, String>, // Внешнее имя -> ID узла
}