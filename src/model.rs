use serde::{Deserialize, Serialize};
use std::fmt;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum Dimension {
    Value(usize),
    Symbol(String),
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dimension::Value(v) => write!(f, "{}", v),
            Dimension::Symbol(s) => write!(f, "{}", s),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum DataType {
    F32,
    I32,
    U32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TensorShape {
    pub dims: Vec<Dimension>,
}

impl TensorShape {
    pub fn size_c_expr(&self) -> String {
        if self.dims.is_empty() {
            "1".to_string()
        } else {
            self.dims.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(" * ")
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn default_strides_c_expr(&self) -> Vec<String> {
        let mut strides = vec!["1".to_string(); self.dims.len()];
        let mut current = "1".to_string();
        for i in (0..self.dims.len()).rev() {
            strides[i] = current.clone();
            if i > 0 {
                current = format!("{} * ({})", current, self.dims[i]);
            }
        }
        strides
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Op {
    Input { name: String },
    Constant { values: Vec<f32> },
    Add { left: String, right: String },
    Sub { left: String, right: String },
    Mul { left: String, right: String },
    Div { left: String, right: String },
    Sin { input: String },
    Transpose { input: String, permutation: Vec<usize> },
    ReduceSum { input: String, axis: usize },
    MatMul { left: String, right: String },
    Conv { input: String, kernel: String },
    Call { subgraph: String, inputs: HashMap<String, String> },
    Output { name: String, input: String },
    Broadcast { input: String },
}

impl Op {
    pub fn get_dependencies(&self) -> Vec<String> {
        match self {
            Op::Input { .. } => vec![],
            Op::Constant { .. } => vec![],
            Op::Add { left, right } => vec![left.clone(), right.clone()],
            Op::Sub { left, right } => vec![left.clone(), right.clone()],
            Op::Mul { left, right } => vec![left.clone(), right.clone()],
            Op::Div { left, right } => vec![left.clone(), right.clone()],
            Op::Sin { input } => vec![input.clone()],
            Op::Transpose { input, .. } => vec![input.clone()],
            Op::ReduceSum { input, .. } => vec![input.clone()],
            Op::MatMul { left, right } => vec![left.clone(), right.clone()],
            Op::Conv { input, kernel } => vec![input.clone(), kernel.clone()],
            Op::Call { inputs, .. } => inputs.values().cloned().collect(),
            Op::Output { input, .. } => vec![input.clone()],
            Op::Broadcast { input } => vec![input.clone()],
        }
    }

    pub fn infer_shape(&self, get_node_shape: impl Fn(&str) -> Option<Vec<Dimension>>) -> Option<Vec<Dimension>> {
        match self {
            Op::Add { left, .. } | Op::Sub { left, .. } | Op::Mul { left, .. } | Op::Div { left, .. } | Op::Sin { input: left } | Op::Output { input: left, .. } | Op::Broadcast { input: left } => {
                get_node_shape(left)
            }
            Op::Transpose { input, permutation } => {
                get_node_shape(input).map(|dims| {
                    permutation.iter().map(|&i| dims[i].clone()).collect()
                })
            }
            Op::ReduceSum { input, axis } => {
                get_node_shape(input).map(|mut dims| {
                    if *axis < dims.len() {
                        dims.remove(*axis);
                    }
                    dims
                })
            }
            Op::MatMul { left, right } => {
                let left_dims = get_node_shape(left)?;
                let right_dims = get_node_shape(right)?;
                if left_dims.len() == 2 && right_dims.len() == 2 {
                    Some(vec![left_dims[0].clone(), right_dims[1].clone()])
                } else {
                    None
                }
            }
            Op::Conv { input, kernel: _ } => {
                // Упрощенный вывод для свертки (сохраняем ранг)
                get_node_shape(input)
            }
            _ => None,
        }
    }

    pub fn generate_c_body(
        &self, 
        prog_id: &str, 
        node_id: &str, 
        target_idx: &str, 
        get_index_expr: impl Fn(&str) -> String
    ) -> String {
        let s = |id: &str| id.replace("/", "__");
        match self {
            Op::Add { left, right } => {
                format!("buffer_{}_{}[{}] = buffer_{}_{}[{}] + buffer_{}_{}[{}];", 
                    prog_id, s(node_id), target_idx, prog_id, s(left), get_index_expr(left), prog_id, s(right), get_index_expr(right))
            }
            Op::Sub { left, right } => {
                format!("buffer_{}_{}[{}] = buffer_{}_{}[{}] - buffer_{}_{}[{}];", 
                    prog_id, s(node_id), target_idx, prog_id, s(left), get_index_expr(left), prog_id, s(right), get_index_expr(right))
            }
            Op::Mul { left, right } => {
                format!("buffer_{}_{}[{}] = buffer_{}_{}[{}] * buffer_{}_{}[{}];", 
                    prog_id, s(node_id), target_idx, prog_id, s(left), get_index_expr(left), prog_id, s(right), get_index_expr(right))
            }
            Op::Div { left, right } => {
                format!("buffer_{}_{}[{}] = buffer_{}_{}[{}] / (buffer_{}_{}[{}] + 1e-9f);", 
                    prog_id, s(node_id), target_idx, prog_id, s(left), get_index_expr(left), prog_id, s(right), get_index_expr(right))
            }
            Op::Sin { input } => {
                format!("buffer_{}_{}[{}] = sinf(buffer_{}_{}[{}]);", 
                    prog_id, s(node_id), target_idx, prog_id, s(input), get_index_expr(input))
            }
            Op::Transpose { input, permutation: _ } => {
                format!("buffer_{}_{}[{}] = buffer_{}_{}[{}];", 
                    prog_id, s(node_id), target_idx, prog_id, s(input), get_index_expr(input))
            }
            Op::Output { input, .. } => {
                format!("buffer_{}_{}[{}] = buffer_{}_{}[{}];", 
                    prog_id, s(node_id), target_idx, prog_id, s(input), get_index_expr(input))
            }
            Op::Broadcast { input } => {
                format!("buffer_{}_{}[{}] = buffer_{}_{}[{}];", 
                    prog_id, s(node_id), target_idx, prog_id, s(input), get_index_expr(input))
            }
            _ => "".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Node {
    pub id: String,
    pub op: Op,
    pub shape: TensorShape,
    pub dtype: DataType,
    /// Если None, используются default_strides_c_expr()
    pub strides: Option<Vec<String>>,
}

impl Node {
    pub fn get_effective_strides_c_expr(&self) -> Vec<String> {
        self.strides.clone().unwrap_or_else(|| self.shape.default_strides_c_expr())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ComputationalGraph {
    pub imports: Option<std::collections::HashMap<String, String>>,
    pub nodes: Vec<Node>,
}

impl ComputationalGraph {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}