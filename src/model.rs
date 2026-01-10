use serde::{Deserialize, Serialize};
use std::fmt;

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
    Mul { left: String, right: String },
    Sin { input: String },
    Transpose { input: String, permutation: Vec<usize> },
    ReduceSum { input: String, axis: usize },
    MatMul { left: String, right: String },
    Conv { input: String, kernel: String },
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
    pub nodes: Vec<Node>,
}

impl ComputationalGraph {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}
