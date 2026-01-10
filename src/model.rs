use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum DataType {
    F32,
    I32,
    U32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

impl TensorShape {
    pub fn size(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn default_strides(&self) -> Vec<usize> {
        let mut strides = vec![0; self.dims.len()];
        let mut current = 1;
        for i in (0..self.dims.len()).rev() {
            strides[i] = current;
            current *= self.dims[i];
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
    /// Если None, используются default_strides()
    pub strides: Option<Vec<usize>>,
}

impl Node {
    pub fn get_effective_strides(&self) -> Vec<usize> {
        self.strides.clone().unwrap_or_else(|| self.shape.default_strides())
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
