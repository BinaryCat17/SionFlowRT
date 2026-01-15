use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    U32,
}

impl DataType {
    pub fn to_c_type(&self) -> &'static str {
        match self {
            DataType::F32 => "float",
            DataType::F64 => "double",
            DataType::I32 => "int32_t",
            DataType::I64 => "int64_t",
            DataType::U32 => "uint32_t",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum Dim {
    Static(usize),
    Variable(String),
}

impl Dim {
    pub fn to_c_expr(&self) -> String {
        match self {
            Dim::Static(v) => v.to_string(),
            Dim::Variable(s) => s.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    pub dims: Vec<Dim>,
}

impl Shape {
    pub fn to_c_size_expr(&self) -> String {
        if self.dims.is_empty() {
            return "1".to_string();
        }
        self.dims
            .iter()
            .map(|d| d.to_c_expr())
            .collect::<Vec<_>>()
            .join(" * ")
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct Port {
    pub name: String,
    pub shape: Shape,
    pub dtype: DataType,
}