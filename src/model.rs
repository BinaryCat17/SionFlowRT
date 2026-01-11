use serde::{Deserialize, Serialize};
use std::fmt;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum Dimension {
    Value(usize),
    #[serde(rename = "_")]
    Wildcard,
    #[serde(rename = "...")]
    Ellipsis,
    Symbol(String),
    Add(Box<Dimension>, Box<Dimension>),
    Sub(Box<Dimension>, Box<Dimension>),
    Mul(Box<Dimension>, Box<Dimension>),
    Div(Box<Dimension>, Box<Dimension>),
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dimension::Value(v) => write!(f, "{}", v),
            Dimension::Symbol(s) => write!(f, "{}", s),
            Dimension::Wildcard => write!(f, "_"),
            Dimension::Ellipsis => write!(f, "..."),
            Dimension::Add(l, r) => write!(f, "({} + {})", l, r),
            Dimension::Sub(l, r) => write!(f, "({} - {})", l, r),
            Dimension::Mul(l, r) => write!(f, "({} * {})", l, r),
            Dimension::Div(l, r) => write!(f, "({} / {})", l, r),
        }
    }
}

impl Dimension {
    pub fn is_ellipsis(&self) -> bool {
        match self {
            Dimension::Ellipsis => true,
            Dimension::Symbol(s) if s == "..." => true,
            _ => false,
        }
    }

    pub fn is_wildcard(&self) -> bool {
        match self {
            Dimension::Wildcard => true,
            Dimension::Symbol(s) if s == "_" => true,
            _ => false,
        }
    }

    pub fn eval(&self, params: &HashMap<String, usize>) -> Dimension {
        match self {
            Dimension::Value(v) => Dimension::Value(*v),
            Dimension::Wildcard => Dimension::Wildcard,
            Dimension::Ellipsis => Dimension::Ellipsis,
            Dimension::Symbol(s) => {
                if let Some(&v) = params.get(s) { Dimension::Value(v) }
                else { Dimension::Symbol(s.clone()) }
            },
            Dimension::Add(l, r) => {
                match (l.eval(params), r.eval(params)) {
                    (Dimension::Value(lv), Dimension::Value(rv)) => Dimension::Value(lv + rv),
                    (le, re) => Dimension::Add(Box::new(le), Box::new(re)),
                }
            },
            Dimension::Sub(l, r) => {
                match (l.eval(params), r.eval(params)) {
                    (Dimension::Value(lv), Dimension::Value(rv)) => Dimension::Value(lv.saturating_sub(rv)),
                    (le, re) => Dimension::Sub(Box::new(le), Box::new(re)),
                }
            },
            Dimension::Mul(l, r) => {
                match (l.eval(params), r.eval(params)) {
                    (Dimension::Value(lv), Dimension::Value(rv)) => Dimension::Value(lv * rv),
                    (le, re) => Dimension::Mul(Box::new(le), Box::new(re)),
                }
            },
            Dimension::Div(l, r) => {
                match (l.eval(params), r.eval(params)) {
                    (Dimension::Value(lv), Dimension::Value(rv)) if rv != 0 => Dimension::Value(lv / rv),
                    (le, re) => Dimension::Div(Box::new(le), Box::new(re)),
                }
            },
        }
    }
}

pub fn sanitize_id(id: &str) -> String {
    id.replace("/", "__").replace(".", "__")
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32, F64, I32, I64, U32,
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(transparent)]
pub struct TensorShape {
    pub dims: Vec<Dimension>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum Op {
    // Unary
    Sin, Abs, Sqrt, Square, Exp, Log,
    // Binary
    Add, Sub, Mul, Div, Min, Max, Pow,
    // Special
    Input { name: String },
    Constant { values: Vec<f32> },
    Transpose { permutation: Vec<usize> },
    ReduceSum { axis: isize },
    MatMul,
    Conv,
    Output { name: String },
    Broadcast,
    Reshape { new_shape: Vec<Dimension> },
}
