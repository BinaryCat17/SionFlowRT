use serde::{Deserialize, Serialize};
use crate::core::types::Dim;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Op {
    // Unary
    Sin, Abs, Sqrt, Square, Exp, Log,
    // Binary
    Add, Sub, Mul, Div, Min, Max, Pow,
    // Special
    Input { name: String },
    Constant { values: Vec<f32> },
    Transpose { permutation: Vec<usize> },
    ReduceSum { axis: usize },
    MatMul,
    Output { name: String },
    Reshape { new_shape: Vec<Dim> },
}

impl Op {
    pub fn from_json(name: &str, params: serde_json::Value) -> anyhow::Result<Self> {
        match name {
            "Sin" => Ok(Op::Sin),
            "Abs" => Ok(Op::Abs),
            "Sqrt" => Ok(Op::Sqrt),
            "Square" => Ok(Op::Square),
            "Exp" => Ok(Op::Exp),
            "Log" => Ok(Op::Log),
            "Add" => Ok(Op::Add),
            "Sub" => Ok(Op::Sub),
            "Mul" => Ok(Op::Mul),
            "Div" => Ok(Op::Div),
            "Min" => Ok(Op::Min),
            "Max" => Ok(Op::Max),
            "Pow" => Ok(Op::Pow),
            "MatMul" => Ok(Op::MatMul),
            "Reshape" => {
                let new_shape: Vec<Dim> = serde_json::from_value(params.get("new_shape").cloned().unwrap_or_default())?;
                Ok(Op::Reshape { new_shape })
            }
            "ReduceSum" => {
                let axis = params.get("axis").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                Ok(Op::ReduceSum { axis })
            }
            "Constant" => {
                let values: Vec<f32> = serde_json::from_value(params.get("values").cloned().unwrap_or_default())?;
                Ok(Op::Constant { values })
            }
            "Input" => {
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                Ok(Op::Input { name })
            }
            "Output" => {
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                Ok(Op::Output { name })
            }
            _ => Err(anyhow::anyhow!("Unknown op: {}", name)),
        }
    }
}