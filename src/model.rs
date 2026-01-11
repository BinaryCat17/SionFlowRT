use serde::{Deserialize, Serialize};
use std::fmt;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum Dimension {
    Value(usize),
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
            Dimension::Add(l, r) => write!(f, "({} + {})", l, r),
            Dimension::Sub(l, r) => write!(f, "({} - {})", l, r),
            Dimension::Mul(l, r) => write!(f, "({} * {})", l, r),
            Dimension::Div(l, r) => write!(f, "({} / {})", l, r),
        }
    }
}

impl Dimension {
    pub fn eval(&self, params: &HashMap<String, usize>) -> Dimension {
        match self {
            Dimension::Value(v) => Dimension::Value(*v),
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
    id.replace("/", "__")
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32, I32, U32,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(transparent)]
pub struct TensorShape {
    pub dims: Vec<Dimension>,
}

impl TensorShape {
    pub fn size_c_expr(&self) -> String {
        if self.dims.is_empty() { "1".to_string() }
        else { self.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" * ") }
    }

    pub fn rank(&self) -> usize { self.dims.len() }

    pub fn default_strides_c_expr(&self) -> Vec<String> {
        let mut strides = vec!["1".to_string(); self.dims.len()];
        let mut current = "1".to_string();
        for i in (0..self.dims.len()).rev() {
            strides[i] = current.clone();
            if i > 0 { current = format!("{} * ({})", current, self.dims[i]); }
        }
        strides
    }
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
    Clamp
}

impl Op {
    pub fn generate_c_body(&self, prog_id: &str, node_id: &str, target_idx: &str, inputs: &[String]) -> String {
        let target = format!("buffer_{}_{}[{}]", prog_id, sanitize_id(node_id), target_idx);

        match self {
            Op::Sin => format!("{} = sinf({});", target, inputs[0]),
            Op::Abs => format!("{} = fabsf({});", target, inputs[0]),
            Op::Sqrt => format!("{} = sqrtf({});", target, inputs[0]),
            Op::Square => format!("{} = ({}) * ({});", target, inputs[0], inputs[0]),
            Op::Exp => format!("{} = expf({});", target, inputs[0]),
            Op::Log => format!("{} = logf({});", target, inputs[0]),
            
            Op::Add => format!("{} = {} + {};", target, inputs[0], inputs[1]),
            Op::Sub => format!("{} = {} - {};", target, inputs[0], inputs[1]),
            Op::Mul => format!("{} = {} * {};", target, inputs[0], inputs[1]),
            Op::Div => format!("{} = {} / ({} + 1e-9f);", target, inputs[0], inputs[1]),
            Op::Min => format!("{} = fminf({}, {});", target, inputs[0], inputs[1]),
            Op::Max => format!("{} = fmaxf({}, {});", target, inputs[0], inputs[1]),
            Op::Pow => format!("{} = powf({}, {});", target, inputs[0], inputs[1]),
            
            Op::Clamp => format!("{} = fminf(fmaxf({}, {}), {});", target, inputs[0], inputs[1], inputs[2]),
            _ => "".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Node {
    pub id: String,
    pub op: Op,
    pub inputs: Vec<String>,
    pub shape: TensorShape,
    pub dtype: DataType,
    pub strides: Option<Vec<String>>,
}

impl Node {
    pub fn get_effective_strides_c_expr(&self) -> Vec<String> {
        self.strides.clone().unwrap_or_else(|| self.shape.default_strides_c_expr())
    }
}

pub struct KernelRegistry;
impl KernelRegistry {
    pub fn get_interface(op: &Op) -> crate::graph_ext::NodeInterface {
        use crate::graph_ext::Port;
        let f32_scalar = serde_json::json!([1]);
        let f32_id = "F32".to_string();

        match op {
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Min | Op::Max | Op::Pow | Op::MatMul | Op::Conv => crate::graph_ext::NodeInterface {
                inputs: vec![
                    Port { name: "left".into(), dtype_id: f32_id.clone(), shape: f32_scalar.clone() },
                    Port { name: "right".into(), dtype_id: f32_id.clone(), shape: f32_scalar.clone() }
                ],
                outputs: vec![Port { name: "output".into(), dtype_id: f32_id, shape: f32_scalar }]
            },
            Op::Sin | Op::Abs | Op::Sqrt | Op::Square | Op::Exp | Op::Log | Op::ReduceSum { .. } | Op::Reshape { .. } | Op::Transpose { .. } | Op::Broadcast => crate::graph_ext::NodeInterface {
                inputs: vec![Port { name: "input".into(), dtype_id: f32_id.clone(), shape: f32_scalar.clone() }],
                outputs: vec![Port { name: "output".into(), dtype_id: f32_id, shape: f32_scalar }]
            },
            Op::Clamp => crate::graph_ext::NodeInterface {
                inputs: vec![
                    Port { name: "input".into(), dtype_id: f32_id.clone(), shape: f32_scalar.clone() },
                    Port { name: "min".into(), dtype_id: f32_id.clone(), shape: f32_scalar.clone() },
                    Port { name: "max".into(), dtype_id: f32_id.clone(), shape: f32_scalar.clone() }
                ],
                outputs: vec![Port { name: "output".into(), dtype_id: f32_id, shape: f32_scalar }]
            },
            Op::Constant { .. } => crate::graph_ext::NodeInterface {
                inputs: vec![],
                outputs: vec![Port { name: "output".into(), dtype_id: f32_id, shape: f32_scalar }]
            },
            Op::Input { .. } => crate::graph_ext::NodeInterface {
                inputs: vec![],
                outputs: vec![Port { name: "output".into(), dtype_id: f32_id, shape: f32_scalar }]
            },
            Op::Output { .. } => crate::graph_ext::NodeInterface {
                inputs: vec![Port { name: "input".into(), dtype_id: f32_id, shape: f32_scalar.clone() }],
                outputs: vec![]
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ComputationalGraph {
    pub imports: Option<HashMap<String, String>>,
    pub nodes: Vec<Node>,
}

impl ComputationalGraph {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}