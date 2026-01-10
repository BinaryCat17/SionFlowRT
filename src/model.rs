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

pub fn sanitize_id(id: &str) -> String {
    id.replace("/", "__")
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum DataType {
    F32, I32, U32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
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

macro_rules! define_ops {
    (
        unary: { $($un_name:ident => $un_expr:expr),* },
        binary: { $($bin_name:ident => $bin_expr:expr),* },
        special: { $($spec_name:ident { $($f_name:ident : $f_type:ty),* }),* }
    ) => {
        #[derive(Debug, Serialize, Deserialize, Clone)]
        pub enum Op {
            $($un_name { input: String },)*
            $($bin_name { left: String, right: String },)*
            $($spec_name { $($f_name : $f_type),* },)*
        }

        impl Op {
            pub fn get_dependencies(&self) -> Vec<String> {
                match self {
                    $(Op::$un_name { input } => vec![input.clone()],)*
                    $(Op::$bin_name { left, right } => vec![left.clone(), right.clone()],)*
                    Op::Input { .. } | Op::Constant { .. } => vec![],
                    Op::Transpose { input, .. } | Op::ReduceSum { input, .. } | Op::Output { input, .. } | Op::Broadcast { input } | Op::Reshape { input, .. } => vec![input.clone()],
                    Op::MatMul { left, right } | Op::Conv { input: left, kernel: right } => vec![left.clone(), right.clone()],
                    Op::Clamp { input, min, max } => vec![input.clone(), min.clone(), max.clone()],
                    Op::Call { inputs, .. } => inputs.values().cloned().collect(),
                }
            }

            pub fn map_dependencies<F>(&self, mut f: F) -> Self 
            where F: FnMut(&str) -> String {
                match self {
                    $(Op::$un_name { input } => Op::$un_name { input: f(input) },)*
                    $(Op::$bin_name { left, right } => Op::$bin_name { left: f(left), right: f(right) },)*
                    Op::Input { name } => Op::Input { name: name.clone() },
                    Op::Constant { values } => Op::Constant { values: values.clone() },
                    Op::Transpose { input, permutation } => Op::Transpose { input: f(input), permutation: permutation.clone() },
                    Op::ReduceSum { input, axis } => Op::ReduceSum { input: f(input), axis: *axis },
                    Op::MatMul { left, right } => Op::MatMul { left: f(left), right: f(right) },
                    Op::Conv { input, kernel } => Op::Conv { input: f(input), kernel: f(kernel) },
                    Op::Broadcast { input } => Op::Broadcast { input: f(input) },
                    Op::Reshape { input, new_shape } => Op::Reshape { input: f(input), new_shape: new_shape.clone() },
                    Op::Clamp { input, min, max } => Op::Clamp { input: f(input), min: f(min), max: f(max) },
                    Op::Output { name, input } => Op::Output { name: name.clone(), input: f(input) },
                    Op::Call { subgraph, inputs } => {
                        let mut new_inputs = HashMap::new();
                        for (k, v) in inputs {
                            new_inputs.insert(k.clone(), f(v));
                        }
                        Op::Call { subgraph: subgraph.clone(), inputs: new_inputs }
                    },
                }
            }

            pub fn infer_shape(&self, get_node_shape: impl Fn(&str) -> Option<Vec<Dimension>>) -> Option<Vec<Dimension>> {
                match self {
                    $(Op::$bin_name { left, right } => {
                        let l = get_node_shape(left);
                        let r = get_node_shape(right);
                        match (l, r) {
                            (Some(ld), Some(rd)) => if ld.len() >= rd.len() { Some(ld) } else { Some(rd) },
                            (Some(ld), None) => Some(ld),
                            (None, Some(rd)) => Some(rd),
                            _ => None,
                        }
                    })*
                    $(Op::$un_name { input } |)* Op::Output { input, .. } | Op::Broadcast { input } | Op::Clamp { input, .. } => get_node_shape(input),
                    Op::Reshape { new_shape, .. } => Some(new_shape.clone()),
                    Op::Transpose { input, permutation } => {
                        get_node_shape(input).map(|dims| permutation.iter().map(|&i| dims[i].clone()).collect())
                    }
                    Op::ReduceSum { input, axis } => {
                        get_node_shape(input).map(|mut dims| {
                            if *axis < dims.len() { dims.remove(*axis); }
                            dims
                        })
                    }
                    Op::MatMul { left, right } => {
                        let l = get_node_shape(left)?;
                        let r = get_node_shape(right)?;
                        if l.len() == 2 && r.len() == 2 {
                            Some(vec![l[0].clone(), r[1].clone()])
                        } else { None }
                    }
                    Op::Conv { input, kernel } => {
                        let in_s = get_node_shape(input)?;
                        let ker_s = get_node_shape(kernel)?;
                        let mut out_s = Vec::new();
                        for i in 0..in_s.len() {
                            if i < ker_s.len() {
                                out_s.push(Dimension::Add(
                                    Box::new(Dimension::Sub(Box::new(in_s[i].clone()), Box::new(ker_s[i].clone()))),
                                    Box::new(Dimension::Value(1))
                                ));
                            } else {
                                out_s.push(in_s[i].clone());
                            }
                        }
                        Some(out_s)
                    }
                    _ => None,
                }
            }

            pub fn generate_c_body(&self, prog_id: &str, node_id: &str, target_idx: &str, get_index_expr: impl Fn(&str) -> String) -> String {
                let buf = |id: &str, idx: &str| format!("buffer_{}_{}[{}]", prog_id, sanitize_id(id), idx);
                let target = buf(node_id, target_idx);

                match self {
                    $(Op::$un_name { input } => {
                        let val = buf(input, &get_index_expr(input));
                        let expr = $un_expr.replace("{}", &val);
                        format!("{} = {};", target, expr)
                    })*
                    $(Op::$bin_name { left, right } => {
                        let l_val = buf(left, &get_index_expr(left));
                        let r_val = buf(right, &get_index_expr(right));
                        let expr = $bin_expr.replacen("{}", &l_val, 1).replacen("{}", &r_val, 1);
                        format!("{} = {};", target, expr)
                    })*
                    Op::Clamp { input, min, max } => format!("{} = fminf(fmaxf({}, {}), {});", target, buf(input, &get_index_expr(input)), buf(min, &get_index_expr(min)), buf(max, &get_index_expr(max))),
                    Op::Transpose { input, .. } | Op::Output { input, .. } | Op::Broadcast { input } | Op::Reshape { input, .. } => format!("{} = {};", target, buf(input, &get_index_expr(input))),
                    _ => "".to_string(),
                }
            }
        }
    };
}

define_ops! {
    unary: {
        Sin => "sinf({})",
        Abs => "fabsf({})",
        Sqrt => "sqrtf({})",
        Square => "({}) * ({})",
        Exp => "expf({})",
        Log => "logf({})"
    },
    binary: {
        Add => "{} + {}",
        Sub => "{} - {}",
        Mul => "{} * {}",
        Div => "{} / ({} + 1e-9f)",
        Min => "fminf({}, {})",
        Max => "fmaxf({}, {})",
        Pow => "powf({}, {})"
    },
    special: {
        Input { name: String },
        Constant { values: Vec<f32> },
        Transpose { input: String, permutation: Vec<usize> },
        ReduceSum { input: String, axis: usize },
        MatMul { left: String, right: String },
        Conv { input: String, kernel: String },
        Call { subgraph: String, inputs: HashMap<String, String> },
        Output { name: String, input: String },
        Broadcast { input: String },
        Reshape { input: String, new_shape: Vec<Dimension> },
        Clamp { input: String, min: String, max: String }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Node {
    pub id: String,
    pub op: Op,
    pub shape: TensorShape,
    pub dtype: DataType,
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
