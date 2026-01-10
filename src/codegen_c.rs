use crate::model::{Op, DataType};
use petgraph::graph::NodeIndex;
use crate::compiler::Compiler;
use tera::{Tera, Context};
use serde::Serialize;
use std::fmt::Write;

pub struct CodegenC<'a> {
    compiler: &'a Compiler,
    tera: Tera,
}

#[derive(Serialize)]
struct NodeRenderInfo {
    id: String,
    c_type: String,
    size: usize,
}

#[derive(Serialize)]
struct OpRenderInfo {
    id: String,
    shape: String,
    is_input: bool,
    size: usize,
    loops_open: String,
    loops_close: String,
    indent: String,
    body: String,
}

impl<'a> CodegenC<'a> {
    pub fn new(compiler: &'a Compiler) -> Self {
        let mut tera = Tera::default();
        tera.add_template_file("templates/module.c", Some("module")).expect("Failed to load template");
        Self { compiler, tera }
    }

    pub fn generate(&self, execution_order: &[NodeIndex]) -> anyhow::Result<String> {
        let mut nodes = Vec::new();
        let mut operations = Vec::new();

        for &idx in execution_order {
            let node = &self.compiler.graph[idx];
            nodes.push(NodeRenderInfo {
                id: node.id.clone(),
                c_type: self.map_dtype(&node.dtype).to_string(),
                size: node.shape.size(),
            });
            operations.push(self.prepare_op_render(node));
        }

        let mut context = Context::new();
        context.insert("nodes", &nodes);
        context.insert("operations", &operations);

        Ok(self.tera.render("module", &context)?)
    }

    fn map_dtype(&self, dtype: &DataType) -> &'static str {
        match dtype {
            DataType::F32 => "float",
            DataType::I32 => "int32_t",
            DataType::U32 => "uint32_t",
        }
    }

    fn get_node_by_id(&self, id: &str) -> &crate::model::Node {
        for idx in self.compiler.graph.node_indices() {
            let node = &self.compiler.graph[idx];
            if node.id == id { return node; }
        }
        panic!("Node {} not found", id);
    }

    fn prepare_op_render(&self, node: &crate::model::Node) -> OpRenderInfo {
        let rank = node.shape.rank();
        let mut is_input = false;
        let mut loops_open = String::new();
        let mut loops_close = String::new();
        let indent = "    ".repeat(rank);

        match &node.op {
            Op::Input { .. } => {
                is_input = true;
            }
            _ => {
                for d in 0..rank {
                    let loop_indent = "    ".repeat(d);
                    if d == 0 { writeln!(loops_open, "{}PARALLEL", loop_indent).unwrap(); }
                    writeln!(loops_open, "{}for(int i{} = 0; i{} < {}; ++i{}) {{", 
                        loop_indent, d, d, node.shape.dims[d], d).unwrap();
                    
                    let mut close = String::new();
                    writeln!(close, "{}}}", loop_indent).unwrap();
                    loops_close.insert_str(0, &close);
                }
            }
        }

        OpRenderInfo {
            id: node.id.clone(),
            shape: format!("{:?}", node.shape.dims),
            is_input,
            size: node.shape.size(),
            loops_open,
            loops_close,
            indent,
            body: self.generate_body_expr(node),
        }
    }

    fn generate_body_expr(&self, node: &crate::model::Node) -> String {
        let rank = node.shape.rank();
        let target_strides = node.get_effective_strides();
        
        let mut target_idx = (0..rank)
            .map(|d| format!("i{} * {}", d, target_strides[d]))
            .collect::<Vec<_>>().join(" + ");
        if target_idx.is_empty() { target_idx = "0".into(); }

        match &node.op {
            Op::Add { left, right } | Op::Mul { left, right } => {
                let op = if let Op::Add {..} = node.op { "+" } else { "*" };
                let l_idx = self.generate_index_expr(self.get_node_by_id(left), rank, &node.shape.dims);
                let r_idx = self.generate_index_expr(self.get_node_by_id(right), rank, &node.shape.dims);
                format!("buffer_{}[{}] = buffer_{}[{}] {} buffer_{}[{}];", 
                    node.id, target_idx, left, l_idx, op, right, r_idx)
            }
            Op::Sin { input } => {
                let in_idx = self.generate_index_expr(self.get_node_by_id(input), rank, &node.shape.dims);
                format!("buffer_{}[{}] = sinf(buffer_{}[{}]);", node.id, target_idx, input, in_idx)
            }
            _ => "".into()
        }
    }

    fn generate_index_expr(&self, node: &crate::model::Node, target_rank: usize, target_dims: &[usize]) -> String {
        let rank = node.shape.rank();
        let strides = node.get_effective_strides();
        let mut parts = Vec::new();

        for d in 0..target_rank {
            let in_dim_idx = (d as i32) - (target_rank as i32 - rank as i32);
            if in_dim_idx >= 0 {
                let in_d = in_dim_idx as usize;
                let stride = if node.shape.dims[in_d] == target_dims[d] {
                    strides[in_d]
                } else {
                    0
                };
                if stride != 0 {
                    parts.push(format!("i{} * {}", d, stride));
                }
            }
        }

        if parts.is_empty() { "0".into() } else { parts.join(" + ") }
    }
}
