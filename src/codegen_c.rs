use crate::model::{Op, DataType, TensorShape};
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
    body: String,
}

#[derive(Serialize)]
struct GroupRenderInfo {
    shape: String,
    loops_open: String,
    loops_close: String,
    indent: String,
    operations: Vec<OpRenderInfo>,
}

#[derive(Clone)]
struct NodeInfo {
    id: String,
    dims: Vec<usize>,
    strides: Vec<usize>,
    op: Op,
}

impl<'a> CodegenC<'a> {
    pub fn new(compiler: &'a Compiler) -> Self {
        let mut tera = Tera::default();
        tera.add_template_file("templates/module.c", Some("module")).expect("Failed to load template");
        Self { compiler, tera }
    }

    pub fn generate(&self, execution_order: &[NodeIndex]) -> anyhow::Result<String> {
        let mut nodes_info = Vec::new();
        let mut groups: Vec<GroupRenderInfo> = Vec::new();

        // 1. Собираем информацию о всех узлах для объявлений буферов
        for &idx in execution_order {
            let node = &self.compiler.graph[idx];
            nodes_info.push(NodeRenderInfo {
                id: node.id.clone(),
                c_type: self.map_dtype(&node.dtype).to_string(),
                size: node.shape.size(),
            });
        }

        // 2. Группируем операции для Fusion
        let mut current_group: Option<GroupRenderInfo> = None;

        for &idx in execution_order {
            let node = &self.compiler.graph[idx];
            let info = self.get_node_info(&node.id);
            let shape_str = format!("{:?}", node.shape.dims);

            // Решаем, можно ли добавить узел в текущую группу
            let can_fuse = match &current_group {
                Some(g) => g.shape == shape_str,
                None => false,
            };

            if !can_fuse {
                if let Some(g) = current_group.take() {
                    groups.push(g);
                }
                current_group = Some(self.create_group(&info));
            }

            if let Some(ref mut g) = current_group {
                let body = self.generate_body_expr(&info);
                g.operations.push(OpRenderInfo {
                    id: node.id.clone(),
                    body,
                });
            }
        }

        if let Some(g) = current_group {
            groups.push(g);
        }

        let mut context = Context::new();
        context.insert("nodes", &nodes_info);
        context.insert("groups", &groups);

        Ok(self.tera.render("module", &context)?)
    }

    fn create_group(&self, target: &NodeInfo) -> GroupRenderInfo {
        let rank = target.dims.len();
        let mut loops_open = String::new();
        let mut loops_close = String::new();
        let indent = "    ".repeat(rank);

        if rank == 0 {
            // Скаляры не требуют циклов, но мы можем обернуть их для единообразия 
            // или просто оставить пустыми
        } else {
            for d in 0..rank {
                let loop_indent = "    ".repeat(d);
                if d == 0 { writeln!(loops_open, "{}PARALLEL", loop_indent).unwrap(); }
                writeln!(loops_open, "{}for(int i{} = 0; i{} < {}; ++i{}) {{", 
                    loop_indent, d, d, target.dims[d], d).unwrap();
                
                let mut close = String::new();
                writeln!(close, "{}}}", loop_indent).unwrap();
                loops_close.insert_str(0, &close);
            }
        }

        GroupRenderInfo {
            shape: format!("{:?}", target.dims),
            loops_open,
            loops_close,
            indent,
            operations: Vec::new(),
        }
    }

    fn generate_body_expr(&self, node: &NodeInfo) -> String {
        let rank = node.dims.len();
        let mut target_idx = (0..rank)
            .map(|d| format!("i{} * {}", d, node.strides[d]))
            .collect::<Vec<_>>().join(" + ");
        if target_idx.is_empty() { target_idx = "0".into(); }

        match &node.op {
            Op::Input { .. } => {
                format!("buffer_{}[{}] = 1.0f;", node.id, target_idx)
            }
            Op::Add { left, right } | Op::Mul { left, right } => {
                let op = if let Op::Add {..} = node.op { "+" } else { "*" };
                let left_node = self.get_node_by_id(left);
                let right_node = self.get_node_by_id(right);
                
                let l_idx = self.generate_index_expr(left_node, rank, &node.dims);
                let r_idx = self.generate_index_expr(right_node, rank, &node.dims);
                
                format!("buffer_{}[{}] = buffer_{}[{}] {} buffer_{}[{}];", 
                    node.id, target_idx, left, l_idx, op, right, r_idx)
            }
            Op::Sin { input } => {
                let in_node = self.get_node_by_id(input);
                let in_idx = self.generate_index_expr(in_node, rank, &node.dims);
                format!("buffer_{}[{}] = sinf(buffer_{}[{}]);", node.id, target_idx, input, in_idx)
            }
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

    fn get_node_info(&self, id: &str) -> NodeInfo {
        let node = self.get_node_by_id(id);
        NodeInfo {
            id: node.id.clone(),
            dims: node.shape.dims.clone(),
            strides: node.get_effective_strides(),
            op: node.op.clone(),
        }
    }

    fn get_node_by_id(&self, id: &str) -> &crate::model::Node {
        for idx in self.compiler.graph.node_indices() {
            let node = &self.compiler.graph[idx];
            if node.id == id { return node; }
        }
        panic!("Node {} not found", id);
    }

    fn map_dtype(&self, dtype: &DataType) -> &'static str {
        match dtype {
            DataType::F32 => "float",
            DataType::I32 => "int32_t",
            DataType::U32 => "uint32_t",
        }
    }
}