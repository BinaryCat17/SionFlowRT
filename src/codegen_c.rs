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
    init_values: Option<Vec<f32>>,
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
            let init_values = if let Op::Constant { values } = &node.op {
                Some(values.clone())
            } else {
                None
            };

            nodes_info.push(NodeRenderInfo {
                id: node.id.clone(),
                c_type: self.map_dtype(&node.dtype).to_string(),
                size: node.shape.size(),
                init_values,
            });
        }

        // 2. Группируем операции для Fusion
        let mut current_group: Option<GroupRenderInfo> = None;

        for &idx in execution_order {
            let node = &self.compiler.graph[idx];
            let info = self.get_node_info(&node.id);
            
            // Константы и входы не нуждаются в циклах вычислений, 
            // так как константы инициализируются при объявлении,
            // а входы заполняются извне.
            if let Op::Constant { .. } | Op::Input { .. } = &node.op {
                continue;
            }

            let shape_str = format!("{:?}", node.shape.dims);

            // Решаем, можно ли добавить узел в текущую группу
            // Для ReduceSum, MatMul и Conv мы пока создаем отдельные группы
            let is_special = matches!(node.op, Op::ReduceSum { .. } | Op::MatMul { .. } | Op::Conv { .. });
            let can_fuse = match &current_group {
                Some(g) => g.shape == shape_str && !is_special,
                None => false,
            };

            if !can_fuse {
                if let Some(g) = current_group.take() {
                    groups.push(g);
                }
                
                if matches!(node.op, Op::ReduceSum { .. }) {
                    groups.push(self.create_reduction_group(&info));
                    continue;
                } else if matches!(node.op, Op::MatMul { .. }) {
                    groups.push(self.create_matmul_group(&info));
                    continue;
                } else if matches!(node.op, Op::Conv { .. }) {
                    groups.push(self.create_conv_group(&info));
                    continue;
                } else {
                    current_group = Some(self.create_group(&info));
                }
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
            // Скаляры
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

    fn create_reduction_group(&self, node: &NodeInfo) -> GroupRenderInfo {
        if let Op::ReduceSum { input, axis } = &node.op {
            let in_node = self.get_node_by_id(input);
            let in_dims = &in_node.shape.dims;
            let rank = in_dims.len();
            
            let mut loops_open = String::new();
            let mut loops_close = String::new();

            // Внешние циклы по всем осям, кроме axis
            let mut current_indent_level = 0;
            for d in 0..rank {
                if d == *axis { continue; }
                let loop_indent = "    ".repeat(current_indent_level);
                if current_indent_level == 0 { writeln!(loops_open, "{}PARALLEL", loop_indent).unwrap(); }
                writeln!(loops_open, "{}for(int i{} = 0; i{} < {}; ++i{}) {{", 
                    loop_indent, d, d, in_dims[d], d).unwrap();
                
                let mut close = String::new();
                writeln!(close, "{}}}", loop_indent).unwrap();
                loops_close.insert_str(0, &close);
                current_indent_level += 1;
            }
            
            let indent = "    ".repeat(current_indent_level);
            
            // Инициализация аккумулятора
            let target_idx = self.generate_target_index_expr(node, rank, axis);
            writeln!(loops_open, "{}buffer_{}[{}] = 0.0f;", indent, node.id, target_idx).unwrap();
            
            // Внутренний цикл по оси редукции
            writeln!(loops_open, "{}for(int i{} = 0; i{} < {}; ++i{}) {{", 
                indent, axis, axis, in_dims[*axis], axis).unwrap();
            
            let mut inner_close = String::new();
            writeln!(inner_close, "{}}}", indent).unwrap();
            loops_close.insert_str(0, &inner_close);
            
            let in_idx = self.generate_index_expr(in_node, rank, in_dims);
            let body = format!("{}buffer_{}[{}] += buffer_{}[{}];", 
                "    ", node.id, target_idx, input, in_idx);

            return GroupRenderInfo {
                shape: format!("Reduction of {} axis {}", input, axis),
                loops_open,
                loops_close,
                indent: indent + "    ",
                operations: vec![OpRenderInfo { id: node.id.clone(), body }],
            };
        }
        panic!("Not a reduction op");
    }

    fn create_matmul_group(&self, node: &NodeInfo) -> GroupRenderInfo {
        if let Op::MatMul { left, right } = &node.op {
            let left_node = self.get_node_by_id(left);
            let right_node = self.get_node_by_id(right);
            
            let m = left_node.shape.dims[0];
            let k = left_node.shape.dims[1];
            let n = right_node.shape.dims[1];
            
            let l_strides = left_node.get_effective_strides();
            let r_strides = right_node.get_effective_strides();

            let mut loops_open = String::new();
            let mut loops_close = String::new();
            
            writeln!(loops_open, "PARALLEL").unwrap();
            writeln!(loops_open, "for(int i = 0; i < {}; ++i) {{", m).unwrap();
            writeln!(loops_open, "    for(int j = 0; j < {}; ++j) {{", n).unwrap();
            writeln!(loops_open, "        float acc = 0.0f;").unwrap();
            writeln!(loops_open, "        for(int k = 0; k < {}; ++k) {{", k).unwrap();
            
            let l_idx = format!("i * {} + k * {}", l_strides[0], l_strides[1]);
            let r_idx = format!("k * {} + j * {}", r_strides[0], r_strides[1]);
            
            let body = format!("acc += buffer_{}[{}] * buffer_{}[{}];", left, l_idx, right, r_idx);
            
            let target_idx = format!("i * {} + j * {}", node.strides[0], node.strides[1]);

            writeln!(loops_close, "        }}").unwrap();
            writeln!(loops_close, "        buffer_{}[{}] = acc;", node.id, target_idx).unwrap();
            writeln!(loops_close, "    }}").unwrap();
            writeln!(loops_close, "}}").unwrap();

            return GroupRenderInfo {
                shape: format!("MatMul {} x {}", left, right),
                loops_open,
                loops_close,
                indent: "            ".into(),
                operations: vec![OpRenderInfo { id: node.id.clone(), body }],
            };
        }
        panic!("Not a matmul op");
    }

    fn create_conv_group(&self, node: &NodeInfo) -> GroupRenderInfo {
        if let Op::Conv { input, kernel } = &node.op {
            let in_node = self.get_node_by_id(input);
            let ker_node = self.get_node_by_id(kernel);
            
            let in_rank = in_node.shape.rank();
            let ker_rank = ker_node.shape.rank();
            let out_rank = node.dims.len();

            let mut loops_open = String::new();
            let mut loops_close = String::new();

            // Внешние циклы по выходному тензору
            for d in 0..out_rank {
                let loop_indent = "    ".repeat(d);
                if d == 0 { writeln!(loops_open, "{}PARALLEL", loop_indent).unwrap(); }
                writeln!(loops_open, "{}for(int i{} = 0; i{} < {}; ++i{}) {{", 
                    loop_indent, d, d, node.dims[d], d).unwrap();
                
                let mut close = String::new();
                writeln!(close, "{}}}", loop_indent).unwrap();
                loops_close.insert_str(0, &close);
            }

            let outer_indent = "    ".repeat(out_rank);
            let target_idx = (0..out_rank)
                .map(|d| format!("i{} * {}", d, node.strides[d]))
                .collect::<Vec<_>>().join(" + ");
            
            writeln!(loops_open, "{}float acc = 0.0f;", outer_indent).unwrap();

            // Внутренние циклы по ядру
            for kd in 0..ker_rank {
                let loop_indent = "    ".repeat(out_rank + kd);
                writeln!(loops_open, "{}for(int k{} = 0; k{} < {}; ++k{}) {{", 
                    loop_indent, kd, kd, ker_node.shape.dims[kd], kd).unwrap();
            }

            let inner_indent = "    ".repeat(out_rank + ker_rank);
            
            // Вычисление индекса входа: i[d] + k[d]
            let in_strides = in_node.get_effective_strides();
            let mut in_parts = Vec::new();
            for d in 0..in_rank {
                if d < ker_rank {
                    in_parts.push(format!("(i{} + k{}) * {}", d, d, in_strides[d]));
                } else {
                    in_parts.push(format!("i{} * {}", d, in_strides[d]));
                }
            }
            let in_idx = if in_parts.is_empty() { "0".into() } else { in_parts.join(" + ") };

            // Вычисление индекса ядра
            let ker_strides = ker_node.get_effective_strides();
            let ker_idx = (0..ker_rank)
                .map(|d| format!("k{} * {}", d, ker_strides[d]))
                .collect::<Vec<_>>().join(" + ");
            let ker_idx = if ker_idx.is_empty() { "0".into() } else { ker_idx };

            let body = format!("acc += buffer_{}[{}] * buffer_{}[{}];", input, in_idx, kernel, ker_idx);
            writeln!(loops_open, "{}{}", inner_indent, body).unwrap();
            
            // Сначала закрываем циклы ЯДРА в loops_open
            for kd in (0..ker_rank).rev() {
                let loop_indent = "    ".repeat(out_rank + kd);
                writeln!(loops_open, "{}}}", loop_indent).unwrap();
            }

            // Записываем результат (внутри внешних циклов)
            writeln!(loops_open, "{}buffer_{}[{}] = acc;", outer_indent, node.id, target_idx).unwrap();

            return GroupRenderInfo {
                shape: format!("Conv {} by {}", input, kernel),
                loops_open,
                loops_close, // Здесь только закрытие внешних циклов
                indent: "".into(),
                operations: vec![],
            };
        }
        panic!("Not a conv op");
    }

    fn generate_target_index_expr(&self, node: &NodeInfo, rank: usize, skip_axis: &usize) -> String {
        let mut parts = Vec::new();
        let mut out_d = 0;
        for d in 0..rank {
            if d == *skip_axis { continue; }
            parts.push(format!("i{} * {}", d, node.strides[out_d]));
            out_d += 1;
        }
        if parts.is_empty() { "0".into() } else { parts.join(" + ") }
    }

    fn generate_body_expr(&self, node: &NodeInfo) -> String {
        let rank = node.dims.len();
        let mut target_idx = (0..rank)
            .map(|d| format!("i{} * {}", d, node.strides[d]))
            .collect::<Vec<_>>().join(" + ");
        if target_idx.is_empty() { target_idx = "0".into(); }

        match &node.op {
            Op::Input { .. } | Op::Constant { .. } | Op::ReduceSum { .. } | Op::MatMul { .. } | Op::Conv { .. } => {
                "".into() // Не должно вызываться в текущей логике или обрабатывается отдельно
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
            Op::Transpose { input, permutation } => {
                let in_node = self.get_node_by_id(input);
                let in_strides = in_node.get_effective_strides();
                let mut in_parts = Vec::new();
                for (n, &p_n) in permutation.iter().enumerate() {
                    in_parts.push(format!("i{} * {}", n, in_strides[p_n]));
                }
                let in_idx = if in_parts.is_empty() { "0".into() } else { in_parts.join(" + ") };
                format!("buffer_{}[{}] = buffer_{}[{}];", node.id, target_idx, input, in_idx)
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