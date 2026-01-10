use crate::model::Op;
use crate::manifest::{Manifest, MappingSource};
use crate::CompiledProgram;
use tera::{Tera, Context};
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Write;

pub struct CodegenC<'a> {
    programs: HashMap<String, CompiledProgram>,
    manifest: &'a Manifest,
    tera: Tera,
}

#[derive(Serialize)]
pub struct GeneratedCode {
    pub module: String,
    pub runtime: String,
}

#[derive(Serialize, Clone)]
struct NodeRenderInfo {
    prog_id: String,
    node_id: String,
    c_type: String,
    size_expr: String,
    init_values: Option<Vec<f32>>,
    is_stateful: bool,
}

#[derive(Serialize, Clone)]
struct OpRenderInfo {
    id: String,
    body: String,
}

#[derive(Serialize, Clone)]
struct GroupRenderInfo {
    prog_id: String,
    shape: String,
    loops_open: String,
    loops_close: String,
    indent: String,
    operations: Vec<OpRenderInfo>,
}

#[derive(Clone)]
struct NodeInfo {
    node_id: String,
    dims: Vec<String>,
    strides: Vec<String>,
    op: Op,
}

impl<'a> CodegenC<'a> {
    pub fn new(programs: HashMap<String, CompiledProgram>, manifest: &'a Manifest) -> Self {
        let mut tera = Tera::default();
        tera.add_template_file("templates/module.c", Some("module")).expect("Failed to load template");
        tera.add_template_file("templates/sdl2_runtime.c", Some("sdl2_runtime")).expect("Failed to load template");
        tera.add_template_file("templates/headless_runtime.c", Some("headless_runtime")).expect("Failed to load template");
        Self { programs, manifest, tera }
    }

    pub fn generate(&self, runtime_type: &str) -> anyhow::Result<GeneratedCode> {
        let mut nodes_info = Vec::new();
        let mut all_groups: Vec<GroupRenderInfo> = Vec::new();
        let mut nodes_map: HashMap<String, HashMap<String, NodeRenderInfo>> = HashMap::new();

        for (prog_id, prog) in &self.programs {
            let mut prog_nodes = HashMap::new();
            for idx in prog.compiler.graph.node_indices() {
                let node = &prog.compiler.graph[idx];
                let init_values = if let Op::Constant { values } = &node.op {
                    Some(values.clone())
                } else {
                    None
                };

                let is_stateful = self.manifest.mappings.iter().any(|m| {
                    matches!(&m.source, MappingSource::Link { program, output } 
                        if program == prog_id && output == &node.id && m.program == *prog_id)
                });

                let info = NodeRenderInfo {
                    prog_id: prog_id.clone(),
                    node_id: self.sanitize_id(&node.id),
                    c_type: self.map_dtype(&node.dtype).to_string(),
                    size_expr: node.shape.size_c_expr(),
                    init_values,
                    is_stateful,
                };
                
                nodes_info.push(info.clone());
                prog_nodes.insert(node.id.clone(), info);
            }
            nodes_map.insert(prog_id.clone(), prog_nodes);

            let mut current_group: Option<GroupRenderInfo> = None;

            for &idx in &prog.execution_order {
                let node = &prog.compiler.graph[idx];
                let info = self.get_node_info(prog_id, &node.id);
                
                if let Op::Constant { .. } | Op::Input { .. } = &node.op {
                    continue;
                }

                let shape_str = format!("{:?}", info.dims);
                let is_special = matches!(node.op, Op::ReduceSum { .. } | Op::MatMul { .. } | Op::Conv { .. });
                let can_fuse = match &current_group {
                    Some(g) => g.shape == shape_str && !is_special,
                    None => false,
                };

                if !can_fuse {
                    if let Some(g) = current_group.take() {
                        all_groups.push(g);
                    }
                    
                    if matches!(node.op, Op::ReduceSum { .. }) {
                        all_groups.push(self.create_reduction_group(prog_id, &info));
                    } else if matches!(node.op, Op::MatMul { .. }) {
                        all_groups.push(self.create_matmul_group(prog_id, &info));
                    } else if matches!(node.op, Op::Conv { .. }) {
                        all_groups.push(self.create_conv_group(prog_id, &info));
                    } else {
                        let mut g = self.create_group(prog_id, &info);
                        let body = self.generate_body_expr(prog_id, &info);
                        g.operations.push(OpRenderInfo {
                            id: node.id.clone(),
                            body,
                        });
                        current_group = Some(g);
                    }
                    continue;
                }

                if let Some(ref mut g) = current_group {
                    let body = self.generate_body_expr(prog_id, &info);
                    g.operations.push(OpRenderInfo {
                        id: node.id.clone(),
                        body,
                    });
                }
            }

            if let Some(g) = current_group {
                all_groups.push(g);
            }
        }

        let mut context = Context::new();
        context.insert("nodes", &nodes_info);
        context.insert("nodes_map", &nodes_map);
        context.insert("groups", &all_groups);
        context.insert("mappings", &self.manifest.mappings);
        
        let empty_params = HashMap::new();
        let params = self.manifest.parameters.as_ref().unwrap_or(&empty_params);
        context.insert("parameters", &params);

        let module = self.tera.render("module", &context)?;
        let runtime = self.tera.render(runtime_type, &context)?;

        Ok(GeneratedCode { module, runtime })
    }

    fn create_group(&self, prog_id: &str, target: &NodeInfo) -> GroupRenderInfo {
        let rank = target.dims.len();
        let mut loops_open = String::new();
        let mut loops_close = String::new();
        let indent = "    ".repeat(rank);

        if rank > 0 {
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
            prog_id: prog_id.to_string(),
            shape: format!("{:?}", target.dims),
            loops_open,
            loops_close,
            indent,
            operations: Vec::new(),
        }
    }

    fn sanitize_id(&self, id: &str) -> String {
        id.replace("/", "__")
    }

    fn create_reduction_group(&self, prog_id: &str, node: &NodeInfo) -> GroupRenderInfo {
        if let Op::ReduceSum { input, axis } = &node.op {
            let in_node = self.get_node_by_id(prog_id, input);
            let in_dims: Vec<String> = in_node.shape.dims.iter().map(|d| d.to_string()).collect();
            let rank = in_node.shape.rank();
            
            let mut loops_open = String::new();
            let mut loops_close = String::new();
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
            let target_idx = self.generate_target_index_expr(node, rank, axis);
            writeln!(loops_open, "{}buffer_{}_{}[{}] = 0.0f;", indent, prog_id, self.sanitize_id(&node.node_id), target_idx).unwrap();
            
            writeln!(loops_open, "{}for(int i{} = 0; i{} < {}; ++i{}) {{", 
                indent, axis, axis, in_dims[*axis], axis).unwrap();
            
            let mut inner_close = String::new();
            writeln!(inner_close, "{}}}", indent).unwrap();
            loops_close.insert_str(0, &inner_close);
            
            let in_idx = self.generate_index_expr(prog_id, in_node, rank, &in_dims);
            let body = format!("{}buffer_{}_{}[{}] += buffer_{}_{}[{}];", 
                "    ", prog_id, self.sanitize_id(&node.node_id), target_idx, prog_id, self.sanitize_id(input), in_idx);

            return GroupRenderInfo {
                prog_id: prog_id.to_string(),
                shape: format!("Reduction of {} axis {}", input, axis),
                loops_open,
                loops_close,
                indent: indent + "    ",
                operations: vec![OpRenderInfo { id: node.node_id.clone(), body }],
            };
        }
        panic!("Not a reduction op");
    }

    fn create_matmul_group(&self, prog_id: &str, node: &NodeInfo) -> GroupRenderInfo {
        if let Op::MatMul { left, right } = &node.op {
            let left_node = self.get_node_by_id(prog_id, left);
            let right_node = self.get_node_by_id(prog_id, right);
            let left_dims: Vec<String> = left_node.shape.dims.iter().map(|d| d.to_string()).collect();
            let right_dims: Vec<String> = right_node.shape.dims.iter().map(|d| d.to_string()).collect();

            let m = &left_dims[0];
            let k_dim = &left_dims[1];
            let n = &right_dims[1];
            let l_strides = left_node.get_effective_strides_c_expr();
            let r_strides = right_node.get_effective_strides_c_expr();

            let mut loops_open = String::new();
            let mut loops_close = String::new();
            
            writeln!(loops_open, "PARALLEL").unwrap();
            writeln!(loops_open, "for(int i = 0; i < {}; ++i) {{", m).unwrap();
            writeln!(loops_open, "    for(int j = 0; j < {}; ++j) {{", n).unwrap();
            writeln!(loops_open, "        float acc = 0.0f;").unwrap();
            writeln!(loops_open, "        for(int k = 0; k < {}; ++k) {{", k_dim).unwrap();
            
            let l_idx = format!("i * ({}) + k * ({})", l_strides[0], l_strides[1]);
            let r_idx = format!("k * ({}) + j * ({})", r_strides[0], r_strides[1]);
            let body = format!("acc += buffer_{}_{}[{}] * buffer_{}_{}[{}];", prog_id, self.sanitize_id(left), l_idx, prog_id, self.sanitize_id(right), r_idx);
            let target_idx = format!("i * ({}) + j * ({})", node.strides[0], node.strides[1]);

            writeln!(loops_close, "        }}").unwrap();
            writeln!(loops_close, "        buffer_{}_{}[{}] = acc;", prog_id, self.sanitize_id(&node.node_id), target_idx).unwrap();
            writeln!(loops_close, "    }}").unwrap();
            writeln!(loops_close, "}}").unwrap();

            return GroupRenderInfo {
                prog_id: prog_id.to_string(),
                shape: format!("MatMul {} x {}", left, right),
                loops_open,
                loops_close,
                indent: "            ".into(),
                operations: vec![OpRenderInfo { id: node.node_id.clone(), body }],
            };
        }
        panic!("Not a matmul op");
    }

    fn create_conv_group(&self, prog_id: &str, node: &NodeInfo) -> GroupRenderInfo {
        if let Op::Conv { input, kernel } = &node.op {
            let in_node = self.get_node_by_id(prog_id, input);
            let ker_node = self.get_node_by_id(prog_id, kernel);
            let _in_dims: Vec<String> = in_node.shape.dims.iter().map(|d| d.to_string()).collect();
            let ker_dims: Vec<String> = ker_node.shape.dims.iter().map(|d| d.to_string()).collect();
            let in_rank = in_node.shape.rank();
            let ker_rank = ker_node.shape.rank();
            let out_rank = node.dims.len();

            let mut loops_open = String::new();
            let mut loops_close = String::new();

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
                .map(|d| format!("i{} * ({})", d, node.strides[d]))
                .collect::<Vec<_>>().join(" + ");
            
            writeln!(loops_open, "{}float acc = 0.0f;", outer_indent).unwrap();

            for kd in 0..ker_rank {
                let loop_indent = "    ".repeat(out_rank + kd);
                writeln!(loops_open, "{}for(int k{} = 0; k{} < {}; ++k{}) {{", 
                    loop_indent, kd, kd, ker_dims[kd], kd).unwrap();
            }

            let inner_indent = "    ".repeat(out_rank + ker_rank);
            let in_strides = in_node.get_effective_strides_c_expr();
            let mut in_parts = Vec::new();
            for d in 0..in_rank {
                if d < ker_rank {
                    in_parts.push(format!("(i{} + k{}) * ({})", d, d, in_strides[d]));
                } else {
                    in_parts.push(format!("i{} * ({})", d, in_strides[d]));
                }
            }
            let in_idx = if in_parts.is_empty() { "0".into() } else { in_parts.join(" + ") };

            let ker_strides = ker_node.get_effective_strides_c_expr();
            let ker_idx = (0..ker_rank)
                .map(|d| format!("k{} * ({})", d, ker_strides[d]))
                .collect::<Vec<_>>().join(" + ");
            let ker_idx = if ker_idx.is_empty() { "0".into() } else { ker_idx };

            let body = format!("acc += buffer_{}_{}[{}] * buffer_{}_{}[{}];", prog_id, self.sanitize_id(input), in_idx, prog_id, self.sanitize_id(kernel), ker_idx);
            writeln!(loops_open, "{}{}", inner_indent, body).unwrap();
            
            for kd in (0..ker_rank).rev() {
                let loop_indent = "    ".repeat(out_rank + kd);
                writeln!(loops_open, "{}}}", loop_indent).unwrap();
            }

            writeln!(loops_open, "{}buffer_{}_{}[{}] = acc;", outer_indent, prog_id, self.sanitize_id(&node.node_id), target_idx).unwrap();

            return GroupRenderInfo {
                prog_id: prog_id.to_string(),
                shape: format!("Conv {} by {}", input, kernel),
                loops_open,
                loops_close,
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
            parts.push(format!("i{} * ({})", d, node.strides[out_d]));
            out_d += 1;
        }
        if parts.is_empty() { "0".into() } else { parts.join(" + ") }
    }

    fn generate_body_expr(&self, prog_id: &str, node: &NodeInfo) -> String {
        let rank = node.dims.len();
        let mut target_idx = (0..rank)
            .map(|d| format!("i{} * ({})", d, node.strides[d]))
            .collect::<Vec<_>>().join(" + ");
        if target_idx.is_empty() { target_idx = "0".into(); }

        if let Op::Transpose { input, permutation } = &node.op {
            let in_node = self.get_node_by_id(prog_id, input);
            let in_strides = in_node.get_effective_strides_c_expr();
            let mut in_parts = Vec::new();
            for (n, &p_n) in permutation.iter().enumerate() {
                in_parts.push(format!("i{} * ({})", n, in_strides[p_n]));
            }
            let in_idx = if in_parts.is_empty() { "0".into() } else { in_parts.join(" + ") };
            return format!("buffer_{}_{}[{}] = buffer_{}_{}[{}];", 
                prog_id, self.sanitize_id(&node.node_id), target_idx, prog_id, self.sanitize_id(input), in_idx);
        }

        node.op.generate_c_body(prog_id, &self.sanitize_id(&node.node_id), &target_idx, |dep_id| {
            let dep_node = self.get_node_by_id(prog_id, dep_id);
            self.generate_index_expr(prog_id, dep_node, rank, &node.dims)
        })
    }

    fn generate_index_expr(&self, _prog_id: &str, node: &crate::model::Node, target_rank: usize, _target_dims: &[String]) -> String {
        let rank = node.shape.rank();
        let strides = node.get_effective_strides_c_expr();
        let mut parts = Vec::new();

        for d in 0..rank {
            if d < target_rank {
                // Если размерность исходного тензора равна 1, то для любой целевой 
                // размерности индекс в этом измерении будет 0.
                let is_one = match &node.shape.dims[d] {
                    crate::model::Dimension::Value(1) => true,
                    _ => false,
                };

                if !is_one {
                    parts.push(format!("i{} * ({})", d, strides[d]));
                }
            }
        }

        if parts.is_empty() { "0".into() } else { parts.join(" + ") }
    }

    fn get_node_info(&self, prog_id: &str, node_id: &str) -> NodeInfo {
        let node = self.get_node_by_id(prog_id, node_id);
        NodeInfo {
            node_id: node_id.to_string(),
            dims: node.shape.dims.iter().map(|d| d.to_string()).collect(),
            strides: node.get_effective_strides_c_expr(),
            op: node.op.clone(),
        }
    }

    fn get_node_by_id(&self, prog_id: &str, node_id: &str) -> &crate::model::Node {
        let prog = &self.programs[prog_id];
        for idx in prog.compiler.graph.node_indices() {
            let node = &prog.compiler.graph[idx];
            if node.id == node_id { return node; }
        }
        panic!("Node {} not found in program {}", node_id, prog_id);
    }

    fn map_dtype(&self, dtype: &crate::model::DataType) -> &'static str {
        match dtype {
            crate::model::DataType::F32 => "float",
            crate::model::DataType::I32 => "int32_t",
            crate::model::DataType::U32 => "uint32_t",
        }
    }
}


    