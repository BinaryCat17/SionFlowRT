use crate::model::Op;
use crate::manifest::{Manifest, MappingSource};
use crate::CompiledProgram;
use tera::{Tera, Context};
use serde::Serialize;
use std::collections::HashMap;

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
struct LoopInfo {
    var: String,
    limit: String,
}

#[derive(Serialize, Clone)]
struct KernelInfo {
    init: String,
    inner_loops: Vec<LoopInfo>,
    body: String,
    finalize: String,
}

#[derive(Serialize, Clone)]
struct GroupRenderInfo {
    prog_id: String,
    shape: String,
    outer_loops: Vec<LoopInfo>,
    is_parallel: bool,
    fusion_ops: Vec<OpRenderInfo>,
    kernel: Option<KernelInfo>,
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
                let info = NodeRenderInfo {
                    prog_id: prog_id.clone(),
                    node_id: crate::model::sanitize_id(&node.id),
                    c_type: match node.dtype {
                        crate::model::DataType::F32 => "float",
                        crate::model::DataType::I32 => "int32_t",
                        crate::model::DataType::U32 => "uint32_t",
                    }.to_string(),
                    size_expr: node.shape.size_c_expr(),
                    init_values: if let Op::Constant { values } = &node.op { Some(values.clone()) } else { None },
                    is_stateful: self.manifest.mappings.iter().any(|m| {
                        matches!(&m.source, MappingSource::Link { program, output } 
                            if program == prog_id && output == &node.id && m.program == *prog_id)
                    }),
                };
                nodes_info.push(info.clone());
                prog_nodes.insert(node.id.clone(), info);
            }
            nodes_map.insert(prog_id.clone(), prog_nodes);

            let mut current_group: Option<GroupRenderInfo> = None;

            for &idx in &prog.execution_order {
                let node = &prog.compiler.graph[idx];
                let info = self.get_node_info(prog_id, &node.id);
                if let Op::Constant { .. } | Op::Input { .. } = &node.op { continue; }

                let shape_str = format!("{:?}", info.dims);
                let is_special = matches!(node.op, Op::ReduceSum { .. } | Op::MatMul { .. } | Op::Conv { .. });
                let can_fuse = match &current_group {
                    Some(g) => g.shape == shape_str && !is_special && g.kernel.is_none(),
                    None => false,
                };

                if !can_fuse {
                    if let Some(g) = current_group.take() { all_groups.push(g); }
                    
                    let group = if matches!(node.op, Op::ReduceSum { .. }) {
                        self.create_reduction_group(prog_id, &info)
                    } else if matches!(node.op, Op::MatMul { .. }) {
                        self.create_matmul_group(prog_id, &info)
                    } else if matches!(node.op, Op::Conv { .. }) {
                        self.create_conv_group(prog_id, &info)
                    } else {
                        let mut g = self.create_fusion_group(prog_id, &info);
                        g.fusion_ops.push(OpRenderInfo { id: node.id.clone(), body: self.generate_body_expr(prog_id, &info) });
                        g
                    };
                    
                    if is_special { all_groups.push(group); }
                    else { current_group = Some(group); }
                    continue;
                }

                if let Some(ref mut g) = current_group {
                    g.fusion_ops.push(OpRenderInfo { id: node.id.clone(), body: self.generate_body_expr(prog_id, &info) });
                }
            }
            if let Some(g) = current_group { all_groups.push(g); }
        }

        let mut context = Context::new();
        context.insert("nodes", &nodes_info);
        context.insert("nodes_map", &nodes_map);
        context.insert("groups", &all_groups);
        context.insert("mappings", &self.manifest.mappings);
        let empty_params = HashMap::new();
        context.insert("parameters", self.manifest.parameters.as_ref().unwrap_or(&empty_params));

        Ok(GeneratedCode {
            module: self.tera.render("module", &context)?,
            runtime: self.tera.render(runtime_type, &context)?,
        })
    }

    fn create_fusion_group(&self, prog_id: &str, target: &NodeInfo) -> GroupRenderInfo {
        GroupRenderInfo {
            prog_id: prog_id.to_string(),
            shape: format!("{:?}", target.dims),
            outer_loops: (0..target.dims.len()).map(|d| LoopInfo { var: format!("i{}", d), limit: target.dims[d].clone() }).collect(),
            is_parallel: !target.dims.is_empty(),
            fusion_ops: Vec::new(),
            kernel: None,
        }
    }

    fn create_reduction_group(&self, prog_id: &str, node: &NodeInfo) -> GroupRenderInfo {
        if let Op::ReduceSum { input, axis } = &node.op {
            let in_node = self.get_node_by_id(prog_id, input);
            let in_dims: Vec<String> = in_node.shape.dims.iter().map(|d| d.to_string()).collect();
            let rank = in_node.shape.rank();
            let target_idx = self.generate_target_index_expr(node, rank, axis);
            let in_idx = self.generate_index_expr(prog_id, in_node, rank, &in_dims);

            return GroupRenderInfo {
                prog_id: prog_id.to_string(),
                shape: format!("Reduction of {} axis {}", input, axis),
                outer_loops: (0..rank).filter(|d| d != axis).map(|d| LoopInfo { var: format!("i{}", d), limit: in_dims[d].clone() }).collect(),
                is_parallel: rank > 1,
                fusion_ops: vec![],
                kernel: Some(KernelInfo {
                    init: format!("buffer_{}_{}[{}] = 0.0f;", prog_id, crate::model::sanitize_id(&node.node_id), target_idx),
                    inner_loops: vec![LoopInfo { var: format!("i{}", axis), limit: in_dims[*axis].clone() }],
                    body: format!("buffer_{}_{}[{}] += buffer_{}_{}[{}];", prog_id, crate::model::sanitize_id(&node.node_id), target_idx, prog_id, crate::model::sanitize_id(input), in_idx),
                    finalize: "".into(),
                }),
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
            let l_strides = left_node.get_effective_strides_c_expr();
            let r_strides = right_node.get_effective_strides_c_expr();
            let target_idx = format!("i * ({}) + j * ({})", node.strides[0], node.strides[1]);

            return GroupRenderInfo {
                prog_id: prog_id.to_string(),
                shape: format!("MatMul {} x {}", left, right),
                outer_loops: vec![
                    LoopInfo { var: "i".into(), limit: left_dims[0].clone() },
                    LoopInfo { var: "j".into(), limit: right_dims[1].clone() },
                ],
                is_parallel: true,
                fusion_ops: vec![],
                kernel: Some(KernelInfo {
                    init: "float acc = 0.0f;".into(),
                    inner_loops: vec![LoopInfo { var: "k".into(), limit: left_dims[1].clone() }],
                    body: format!("acc += buffer_{}_{}[i * ({}) + k * ({})] * buffer_{}_{}[k * ({}) + j * ({})];", prog_id, crate::model::sanitize_id(left), l_strides[0], l_strides[1], prog_id, crate::model::sanitize_id(right), r_strides[0], r_strides[1]),
                    finalize: format!("buffer_{}_{}[{}] = acc;", prog_id, crate::model::sanitize_id(&node.node_id), target_idx),
                }),
            };
        }
        panic!("Not a matmul op");
    }

    fn create_conv_group(&self, prog_id: &str, node: &NodeInfo) -> GroupRenderInfo {
        if let Op::Conv { input, kernel } = &node.op {
            let in_node = self.get_node_by_id(prog_id, input);
            let ker_node = self.get_node_by_id(prog_id, kernel);
            let ker_dims: Vec<String> = ker_node.shape.dims.iter().map(|d| d.to_string()).collect();
            let in_rank = in_node.shape.rank();
            let ker_rank = ker_node.shape.rank();
            let out_rank = node.dims.len();
            let target_idx = (0..out_rank).map(|d| format!("i{} * ({})", d, node.strides[d])).collect::<Vec<_>>().join(" + ");
            
            let in_strides = in_node.get_effective_strides_c_expr();
            let mut in_parts = Vec::new();
            for d in 0..in_rank {
                if d < ker_rank { in_parts.push(format!("(i{} + k{}) * ({})", d, d, in_strides[d])); }
                else { in_parts.push(format!("i{} * ({})", d, in_strides[d])); }
            }
            let in_idx = if in_parts.is_empty() { "0".into() } else { in_parts.join(" + ") };
            let ker_strides = ker_node.get_effective_strides_c_expr();
            let ker_idx = (0..ker_rank).map(|d| format!("k{} * ({})", d, ker_strides[d])).collect::<Vec<_>>().join(" + ");

            return GroupRenderInfo {
                prog_id: prog_id.to_string(),
                shape: format!("Conv {} by {}", input, kernel),
                outer_loops: (0..out_rank).map(|d| LoopInfo { var: format!("i{}", d), limit: node.dims[d].clone() }).collect(),
                is_parallel: true,
                fusion_ops: vec![],
                kernel: Some(KernelInfo {
                    init: "float acc = 0.0f;".into(),
                    inner_loops: (0..ker_rank).map(|d| LoopInfo { var: format!("k{}", d), limit: ker_dims[d].clone() }).collect(),
                    body: format!("acc += buffer_{}_{}[{}] * buffer_{}_{}[{}];", prog_id, crate::model::sanitize_id(input), in_idx, prog_id, crate::model::sanitize_id(kernel), if ker_idx.is_empty() { "0".into() } else { ker_idx }),
                    finalize: format!("buffer_{}_{}[{}] = acc;", prog_id, crate::model::sanitize_id(&node.node_id), target_idx),
                }),
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
        let target_idx = if rank == 0 { "0".into() } else {
            (0..rank).map(|d| format!("i{} * ({})", d, node.strides[d])).collect::<Vec<_>>().join(" + ")
        };

        if let Op::Transpose { input, permutation } = &node.op {
            let in_node = self.get_node_by_id(prog_id, input);
            let in_strides = in_node.get_effective_strides_c_expr();
            let in_idx = permutation.iter().enumerate()
                .map(|(n, &p_n)| format!("i{} * ({})", n, in_strides[p_n]))
                .collect::<Vec<_>>().join(" + ");
            return format!("buffer_{}_{}[{}] = buffer_{}_{}[{}];", 
                prog_id, crate::model::sanitize_id(&node.node_id), target_idx, prog_id, crate::model::sanitize_id(input), if in_idx.is_empty() { "0".into() } else { in_idx });
        }

        node.op.generate_c_body(prog_id, &node.node_id, &target_idx, |dep_id| {
            let dep_node = self.get_node_by_id(prog_id, dep_id);
            self.generate_index_expr(prog_id, dep_node, rank, &node.dims)
        })
    }

    fn generate_index_expr(&self, _prog_id: &str, node: &crate::model::Node, target_rank: usize, target_dims: &[String]) -> String {
        let rank = node.shape.rank();
        let strides = node.get_effective_strides_c_expr();
        let node_dims: Vec<String> = node.shape.dims.iter().map(|d| d.to_string()).collect();
        let mut parts = Vec::new();

        // 1. Try to match dimensions by name/value (for Reshape and smarter broadcasting)
        let mut node_to_target = vec![None; rank];
        let mut used_target_dims = vec![false; target_rank];

        for d in 0..rank {
            if matches!(node.shape.dims[d], crate::model::Dimension::Value(1)) { continue; }
            
            // Look for a matching dimension in the target loop
            for td in 0..target_rank {
                if !used_target_dims[td] && node_dims[d] == target_dims[td] {
                    node_to_target[d] = Some(td);
                    used_target_dims[td] = true;
                    break;
                }
            }
        }

        // 2. Check if all non-1 dimensions were matched
        let all_matched = (0..rank).all(|d| 
            matches!(node.shape.dims[d], crate::model::Dimension::Value(1)) || node_to_target[d].is_some()
        );

        if all_matched {
            for d in 0..rank {
                if let Some(td) = node_to_target[d] {
                    parts.push(format!("i{} * ({})", td, strides[d]));
                }
            }
        } else {
            // 3. Fallback to standard right-aligned broadcasting
            for d in 0..rank {
                let target_d = (target_rank as isize - rank as isize + d as isize) as usize;
                if target_d < target_rank {
                    if !matches!(node.shape.dims[d], crate::model::Dimension::Value(1)) {
                        parts.push(format!("i{} * ({})", target_d, strides[d]));
                    }
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
        prog.compiler.graph.node_indices()
            .map(|idx| &prog.compiler.graph[idx])
            .find(|n| n.id == node_id)
            .expect("Node not found")
    }
}
