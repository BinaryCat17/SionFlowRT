use crate::linear_ir::LinearIR;
use crate::manifest::Manifest;
use crate::model::{Op, TensorShape, Dimension};
use std::collections::HashMap;
use serde::Serialize;

#[derive(Debug, Serialize, Clone)]
pub struct ProgramBinding {
    pub input_node_id: String,
    pub input_name: String,
    pub source_id: String,
    pub source_type: String, // Добавили тип
    pub shape: Vec<Dimension>,
}

#[derive(Debug, Serialize, Clone)]
pub struct InterProgramLink {
    pub src_prog: String,
    pub src_node: String,
    pub dst_prog: String,
    pub dst_node: String,
    pub size_expr: String,
}

#[derive(Debug, Default, Serialize)]
pub struct LinkPlan {
    pub bindings: Vec<ProgramBinding>,
    pub inter_links: Vec<InterProgramLink>,
    pub display_source: Option<InterProgramLink>, // Куда выводить результат
}

pub struct Linker;

impl Linker {
    pub fn bind_program(
        ir: &mut LinearIR, 
        manifest: &Manifest, 
        program_id: &str
    ) -> anyhow::Result<LinkPlan> {
        let mut plan = LinkPlan::default();
        let mut input_to_source = HashMap::new();

        for (src_id, dst_addr) in &manifest.links {
            let parts: Vec<&str> = dst_addr.split('.').collect();
            let is_target = if parts.len() == 2 {
                parts[0] == program_id
            } else {
                dst_addr == program_id
            };

            if is_target {
                let input_name = parts.last().unwrap().to_string();
                input_to_source.insert(input_name, src_id.clone());
            }
            
            // Проверка на Display (выход)
            if dst_addr.starts_with("sources.") {
                let clean_dst = dst_addr.strip_prefix("sources.").unwrap();
                if let Some(source_def) = manifest.sources.get(clean_dst) {
                    if source_def.source_type == "Display" {
                        let src_parts: Vec<&str> = src_id.split('.').collect();
                        if src_parts.len() == 2 && src_parts[0] == program_id {
                            plan.display_source = Some(InterProgramLink {
                                src_prog: src_parts[0].to_string(),
                                src_node: src_parts[1].to_string(),
                                dst_prog: "display".into(),
                                dst_node: "display".into(),
                                size_expr: "0".into(), // Пока не важно
                            });
                        }
                    }
                }
            }
        }

        // 2. Проставляем формы и формируем план
        for node in &mut ir.nodes {
            if let Op::Input { name } = &node.op {
                if let Some(src_id) = input_to_source.get(name) {
                    if let Some(source_name) = src_id.strip_prefix("sources.") {
                        if let Some(source_def) = manifest.sources.get(source_name) {
                            node.shape = TensorShape { dims: source_def.shape.clone() };
                            plan.bindings.push(ProgramBinding {
                                input_node_id: node.id.clone(),
                                input_name: name.clone(),
                                source_id: source_name.to_string(),
                                source_type: source_def.source_type.clone(),
                                shape: source_def.shape.clone(),
                            });
                        }
                    } else {
                        // Межпрограммная связь
                        let src_parts: Vec<&str> = src_id.split('.').collect();
                        if src_parts.len() == 2 {
                            plan.inter_links.push(InterProgramLink {
                                src_prog: src_parts[0].to_string(),
                                src_node: src_parts[1].to_string(),
                                dst_prog: program_id.to_string(),
                                dst_node: node.id.clone(),
                                size_expr: "0".into(), // Будет заполнено позже или в шаблоне
                            });
                        }
                    }
                }
            }
        }

        Ok(plan)
    }
}
