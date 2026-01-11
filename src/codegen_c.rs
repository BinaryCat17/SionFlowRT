use crate::model::{Op, TensorShape, sanitize_id};
use crate::linear_ir::LinearIR;
use crate::linker::LinkPlan;
use tera::{Tera, Context};
use std::collections::HashMap;

pub struct CodegenC {
    programs: HashMap<String, LinearIR>,
    link_plans: HashMap<String, LinkPlan>,
    parameters: HashMap<String, usize>,
    tera: Tera,
}

pub struct GeneratedCode {
    pub module: String,
    pub runtime: String,
}

fn generate_tensor_size_c_expr(shape: &TensorShape) -> String {
    if shape.dims.is_empty() { "1".to_string() }
    else { shape.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" * ") }
}

fn generate_op_c_body(op: &Op, prog_id: &str, node_id: &str, target_idx: &str, inputs: &[String]) -> String {
    let target = format!("buffer_{}_{}[{}]", prog_id, sanitize_id(node_id), target_idx);
    
    // Safety check
    let get_in = |i: usize| inputs.get(i).cloned().unwrap_or_else(|| "0.0f".to_string());

    match op {
        Op::Sin => format!("{} = sinf({});", target, get_in(0)),
        Op::Abs => format!("{} = fabsf({});", target, get_in(0)),
        Op::Sqrt => format!("{} = sqrtf({});", target, get_in(0)),
        Op::Square => format!("{} = ({}) * ({});", target, get_in(0), get_in(0)),
        Op::Exp => format!("{} = expf({});", target, get_in(0)),
        Op::Log => format!("{} = logf({});", target, get_in(0)),
        
        Op::Add => format!("{} = {} + {};", target, get_in(0), get_in(1)),
        Op::Sub => format!("{} = {} - {};", target, get_in(0), get_in(1)),
        Op::Mul => format!("{} = {} * {};", target, get_in(0), get_in(1)),
        Op::Div => format!("{} = {} / ({} + 1e-9f);", target, get_in(0), get_in(1)),
        Op::Min => format!("{} = fminf({}, {});", target, get_in(0), get_in(1)),
        Op::Max => format!("{} = fmaxf({}, {});", target, get_in(0), get_in(1)),
        Op::Pow => format!("{} = powf({}, {});", target, get_in(0), get_in(1)),
        
        _ => "".to_string(),
    }
}

impl CodegenC {
    pub fn new(
        programs: HashMap<String, LinearIR>, 
        link_plans: HashMap<String, LinkPlan>,
        parameters: HashMap<String, usize>
    ) -> Self {
        let mut tera = Tera::default();
        tera.add_raw_template("module", &include_raw("templates/module.c")).unwrap();
        tera.add_raw_template("sdl2_runtime", &include_raw("templates/sdl2_runtime.c")).unwrap();
        tera.add_raw_template("headless_runtime", &include_raw("templates/headless_runtime.c")).unwrap();
        Self { programs, link_plans, parameters, tera }
    }

    pub fn generate(&self, runtime_template: &str) -> anyhow::Result<GeneratedCode> {
        let mut context = Context::new();
        let mut programs_data = Vec::new();
        let mut all_nodes = Vec::new();
        let mut nodes_map = HashMap::new();

        for (prog_id, prog) in &self.programs {
            let mut nodes_data = Vec::new();
            let mut prog_nodes_map = HashMap::new();

            for node in &prog.nodes {
                let mut inputs_exprs = Vec::new();
                for dep_id in &node.inputs {
                    inputs_exprs.push(format!("buffer_{}_{}[i]", prog_id, sanitize_id(dep_id)));
                }

                let body = generate_op_c_body(&node.op, prog_id, &node.id, "i", &inputs_exprs);
                let size_expr = generate_tensor_size_c_expr(&node.shape);
                let is_input = matches!(node.op, Op::Input { .. });
                let is_output = matches!(node.op, Op::Output { .. });

                let node_json = serde_json::json!({
                    "id": sanitize_id(&node.id),
                    "node_id": sanitize_id(&node.id),
                    "prog_id": prog_id,
                    "size": size_expr,
                    "size_expr": size_expr,
                    "body": body,
                    "is_input": is_input,
                    "is_output": is_output,
                    "is_stateful": false,
                    "c_type": node.dtype.to_c_type()
                });

                nodes_data.push(node_json.clone());
                all_nodes.push(node_json.clone());
                prog_nodes_map.insert(node.id.clone(), node_json);
            }

            // Добавляем алиасы для выходов
            for (alias, real_id) in &prog.outputs {
                if let Some(real_node) = prog_nodes_map.get(real_id).cloned() {
                    let mut alias_node = real_node.clone();
                    alias_node["id"] = serde_json::json!(sanitize_id(alias));
                    alias_node["node_id"] = serde_json::json!(sanitize_id(alias));
                    alias_node["is_alias"] = serde_json::json!(true);
                    alias_node["real_id"] = real_node["id"].clone();
                    prog_nodes_map.insert(alias.clone(), alias_node);
                }
            }

            programs_data.push(serde_json::json!({
                "id": prog_id,
                "nodes": nodes_data,
                "link_plan": self.link_plans.get(prog_id),
                "outputs": prog.outputs.iter().map(|(k, v)| serde_json::json!({"alias": sanitize_id(k), "real_id": sanitize_id(v)})).collect::<Vec<_>>()
            }));
            nodes_map.insert(prog_id.clone(), prog_nodes_map);
        }

        context.insert("programs", &programs_data);
        context.insert("nodes", &all_nodes);
        context.insert("nodes_map", &nodes_map);
        context.insert("parameters", &self.parameters);
        
        Ok(GeneratedCode {
            module: self.tera.render("module", &context)?,
            runtime: self.tera.render(runtime_template, &context)?,
        })
    }
}

// Вспомогательная функция, так как tera хочет &str
fn include_raw(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}