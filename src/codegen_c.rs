use crate::model::{Op, TensorShape, sanitize_id};
use crate::linear_ir::LinearIR;
use crate::orchestrator::ProjectOrchestration;
use crate::pipeline::{Stage, CompilerContext};
use tera::{Tera, Context};
use std::collections::HashMap;

pub struct CodegenStage;
impl Stage for CodegenStage {
    fn name(&self) -> &str { "C Code Generation" }
    fn run(&self, ctx: &mut CompilerContext) -> anyhow::Result<()> {
        let orchestration = ctx.orchestration.as_ref().ok_or_else(|| anyhow::anyhow!("No orchestration data"))?;
        let codegen = CodegenC::new(orchestration.programs.clone(), orchestration.clone(), ctx.parameters.clone());
        let gen_code = codegen.generate("sdl2_runtime")?;
        
        ctx.generated_module = Some(gen_code.module);
        ctx.generated_runtime = Some(gen_code.runtime);
        Ok(())
    }
}

pub struct CodegenC {
    programs: HashMap<String, LinearIR>,
    orchestration: ProjectOrchestration,
    parameters: HashMap<String, usize>,
    tera: Tera,
}

pub struct GeneratedCode {
    pub module: String,
    pub runtime: String,
}

fn generate_tensor_size_c_expr(shape: &TensorShape) -> anyhow::Result<String> {
    if shape.dims.is_empty() { return Ok("1".to_string()); }
    let mut parts = Vec::new();
    for d in &shape.dims {
        if d.is_wildcard() || d.is_ellipsis() {
            return Err(anyhow::anyhow!("Unresolved dimension '{}' in shape {:?}", d, shape));
        }
        parts.push(d.to_string());
    }
    Ok(parts.join(" * "))
}

fn generate_op_c_body(op: &Op, prog_id: &str, node_id: &str, target_idx: &str, inputs: &[String]) -> String {
    let target = format!("buffer_{}_{}[{}]", prog_id, sanitize_id(node_id), target_idx);
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
        orchestration: ProjectOrchestration,
        parameters: HashMap<String, usize>
    ) -> Self {
        let mut tera = Tera::default();
        tera.add_raw_template("module", &std::fs::read_to_string("templates/module.c").unwrap_or_default()).unwrap();
        tera.add_raw_template("sdl2_runtime", &std::fs::read_to_string("templates/sdl2_runtime.c").unwrap_or_default()).unwrap();
        tera.add_raw_template("headless_runtime", &std::fs::read_to_string("templates/headless_runtime.c").unwrap_or_default()).unwrap();
        Self { programs, orchestration, parameters, tera }
    }

    pub fn generate(&self, runtime_template: &str) -> anyhow::Result<GeneratedCode> {
        let mut context = Context::new();
        let mut programs_data = Vec::new();
        let mut all_nodes = Vec::new();
        let mut nodes_map = HashMap::new();

        for (prog_id, prog) in &self.programs {
            let mut prog_nodes_map = HashMap::new();
            
            for node in &prog.nodes {
                let size_expr = generate_tensor_size_c_expr(&node.shape)?;
                let node_json = serde_json::json!({
                    "id": sanitize_id(&node.id),
                    "node_id": sanitize_id(&node.id),
                    "prog_id": prog_id,
                    "size": size_expr,
                    "is_input": matches!(node.op, Op::Input { .. }),
                    "is_output": matches!(node.op, Op::Output { .. }),
                    "c_type": node.dtype.to_c_type()
                });
                prog_nodes_map.insert(node.id.clone(), node_json);
            }

            let instance = self.orchestration.instances.iter().find(|i| &i.id == prog_id);

            let mut loop_groups = Vec::new();
            for group in &prog.groups {
                let mut group_nodes = Vec::new();
                let mut size_expr = "0".to_string();
                let mut is_fusible = true;
                let mut group_has_body = false;

                for &node_idx in group {
                    let node = &prog.nodes[node_idx];
                    let mut inputs_exprs = Vec::new();
                    for dep_id in &node.inputs {
                        inputs_exprs.push(format!("buffer_{}_{}[i]", prog_id, sanitize_id(dep_id)));
                    }

                    let body = generate_op_c_body(&node.op, prog_id, &node.id, "i", &inputs_exprs);
                    if !body.is_empty() { group_has_body = true; }
                    size_expr = generate_tensor_size_c_expr(&node.shape)?;
                    
                    if matches!(node.op, Op::Input { .. } | Op::Constant { .. }) {
                        is_fusible = false;
                    }

                    group_nodes.push(serde_json::json!({
                        "id": sanitize_id(&node.id),
                        "body": body,
                        "is_input": matches!(node.op, Op::Input { .. })
                    }));
                }

                loop_groups.push(serde_json::json!({
                    "nodes": group_nodes,
                    "size": size_expr,
                    "is_fusible": is_fusible && !group_nodes.is_empty(),
                    "has_body": group_has_body
                }));
            }

            for (alias, real_id) in &prog.outputs {
                if let Some(real_node) = prog_nodes_map.get(real_id).cloned() {
                    let mut alias_node = real_node.clone();
                    alias_node["id"] = serde_json::json!(sanitize_id(alias));
                    prog_nodes_map.insert(alias.clone(), alias_node);
                }
            }

            programs_data.push(serde_json::json!({
                "id": prog_id,
                "loop_groups": loop_groups,
                "instance": instance,
                "outputs": prog.outputs.iter().map(|(k, v)| serde_json::json!({"alias": sanitize_id(k), "real_id": sanitize_id(v)})).collect::<Vec<_>>()
            }));
            
            for node_json in prog_nodes_map.values() {
                all_nodes.push(node_json.clone());
            }
            nodes_map.insert(prog_id.clone(), prog_nodes_map);
        }

        context.insert("programs", &programs_data);
        context.insert("nodes", &all_nodes);
        context.insert("nodes_map", &nodes_map);
        context.insert("orchestration", &self.orchestration);
        context.insert("parameters", &self.parameters);
        
        Ok(GeneratedCode {
            module: self.tera.render("module", &context)?,
            runtime: self.tera.render(runtime_template, &context)?,
        })
    }
}
