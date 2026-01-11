use crate::model::{Node, Op};
use crate::manifest::Manifest;
use crate::CompiledProgram;
use tera::{Tera, Context};
use std::collections::HashMap;

pub struct CodegenC<'a> {
    programs: HashMap<String, CompiledProgram>,
    _manifest: &'a Manifest,
    tera: Tera,
}

pub struct GeneratedCode {
    pub module: String,
    pub runtime: String,
}

impl<'a> CodegenC<'a> {
    pub fn new(programs: HashMap<String, CompiledProgram>, manifest: &'a Manifest) -> Self {
        let mut tera = Tera::default();
        tera.add_raw_template("module", include_str!("../templates/module.c")).unwrap();
        tera.add_raw_template("sdl2_runtime", include_str!("../templates/sdl2_runtime.c")).unwrap();
        tera.add_raw_template("headless_runtime", include_str!("../templates/headless_runtime.c")).unwrap();
        Self { programs, _manifest: manifest, tera }
    }

    pub fn generate(&self, runtime_template: &str) -> anyhow::Result<GeneratedCode> {
        let mut context = Context::new();
        let mut programs_data = Vec::new();
        let mut all_nodes = Vec::new();
        let mut nodes_map = HashMap::new();

        for (prog_id, prog) in &self.programs {
            let mut nodes_data = Vec::new();
            let mut prog_nodes_map = HashMap::new();

            for &idx in &prog.execution_order {
                let node = &prog.compiler.graph[idx];
                
                let mut inputs_exprs = Vec::new();
                for dep_id in &node.inputs {
                    let dep_node = self.get_node_by_id(prog_id, dep_id);
                    inputs_exprs.push(self.generate_buffer_access(prog_id, dep_node));
                }

                let body = node.op.generate_c_body(prog_id, &node.id, "i", &inputs_exprs);
                let size_expr = node.shape.size_c_expr();
                let is_input = matches!(node.op, Op::Input { .. });
                let is_output = matches!(node.op, Op::Output { .. });

                let node_json = serde_json::json!({
                    "id": crate::model::sanitize_id(&node.id),
                    "node_id": crate::model::sanitize_id(&node.id),
                    "prog_id": prog_id,
                    "size": size_expr,
                    "size_expr": size_expr,
                    "body": body,
                    "is_input": is_input,
                    "is_output": is_output,
                    "is_stateful": false, // TODO
                    "c_type": "float"
                });

                nodes_data.push(node_json.clone());
                all_nodes.push(node_json.clone());
                prog_nodes_map.insert(node.id.clone(), node_json);
            }
            programs_data.push(serde_json::json!({ "id": prog_id, "nodes": nodes_data }));
            nodes_map.insert(prog_id.clone(), prog_nodes_map);
        }

        context.insert("programs", &programs_data);
        context.insert("nodes", &all_nodes);
        context.insert("nodes_map", &nodes_map);
        context.insert("parameters", &self._manifest.parameters);
        context.insert("mappings", &self._manifest.mappings);
        
        Ok(GeneratedCode {
            module: self.tera.render("module", &context)?,
            runtime: self.tera.render(runtime_template, &context)?,
        })
    }

    fn get_node_by_id(&self, prog_id: &str, node_id: &str) -> &Node {
        let prog = &self.programs[prog_id];
        let idx = *prog.compiler.node_map.get(node_id).unwrap();
        &prog.compiler.graph[idx]
    }

    fn generate_buffer_access(&self, prog_id: &str, node: &Node) -> String {
        format!("buffer_{}_{}[i]", prog_id, crate::model::sanitize_id(&node.id))
    }
}
