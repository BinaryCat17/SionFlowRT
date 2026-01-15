use crate::analyzer::ProjectPlan;

pub fn generate_runtime_c(plan: &ProjectPlan) -> String {
    let mut c = String::new();

    c.push_str("#include <stdint.h>\n");
    c.push_str("#include <stdbool.h>\n");
    c.push_str("#include <stdlib.h>\n");
    c.push_str("#include <string.h>\n\n");

    c.push_str("/* --- Module Declarations --- */\n");
    for (prog_id, interface) in &plan.programs {
        let mut args = Vec::new();
        let mut in_names: Vec<_> = interface.inputs.keys().collect();
        in_names.sort();
        for name in in_names {
            let port = &interface.inputs[name];
            args.push(format!("const {}* restrict in_{}", port.dtype.to_c_type(), sanitize_id(name)));
        }
        let mut out_names: Vec<_> = interface.outputs.keys().collect();
        out_names.sort();
        for name in out_names {
            let port = &interface.outputs[name];
            args.push(format!("{}* restrict out_{}", port.dtype.to_c_type(), sanitize_id(name)));
        }
        c.push_str(&format!("extern void {}({});\n", sanitize_id(prog_id), args.join(", ")));
    }
    c.push_str("\n");

    c.push_str("/* --- Global Resources --- */\n");
    for (res_id, res) in &plan.resources {
        let kind_info = res.kind.as_ref().map(|k| format!(" // Type: {}", k)).unwrap_or_default();
        c.push_str(&format!("{}* resource_{} = NULL;{}\n", res.dtype.to_c_type(), sanitize_id(res_id), kind_info));
    }
    c.push_str("\n");

    c.push_str("void reallocate_resources() {\n");
    for (res_id, res) in &plan.resources {
        let size_expr = res.shape.to_c_size_expr();
        let id = sanitize_id(res_id);
        let c_type = res.dtype.to_c_type();
        c.push_str(&format!("    resource_{} = ({}*)realloc(resource_{}, sizeof({}) * ({}));\n", 
            id, c_type, id, c_type, size_expr));
        c.push_str(&format!("    memset(resource_{}, 0, sizeof({}) * ({}));\n", id, c_type, size_expr));
    }
    c.push_str("}\n\n");

    c.push_str("void execute_all() {\n");
    for prog_id in &plan.execution_order {
        let interface = &plan.programs[prog_id];
        let mut call_args = Vec::new();

        let mut in_names: Vec<_> = interface.inputs.keys().collect();
        in_names.sort();
        for name in in_names {
            call_args.push(format!("resource_{}", sanitize_id(name)));
        }

        let mut out_names: Vec<_> = interface.outputs.keys().collect();
        out_names.sort();
        for name in out_names {
            call_args.push(format!("resource_{}", sanitize_id(name)));
        }

        c.push_str(&format!("    {}({});\n", sanitize_id(prog_id), call_args.join(", ")));
    }
    c.push_str("}\n");

    c
}

fn sanitize_id(id: &str) -> String {
    id.replace("/", "_").replace(".", "_")
}
