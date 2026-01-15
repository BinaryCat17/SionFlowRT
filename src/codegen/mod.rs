use crate::linearizer::ir::{LinearIR, LinearNode};
use crate::core::op::Op;

pub fn generate_module_header(module_id: &str, ir: &LinearIR) -> String {
    let mut h = String::new();
    h.push_str("#ifndef GENERATED_MODULE_H\n#define GENERATED_MODULE_H\n\n");
    h.push_str("#include <stdint.h>\n#include <stdbool.h>\n\n");
    
    let args = get_function_args(ir);
    h.push_str(&format!("void {}({});\n\n", module_id, args.join(", ")));
    
    h.push_str("#endif\n");
    h
}

pub fn generate_module_source(module_id: &str, ir: &LinearIR) -> String {
    let mut c = String::new();
    c.push_str(&format!("#include \"{}.h\"\n", module_id));
    c.push_str("#include <math.h>\n");
    c.push_str("#ifdef _OPENMP\n#include <omp.h>\n#endif\n\n");

    let args = get_function_args(ir);
    c.push_str(&format!("void {}({}) {{ \n", module_id, args.join(", ")));

    for node in &ir.nodes {
        if matches!(node.op, Op::Input { .. } | Op::Output { .. }) { continue; }
        let size_expr = node.shape.to_c_size_expr();
        let c_type = node.dtype.to_c_type();
        let id = sanitize_id(&node.id);
        c.push_str(&format!("    static {} {}[{}] __attribute__((aligned(64)));\n", c_type, id, size_expr));
    }

    c.push_str("\n");

    for node in &ir.nodes {
        emit_node_code(&mut c, node, ir);
    }

    c.push_str("}\n");
    c
}

fn get_function_args(ir: &LinearIR) -> Vec<String> {
    let mut args = Vec::new();
    for input in &ir.inputs {
        args.push(format!("const {}* restrict in_{}", input.dtype.to_c_type(), sanitize_id(&input.name)));
    }
    let mut out_ports: Vec<_> = ir.outputs.keys().collect();
    out_ports.sort();
    for port in out_ports {
        args.push(format!("float* restrict out_{}", sanitize_id(port)));
    }
    args
}

fn emit_node_code(c: &mut String, node: &LinearNode, ir: &LinearIR) {
    let node_var = sanitize_id(&node.id);
    let size_expr = node.shape.to_c_size_expr();

    match &node.op {
        Op::Input { name } => {
            c.push_str(&format!("    // Input {} handled via args\n", name));
        }
        Op::Add => {
            let left = get_input_var(&node.inputs[0].0, ir);
            let right = get_input_var(&node.inputs[1].0, ir);
            c.push_str("    #pragma omp parallel for simd\n");
            c.push_str(&format!("    for (int i = 0; i < {}; i++) {{ {}[i] = {}[i] + {}[i]; }}\n", size_expr, node_var, left, right));
        }
        Op::Constant { values } => {
            for (i, v) in values.iter().enumerate() {
                c.push_str(&format!("    {}[{}] = {}f;\n", node_var, i, v));
            }
        }
        Op::Output { name } => {
            let src = get_input_var(&node.inputs[0].0, ir);
            c.push_str("    #pragma omp parallel for simd\n");
            c.push_str(&format!("    for (int i = 0; i < {}; i++) {{ out_{}[i] = {}[i]; }}\n", size_expr, sanitize_id(name), src));
        }
        _ => {
            c.push_str(&format!("    // Op {:?} not implemented\n", node.op));
        }
    }
}

fn get_input_var(node_id: &str, ir: &LinearIR) -> String {
    for input in &ir.inputs {
        if input.name == node_id {
            return format!("in_{}", sanitize_id(&input.name));
        }
    }
    sanitize_id(node_id)
}

fn sanitize_id(id: &str) -> String {
    id.replace("/", "_").replace(".", "_")
}
