use crate::linearizer::ir::{LinearIR, LinearNode, InputConnection};
use crate::core::op::Op;
use crate::core::utils::sanitize_id;

pub fn generate_module_source(module_id: &str, ir: &LinearIR) -> String {
    let mut c = String::new();
    
    // Header includes
    c.push_str("#include \"{}.h\"\n".replace("{}", module_id).as_str());
    c.push_str("#include <math.h>\n");
    c.push_str("#ifdef _OPENMP\n#include <omp.h>\n#endif\n\n");

    let args = get_function_args(ir);
    let func_sig = "void {}({}) {{ \n".replace("{}", module_id).replace("{}", &args.join(", "));
    c.push_str(&func_sig);

    // Workspace pointers casting
    for node in &ir.nodes {
        if matches!(node.op, Op::Input { .. } | Op::Output { .. }) { continue; }
        let c_type = node.dtype.to_c_type();
        let id = sanitize_id(&node.id);
        let cast = "    {}* restrict {} = ({}*)workspace[{}];\n"
            .replace("{}", c_type)
            .replace("{}", &id)
            .replace("{}", c_type)
            .replace("{}", &node.offset.to_string());
        c.push_str(&cast);
    }

    c.push_str("\n");

    for node in &ir.nodes {
        emit_node_code(&mut c, node, ir);
    }

    c.push_str("}\n");
    c
}

pub fn generate_module_header(module_id: &str, ir: &LinearIR) -> String {
    let mut c = String::new();
    let guard = format!("{}_H", module_id.to_uppercase());
    
    let header = "#ifndef {}
#define {}

#include <stdint.h>

"
        .replace("{}", &guard)
        .replace("{}", &guard);
    c.push_str(&header);

    let args = get_function_args(ir);
    let decl = "void {}({});\n\n"
        .replace("{}", module_id)
        .replace("{}", &args.join(", "));
    c.push_str(&decl);

    c.push_str("#endif\n");
    c
}

fn get_function_args(ir: &LinearIR) -> Vec<String> {
    let mut args = Vec::new();
    args.push("void** workspace".to_string());

    for input in &ir.inputs {
        args.push("const {}* restrict in_{}"
            .replace("{}", input.dtype.to_c_type())
            .replace("{}", &sanitize_id(&input.name)));
    }
    
    for port in &ir.outputs {
        args.push("{}* restrict out_{}"
            .replace("{}", port.dtype.to_c_type())
            .replace("{}", &sanitize_id(&port.name)));
    }
    args
}

fn emit_node_code(c: &mut String, node: &LinearNode, _ir: &LinearIR) {
    let node_var = sanitize_id(&node.id);
    let size_expr = node.shape.to_c_size_expr();

    match &node.op {
        Op::Input { name } => {
            c.push_str("    // Input {} handled via args\n".replace("{}", name).as_str());
        }
        Op::Constant { values } => {
            for (i, v) in values.iter().enumerate() {
                c.push_str("    {}[{{}}] = {{}}f;\n"
                    .replace("{}", &node_var)
                    .replace("{}", &i.to_string())
                    .replace("{}", &v.to_string()).as_str());
            }
        }
        Op::Output { name } => {
            let src = get_input_var(&node.inputs[0]);
            c.push_str("    #pragma omp parallel for simd\n");
            c.push_str("    for (int i = 0; i < {}; i++) {{ out_{}[i] = {}[i]; }}\n"
                .replace("{}", &size_expr)
                .replace("{}", &sanitize_id(name))
                .replace("{}", &src).as_str());
        }
        Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Min | Op::Max | Op::Pow => {
            let left = get_input_var(&node.inputs[0]);
            let right = get_input_var(&node.inputs[1]);
            let op_sym = match node.op {
                Op::Add => "+",
                Op::Sub => "-",
                Op::Mul => "*",
                Op::Div => "/",
                _ => "",
            };

            c.push_str("    #pragma omp parallel for simd\n");
            if !op_sym.is_empty() {
                c.push_str("    for (int i = 0; i < {}; i++) {{ {}[i] = {}[i] {} {}[i]; }}\n"
                    .replace("{}", &size_expr)
                    .replace("{}", &node_var)
                    .replace("{}", &left)
                    .replace("{}", op_sym)
                    .replace("{}", &right).as_str());
            } else {
                let func = match node.op {
                    Op::Min => "fminf",
                    Op::Max => "fmaxf",
                    Op::Pow => "powf",
                    _ => unreachable!(),
                };
                c.push_str("    for (int i = 0; i < {}; i++) {{ {}[i] = {} ({}[i], {}[i]); }}\n"
                    .replace("{}", &size_expr)
                    .replace("{}", &node_var)
                    .replace("{}", func)
                    .replace("{}", &left)
                    .replace("{}", &right).as_str());
            }
        }
        Op::Sin | Op::Abs | Op::Sqrt | Op::Square | Op::Exp | Op::Log => {
            let src = get_input_var(&node.inputs[0]);
            let func = match node.op {
                Op::Sin => "sinf",
                Op::Abs => "fabsf",
                Op::Sqrt => "sqrtf",
                Op::Exp => "expf",
                Op::Log => "logf",
                Op::Square => "",
                _ => unreachable!(),
            };
            c.push_str("    #pragma omp parallel for simd\n");
            if func.is_empty() { // Square
                c.push_str("    for (int i = 0; i < {}; i++) {{ {}[i] = {}[i] * {}[i]; }}\n"
                    .replace("{}", &size_expr)
                    .replace("{}", &node_var)
                    .replace("{}", &src)
                    .replace("{}", &src).as_str());
            } else {
                c.push_str("    for (int i = 0; i < {}; i++) {{ {}[i] = {} ({}[i]); }}\n"
                    .replace("{}", &size_expr)
                    .replace("{}", &node_var)
                    .replace("{}", func)
                    .replace("{}", &src).as_str());
            }
        }
        Op::Reshape { .. } => {
            let src = get_input_var(&node.inputs[0]);
            c.push_str("    #pragma omp parallel for simd\n");
            c.push_str("    for (int i = 0; i < {}; i++) {{ {}[i] = {}[i]; }}\n"
                .replace("{}", &size_expr)
                .replace("{}", &node_var)
                .replace("{}", &src).as_str());
        }
        Op::ReduceSum { axis } => {
            let src = get_input_var(&node.inputs[0]);
            let input_shape = &node.inputs[0].shape;
            
            c.push_str("    for (int i = 0; i < {}; i++) {{ {}[i] = 0.0f; }}\n"
                .replace("{}", &size_expr)
                .replace("{}", &node_var).as_str());
            
            let reduce_dim = input_shape.dims[*axis].to_c_expr();
            let outer_size_raw = input_shape.dims[0..*axis].iter().map(|d| d.to_c_expr()).collect::<Vec<_>>().join(" * ");
            let inner_size_raw = input_shape.dims[*axis+1..].iter().map(|d| d.to_c_expr()).collect::<Vec<_>>().join(" * ");
            
            let outer_size = if outer_size_raw.is_empty() { "1".to_string() } else { outer_size_raw };
            let inner_size = if inner_size_raw.is_empty() { "1".to_string() } else { inner_size_raw };

            let loops = r#"\n    for (int out = 0; out < OUTER * INNER; out++) {\n        int o = out / INNER;\n        int i = out % INNER;\n        for (int r = 0; r < REDUCE; r++) {\n            VAR[o * INNER + i] += SRC[o * REDUCE * INNER + r * INNER + i];\n        }\n    }\n"#
            .replace("OUTER", &outer_size)
            .replace("INNER", &inner_size)
            .replace("REDUCE", &reduce_dim)
            .replace("VAR", &node_var)
            .replace("SRC", &src);
            c.push_str(&loops);
        }
        Op::MatMul => {
            let left = get_input_var(&node.inputs[0]);
            let right = get_input_var(&node.inputs[1]);
            let a_shape = &node.inputs[0].shape;
            let b_shape = &node.inputs[1].shape;

            let m = a_shape.dims[a_shape.dims.len() - 2].to_c_expr();
            let k = a_shape.dims[a_shape.dims.len() - 1].to_c_expr();
            let n = b_shape.dims[b_shape.dims.len() - 1].to_c_expr();
            
            c.push_str("    for (int i = 0; i < {}; i++) {{ {}[i] = 0.0f; }}\n"
                .replace("{}", &size_expr)
                .replace("{}", &node_var).as_str());

            let loops = r#"\n    int batch_size = (SIZE) / ((M) * (N));\n    for (int b = 0; b < batch_size; b++) {\n        for (int i = 0; i < M; i++) {\n            for (int j = 0; j < N; j++) {\n                for (int l = 0; l < K; l++) {\n                    VAR[b * M * N + i * N + j] += LEFT[b * M * K + i * K + l] * RIGHT[b * K * N + l * N + j];\n                }\n            }\n        }\n    }\n"#
            .replace("SIZE", &size_expr)
            .replace("M", &m)
            .replace("N", &n)
            .replace("K", &k)
            .replace("VAR", &node_var)
            .replace("LEFT", &left)
            .replace("RIGHT", &right);
            c.push_str(&loops);
        }
        Op::Split { parts, .. } => {
            let src = get_input_var(&node.inputs[0]);
            c.push_str("    #pragma omp parallel for simd\n");
            c.push_str("    for (int i = 0; i < SIZE * PARTS; i++) {{ VAR[i] = SRC[i]; }}\n"
                .replace("SIZE", &size_expr)
                .replace("PARTS", &parts.to_string())
                .replace("VAR", &node_var)
                .replace("SRC", &src).as_str());
        }
        Op::Transpose { permutation } => {
            let src = get_input_var(&node.inputs[0]);
            let in_shape = &node.inputs[0].shape;
            
            for (i, _) in in_shape.dims.iter().enumerate() {
                let mut line = "    for (int dIDX = 0; dIDX < DIM; dIDX++) {{ \n".to_string();
                line = line.replace("IDX", &i.to_string());
                line = line.replace("DIM", &in_shape.dims[i].to_c_expr());
                c.push_str(&line);
            }
            
            let mut in_idx = "0".to_string();
            let mut stride = "1".to_string();
            for i in (0..in_shape.dims.len()).rev() {
                in_idx = "(IN_IDX) + (dIDX) * (STRIDE)"
                    .replace("IN_IDX", &in_idx)
                    .replace("IDX", &i.to_string())
                    .replace("STRIDE", &stride);
                stride = "(STRIDE) * (DIM)"
                    .replace("STRIDE", &stride)
                    .replace("DIM", &in_shape.dims[i].to_c_expr());
            }

            let mut out_idx = "0".to_string();
            let mut out_stride = "1".to_string();
            for i in (0..permutation.len()).rev() {
                let target_axis = permutation[i];
                out_idx = "(OUT_IDX) + (dIDX) * (STRIDE)"
                    .replace("OUT_IDX", &out_idx)
                    .replace("IDX", &target_axis.to_string())
                    .replace("STRIDE", &out_stride);
                out_stride = "(STRIDE) * (DIM)"
                    .replace("STRIDE", &out_stride)
                    .replace("DIM", &in_shape.dims[target_axis].to_c_expr());
            }

            c.push_str("    VAR[OUT_IDX] = SRC[IN_IDX];\n"
                .replace("VAR", &node_var)
                .replace("OUT_IDX", &out_idx)
                .replace("SRC", &src)
                .replace("IN_IDX", &in_idx).as_str());
            
            for _ in &in_shape.dims {
                c.push_str("    }\n");
            }
        }
    }
}

fn get_input_var(input: &InputConnection) -> String {
    let base = if let Some(in_name) = input.node_id.strip_prefix("inputs.") {
        "in_NAME".replace("NAME", &sanitize_id(in_name))
    } else {
        sanitize_id(&input.node_id)
    };

    if let Ok(idx) = input.src_port.parse::<usize>() {
        if idx > 0 {
            return "(BASE + IDX * (SIZE))"
                .replace("BASE", &base)
                .replace("IDX", &idx.to_string())
                .replace("SIZE", &input.shape.to_c_size_expr());
        }
    }
    base
}