use crate::linearizer::ir::{LinearIR, LinearNode, InputConnection};
use crate::core::op::Op;
use crate::core::utils::sanitize_id;

pub fn generate_module_source(module_id: &str, ir: &LinearIR) -> String {
    let mut c = String::new();
    
    // Header includes
    c.push_str("#include \"MOD_ID.h\"\n".replace("MOD_ID", module_id).as_str());
    c.push_str("#include <math.h>\n");
    c.push_str("#ifdef _OPENMP\n#include <omp.h>\n#endif\n\n");

    let args = get_function_args(ir);
    let mut func_sig = "void FUNC_NAME_func(ARGS) { 
".to_string();
    func_sig = func_sig.replace("FUNC_NAME", module_id);
    func_sig = func_sig.replace("ARGS", &args.join(", "));
    c.push_str(&func_sig);

    // Workspace pointers casting
    for node in &ir.nodes {
        if matches!(node.op, Op::Input { .. } | Op::Output { .. }) { continue; }
        let c_type = node.dtype.to_c_type();
        let id = sanitize_id(&node.id);
        let mut cast = "    TYPE* restrict ID = (TYPE*)workspace[OFFSET];\n".to_string();
        cast = cast.replace("TYPE", c_type);
        cast = cast.replace("ID", &id);
        cast = cast.replace("OFFSET", &node.offset.to_string());
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
    let guard = "MOD_ID_H".replace("MOD_ID", &module_id.to_uppercase());
    
    let mut header = "#ifndef GUARD\n#define GUARD\n\n#include <stdint.h>\n\n".to_string();
    header = header.replace("GUARD", &guard);
    c.push_str(&header);

    let args = get_function_args(ir);
    let mut decl = "void FUNC_NAME_func(ARGS);\n\n".to_string();
    decl = decl.replace("FUNC_NAME", module_id);
    decl = decl.replace("ARGS", &args.join(", "));
    c.push_str(&decl);

    c.push_str("#endif\n");
    c
}

fn get_function_args(ir: &LinearIR) -> Vec<String> {
    let mut args = Vec::new();
    args.push("void** workspace".to_string());

    for input in &ir.inputs {
        let mut arg = "const TYPE* restrict in_NAME".to_string();
        arg = arg.replace("TYPE", input.dtype.to_c_type());
        arg = arg.replace("NAME", &sanitize_id(&input.name));
        args.push(arg);
    }
    
    for port in &ir.outputs {
        let mut arg = "TYPE* restrict out_NAME".to_string();
        arg = arg.replace("TYPE", port.dtype.to_c_type());
        arg = arg.replace("NAME", &sanitize_id(&port.name));
        args.push(arg);
    }
    args
}

fn emit_node_code(c: &mut String, node: &LinearNode, _ir: &LinearIR) {
    let node_var = sanitize_id(&node.id);
    let size_expr = node.shape.to_c_size_expr();

    match &node.op {
        Op::Input { name } => {
            c.push_str("    // Input NAME handled via args\n".replace("NAME", name).as_str());
        }
        Op::Constant { values } => {
            for (i, v) in values.iter().enumerate() {
                let mut line = "    VAR[IDX] = VALf;\n".to_string();
                line = line.replace("VAR", &node_var);
                line = line.replace("IDX", &i.to_string());
                line = line.replace("VAL", &v.to_string());
                c.push_str(&line);
            }
        }
        Op::Output { name } => {
            let src = get_input_var(&node.inputs[0]);
            let mut line = "    #pragma omp parallel for simd\n    for (int i = 0; i < SIZE; i++) { out_NAME[i] = SRC[i]; }\n".to_string();
            line = line.replace("SIZE", &size_expr);
            line = line.replace("NAME", &sanitize_id(name));
            line = line.replace("SRC", &src);
            c.push_str(&line);
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
                let mut line = "    for (int i = 0; i < SIZE; i++) { VAR[i] = LEFT[i] SYM RIGHT[i]; }\n".to_string();
                line = line.replace("SIZE", &size_expr);
                line = line.replace("VAR", &node_var);
                line = line.replace("LEFT", &left);
                line = line.replace("SYM", op_sym);
                line = line.replace("RIGHT", &right);
                c.push_str(&line);
            } else {
                let func = match node.op {
                    Op::Min => "fminf",
                    Op::Max => "fmaxf",
                    Op::Pow => "powf",
                    _ => unreachable!(),
                };
                let mut line = "    for (int i = 0; i < SIZE; i++) { VAR[i] = FUNC (LEFT[i], RIGHT[i]); }\n".to_string();
                line = line.replace("SIZE", &size_expr);
                line = line.replace("VAR", &node_var);
                line = line.replace("FUNC", func);
                line = line.replace("LEFT", &left);
                line = line.replace("RIGHT", &right);
                c.push_str(&line);
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
                let mut line = "    for (int i = 0; i < SIZE; i++) { VAR[i] = SRC[i] * SRC[i]; }\n".to_string();
                line = line.replace("SIZE", &size_expr);
                line = line.replace("VAR", &node_var);
                line = line.replace("SRC", &src);
                c.push_str(&line);
            } else {
                let mut line = "    for (int i = 0; i < SIZE; i++) { VAR[i] = FUNC (SRC[i]); }\n".to_string();
                line = line.replace("SIZE", &size_expr);
                line = line.replace("VAR", &node_var);
                line = line.replace("FUNC", func);
                line = line.replace("SRC", &src);
                c.push_str(&line);
            }
        }
        Op::Reshape { .. } => {
            let src = get_input_var(&node.inputs[0]);
            let mut line = "    #pragma omp parallel for simd\n    for (int i = 0; i < SIZE; i++) { VAR[i] = SRC[i]; }\n".to_string();
            line = line.replace("SIZE", &size_expr);
            line = line.replace("VAR", &node_var);
            line = line.replace("SRC", &src);
            c.push_str(&line);
        }
        Op::ReduceSum { axis } => {
            let src = get_input_var(&node.inputs[0]);
            let input_shape = &node.inputs[0].shape;
            
            let mut init = "    for (int i = 0; i < SIZE; i++) { VAR[i] = 0.0f; }\n".to_string();
            init = init.replace("SIZE", &size_expr).replace("VAR", &node_var);
            c.push_str(&init);
            
            let reduce_dim = input_shape.dims[*axis].to_c_expr();
            let outer_size_raw = input_shape.dims[0..*axis].iter().map(|d| d.to_c_expr()).collect::<Vec<_>>().join(" * ");
            let inner_size_raw = input_shape.dims[*axis+1..].iter().map(|d| d.to_c_expr()).collect::<Vec<_>>().join(" * ");
            
            let outer_size = if outer_size_raw.is_empty() { "1".to_string() } else { outer_size_raw };
            let inner_size = if inner_size_raw.is_empty() { "1".to_string() } else { inner_size_raw };

            let mut loops = "\n    for (int out = 0; out < OUTER * INNER; out++) {\n        int o = out / INNER;\n        int i = out % INNER;\n        for (int r = 0; r < REDUCE; r++) {\n            VAR[o * INNER + i] += SRC[o * REDUCE * INNER + r * INNER + i];\n        }\n    }\n".to_string();
            loops = loops.replace("OUTER", &outer_size);
            loops = loops.replace("INNER", &inner_size);
            loops = loops.replace("REDUCE", &reduce_dim);
            loops = loops.replace("VAR", &node_var);
            loops = loops.replace("SRC", &src);
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
            
            let mut init = "    for (int i = 0; i < SIZE; i++) { VAR[i] = 0.0f; }\n".to_string();
            init = init.replace("SIZE", &size_expr).replace("VAR", &node_var);
            c.push_str(&init);

            let mut loops = "\n    int batch_size = (SIZE) / ((M) * (N));\n    for (int b = 0; b < batch_size; b++) {\n        for (int i = 0; i < M; i++) {\n            for (int j = 0; j < N; j++) {\n                for (int l = 0; l < K; l++) {\n                    VAR[b * M * N + i * N + j] += LEFT[b * M * K + i * K + l] * RIGHT[b * K * N + l * N + j];\n                }\n            }\n        }\n    }\n".to_string();
            loops = loops.replace("SIZE", &size_expr);
            loops = loops.replace("M", &m);
            loops = loops.replace("N", &n);
            loops = loops.replace("K", &k);
            loops = loops.replace("VAR", &node_var);
            loops = loops.replace("LEFT", &left);
            loops = loops.replace("RIGHT", &right);
            c.push_str(&loops);
        }
        Op::Split { parts, .. } => {
            let src = get_input_var(&node.inputs[0]);
            let mut line = "    #pragma omp parallel for simd\n    for (int i = 0; i < SIZE * PARTS; i++) { VAR[i] = SRC[i]; }\n".to_string();
            line = line.replace("SIZE", &size_expr);
            line = line.replace("PARTS", &parts.to_string());
            line = line.replace("VAR", &node_var);
            line = line.replace("SRC", &src);
            c.push_str(&line);
        }
        Op::Transpose { permutation } => {
            let src = get_input_var(&node.inputs[0]);
            let in_shape = &node.inputs[0].shape;
            
            for (i, _) in in_shape.dims.iter().enumerate() {
                let mut line = "    for (int dIDX = 0; dIDX < DIM; dIDX++) { \n".to_string();
                line = line.replace("IDX", &i.to_string());
                line = line.replace("DIM", &in_shape.dims[i].to_c_expr());
                c.push_str(&line);
            }
            
            let mut in_idx = "0".to_string();
            let mut stride = "1".to_string();
            for i in (0..in_shape.dims.len()).rev() {
                let mut term = "((IN_IDX) + (dIDX) * (STRIDE))".to_string();
                term = term.replace("IN_IDX", &in_idx).replace("IDX", &i.to_string()).replace("STRIDE", &stride);
                in_idx = term;
                
                let mut next_stride = "((STRIDE) * (DIM))".to_string();
                next_stride = next_stride.replace("STRIDE", &stride).replace("DIM", &in_shape.dims[i].to_c_expr());
                stride = next_stride;
            }

            let mut out_idx = "0".to_string();
            let mut out_stride = "1".to_string();
            for i in (0..permutation.len()).rev() {
                let target_axis = permutation[i];
                let mut term = "((OUT_IDX) + (dIDX) * (STRIDE))".to_string();
                term = term.replace("OUT_IDX", &out_idx).replace("IDX", &target_axis.to_string()).replace("STRIDE", &out_stride);
                out_idx = term;

                let mut next_stride = "((STRIDE) * (DIM))".to_string();
                next_stride = next_stride.replace("STRIDE", &out_stride).replace("DIM", &in_shape.dims[target_axis].to_c_expr());
                out_stride = next_stride;
            }

            let mut copy_line = "    VAR[OUT_IDX] = SRC[IN_IDX];\n".to_string();
            copy_line = copy_line.replace("VAR", &node_var);
            copy_line = copy_line.replace("OUT_IDX", &out_idx);
            copy_line = copy_line.replace("SRC", &src);
            copy_line = copy_line.replace("IN_IDX", &in_idx);
            c.push_str(&copy_line);
            
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
            let mut term = "(BASE + IDX * (SIZE))".to_string();
            term = term.replace("BASE", &base);
            term = term.replace("IDX", &idx.to_string());
            term = term.replace("SIZE", &input.shape.to_c_size_expr());
            return term;
        }
    }
    base
}
