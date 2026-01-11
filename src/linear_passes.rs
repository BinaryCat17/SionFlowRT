use crate::model::{Op, Dimension, TensorShape};
use crate::linear_ir::LinearIR;
use crate::manifest::Mapping;
use std::collections::HashMap;

pub fn run_shape_inference(
    ir: &mut LinearIR, 
    parameters: &HashMap<String, usize>,
    mappings: &[Mapping],
    program_id: &str
) -> anyhow::Result<()> {
    let mut known_shapes: HashMap<String, TensorShape> = HashMap::new();

    for node in &ir.nodes {
        if let Op::Input { name } = &node.op {
            let m_shape = mappings.iter()
                .find(|m| m.program == program_id && (m.tensor == *name || m.tensor == node.id))
                .and_then(|m| m.shape.as_ref());
            
            if let Some(s) = m_shape {
                known_shapes.insert(node.id.clone(), TensorShape { dims: s.clone() });
            }
        }
    }

    let mut changed = true;
    for _ in 0..15 {
        if !changed { break; }
        changed = false;

        for node_idx in 0..ir.nodes.len() {
            let node_id = ir.nodes[node_idx].id.clone();
            let mut input_shapes = Vec::new();
            for src_id in &ir.nodes[node_idx].inputs {
                if let Some(shape) = known_shapes.get(src_id) {
                    input_shapes.push(shape.clone());
                }
            }

            let current_shape = known_shapes.get(&node_id).cloned();
            let mut new_shape = infer_node_shape(&ir.nodes[node_idx].op, &input_shapes, current_shape.as_ref());

            if let Some(ref mut s) = new_shape {
                for dim in &mut s.dims { *dim = dim.eval(parameters); }
            }

            if new_shape.is_some() && new_shape != current_shape {
                known_shapes.insert(node_id, new_shape.unwrap());
                changed = true;
            }
        }
    }

    for node in &mut ir.nodes {
        if let Some(s) = known_shapes.get(&node.id) {
            node.shape = s.clone();
        }
    }

    Ok(())
}

fn infer_node_shape(op: &Op, inputs: &[TensorShape], current: Option<&TensorShape>) -> Option<TensorShape> {
    match op {
        Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Min | Op::Max | Op::Pow | Op::Clamp => {
            unify_and_broadcast(inputs).ok()
        },
        Op::Reshape { new_shape } => {
            if let Some(in_s) = inputs.first() {
                let mut res = Vec::new();
                for d in new_shape {
                    if d.is_ellipsis() {
                        // Поглощаем все измерения входа кроме последнего (если в решейпе есть что-то после ...)
                        let after_count = new_shape.len() - 1 - new_shape.iter().position(|x| x.is_ellipsis()).unwrap();
                        let take = in_s.dims.len().saturating_sub(after_count);
                        for i in 0..take { res.push(in_s.dims[i].clone()); }
                    } else if matches!(d, Dimension::Symbol(s) if s == "_") {
                        // TODO: Умный расчет объема. Пока просто берем 1-к-1.
                        res.push(Dimension::Value(1));
                    } else {
                        res.push(d.clone());
                    }
                }
                Some(TensorShape { dims: res })
            } else {
                Some(TensorShape { dims: new_shape.clone() })
            }
        },
        Op::ReduceSum { axis } => {
            let mut s = inputs.first().cloned()?;
            let axis_idx = if *axis < 0 { (s.dims.len() as isize + *axis) as usize } else { *axis as usize };
            if axis_idx < s.dims.len() { s.dims.remove(axis_idx); }
            if s.dims.is_empty() { s.dims.push(Dimension::Value(1)); }
            Some(s)
        },
        Op::Constant { values } => Some(TensorShape { dims: vec![Dimension::Value(values.len())] }),
        _ => inputs.first().cloned().or_else(|| current.cloned()),
    }
}

fn unify_and_broadcast(shapes: &[TensorShape]) -> anyhow::Result<TensorShape> {
    if shapes.is_empty() { return Err(anyhow::anyhow!("No inputs")); }
    
    let mut result = shapes[0].clone();
    for next in &shapes[1..] {
        result = unify_two(&result, next)?;
    }
    Ok(result)
}

fn unify_two(s1: &TensorShape, s2: &TensorShape) -> anyhow::Result<TensorShape> {
    // 1. Расширяем многоточия
    let e1 = s1.dims.iter().position(|d| d.is_ellipsis());
    let e2 = s2.dims.iter().position(|d| d.is_ellipsis());

    let (mut d1, mut d2) = (s1.dims.clone(), s2.dims.clone());

    if let Some(pos) = e1 {
        let needed = s2.dims.len().saturating_sub(s1.dims.len() - 1);
        let mut patch = Vec::new();
        for i in 0..needed {
            // Пытаемся взять размерность из s2
            patch.push(s2.dims.get(pos + i).cloned().unwrap_or(Dimension::Value(1)));
        }
        d1.splice(pos..pos+1, patch);
    }
    if let Some(pos) = e2 {
        let needed = d1.len().saturating_sub(s2.dims.len() - 1);
        let mut patch = Vec::new();
        for i in 0..needed {
            patch.push(d1.get(pos + i).cloned().unwrap_or(Dimension::Value(1)));
        }
        d2.splice(pos..pos+1, patch);
    }

    // 2. Обычный бродкастинг
    let mut res = Vec::new();
    let (mut i, mut j) = (d1.len() as isize - 1, d2.len() as isize - 1);
    while i >= 0 || j >= 0 {
        let v1 = if i >= 0 { Some(&d1[i as usize]) } else { None };
        let v2 = if j >= 0 { Some(&d2[j as usize]) } else { None };
        match (v1, v2) {
            (Some(a), Some(b)) => {
                if a == b { res.push(a.clone()); }
                else if matches!(a, Dimension::Value(1)) { res.push(b.clone()); }
                else if matches!(b, Dimension::Value(1)) { res.push(a.clone()); }
                else if matches!(a, Dimension::Symbol(s) if s == "_") { res.push(b.clone()); }
                else if matches!(b, Dimension::Symbol(s) if s == "_") { res.push(a.clone()); }
                else { res.push(a.clone()); }
            }
            (Some(a), None) | (None, Some(a)) => res.push(a.clone()),
            _ => {}
        }
        i -= 1; j -= 1;
    }
    res.reverse();
    Ok(TensorShape { dims: res })
}