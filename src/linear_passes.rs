use crate::model::{Op, Dimension, TensorShape};
use crate::linear_ir::LinearIR;
use crate::shape_engine::ShapeEngine;
use std::collections::HashMap;

pub fn run_shape_inference(
    ir: &mut LinearIR, 
    parameters: &HashMap<String, usize>,
) -> anyhow::Result<()> {
    let mut known_shapes: HashMap<String, TensorShape> = HashMap::new();

    // 1. Предварительно заполняем уже известные формы (например, от входов, которые проставил линкер)
    for node in &ir.nodes {
        if !node.shape.dims.is_empty() {
            known_shapes.insert(node.id.clone(), node.shape.clone());
        }
    }

    // 2. Линейный проход (один раз, так как узлы в топологическом порядке)
    for node in &mut ir.nodes {
        let mut input_shapes = Vec::new();
        for src_id in &node.inputs {
            if let Some(shape) = known_shapes.get(src_id) {
                input_shapes.push(shape.clone());
            }
        }

        let current_shape = known_shapes.get(&node.id).cloned();
        let mut new_shape = infer_node_shape(&node.op, &input_shapes, current_shape.as_ref());

        if let Some(ref mut s) = new_shape {
            for dim in &mut s.dims { 
                *dim = dim.eval(parameters); 
                *dim = ShapeEngine::simplify(dim.clone());
            }
            
            // Записываем результат обратно в ноду и в кэш для следующих узлов
            node.shape = s.clone();
            known_shapes.insert(node.id.clone(), s.clone());
        }
    }

    Ok(())
}

fn infer_node_shape(op: &Op, inputs: &[TensorShape], current: Option<&TensorShape>) -> Option<TensorShape> {
    match op {
        Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Min | Op::Max | Op::Pow => {
            if inputs.is_empty() { return None; }
            let mut res = inputs[0].clone();
            for next in &inputs[1..] {
                res = ShapeEngine::unify(&res, next).ok()?;
            }
            Some(res)
        },
        Op::Reshape { new_shape } => {
            if let Some(in_s) = inputs.first() {
                let mut res_dims = Vec::new();
                let in_volume = ShapeEngine::volume(in_s);
                
                let mut wildcard_idx = None;
                let mut known_volume = Dimension::Value(1);

                let expanded = expand_reshape_dims(&new_shape, &in_s.dims);

                for (i, d) in expanded.iter().enumerate() {
                    if d.is_wildcard() {
                        if wildcard_idx.is_some() { return None; } 
                        wildcard_idx = Some(i);
                        res_dims.push(Dimension::Wildcard);
                    } else {
                        known_volume = Dimension::Mul(Box::new(known_volume), Box::new(d.clone()));
                        res_dims.push(d.clone());
                    }
                }

                if let Some(idx) = wildcard_idx {
                    let calculated = Dimension::Div(Box::new(in_volume), Box::new(known_volume));
                    res_dims[idx] = ShapeEngine::simplify(calculated);
                }

                Some(TensorShape { dims: res_dims })
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

fn expand_reshape_dims(dims: &[Dimension], in_dims: &[Dimension]) -> Vec<Dimension> {
    if let Some(pos) = dims.iter().position(|d| d.is_ellipsis()) {
        let mut res = Vec::new();
        for i in 0..pos { res.push(dims[i].clone()); }
        
        let after_count = dims.len() - 1 - pos;
        let take = in_dims.len().saturating_sub(after_count).saturating_sub(pos);
        
        for i in 0..take { 
            res.push(in_dims[pos + i].clone()); 
        }
        
        for i in (pos + 1)..dims.len() { res.push(dims[i].clone()); }
        res
    } else {
        dims.to_vec()
    }
}