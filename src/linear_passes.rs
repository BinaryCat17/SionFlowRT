use crate::model::{Op, TensorShape, Dimension};
use crate::linear_ir::LinearIR;
use crate::shape_engine::ShapeEngine;

pub fn run_loop_fusion(ir: &mut LinearIR) -> anyhow::Result<()> {
    if ir.nodes.is_empty() { return Ok(()); }

    let mut new_groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group: Vec<usize> = Vec::new();

    for i in 0..ir.nodes.len() {
        let node = &ir.nodes[i];
        let is_fusible = !matches!(node.op, Op::Input { .. } | Op::Constant { .. });
        
        if is_fusible {
            if current_group.is_empty() {
                current_group.push(i);
            } else {
                let prev_node_idx = *current_group.last().unwrap();
                let prev_node = &ir.nodes[prev_node_idx];
                
                if node.shape == prev_node.shape && !node.shape.dims.is_empty() {
                    current_group.push(i);
                } else {
                    new_groups.push(current_group);
                    current_group = vec!(i);
                }
            }
        } else {
            if !current_group.is_empty() {
                new_groups.push(current_group);
                current_group = Vec::new();
            }
            new_groups.push(vec![i]);
        }
    }

    if !current_group.is_empty() {
        new_groups.push(current_group);
    }

    ir.groups = new_groups;
    Ok(())
}

pub fn infer_node_shape_generic(op: &Op, inputs: &[TensorShape], current: Option<&TensorShape>) -> Option<TensorShape> {
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
            let s = inputs.first().cloned()?;
            let mut dims = s.dims.clone();
            let rank = dims.len();
            if rank == 0 { return Some(TensorShape { dims: vec![Dimension::Value(1)] }); }
            
            let axis_idx = if *axis < 0 { 
                (rank as isize + *axis) as usize 
            } else { 
                *axis as usize 
            };
            
            if axis_idx < rank { 
                dims.remove(axis_idx); 
            }
            if dims.is_empty() { 
                dims.push(Dimension::Value(1)); 
            }
            Some(TensorShape { dims })
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
