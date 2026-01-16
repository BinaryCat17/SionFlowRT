pub mod ir;

use crate::core::types::{Shape, DataType, Dim, Port};
use crate::inliner::raw_ir::{RawIR};
use crate::resolver::ir::{ResolvedIR, ResolvedNode, ResolvedEdge};
use crate::core::op::Op;
use petgraph::algo::toposort;
use petgraph::visit::EdgeRef;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use anyhow::{Context, anyhow};

pub fn resolve_module(
    raw: RawIR,
    input_specs: HashMap<String, Port>,
) -> anyhow::Result<ResolvedIR> {
    let mut resolved_graph = petgraph::graph::DiGraph::new();
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new(); 
    let mut shapes: HashMap<NodeIndex, Shape> = HashMap::new();

    let order = toposort(&raw.graph, None)
        .map_err(|_| anyhow!("Cycle detected in module graph"))?;

    for old_idx in order {
        let raw_node = &raw.graph[old_idx];
        let op = raw_node.op.clone();

        let mut input_shapes = Vec::new();
        let mut incoming_edges: Vec<_> = raw.graph.edges_directed(old_idx, petgraph::Direction::Incoming).collect();
        incoming_edges.sort_by(|a, b| a.weight().dst_port.cmp(&b.weight().dst_port));
        
        for edge in incoming_edges {
            let src_old_idx = edge.source();
            let src_new_idx = node_map.get(&src_old_idx)
                .ok_or_else(|| anyhow!("Source node not found in map for edge to '{}'", raw_node.id))?;
            let shape = shapes.get(src_new_idx)
                .ok_or_else(|| anyhow!("Shape not found for source node of '{}'", raw_node.id))?;
            input_shapes.push(shape.clone());
        }

        let node_shape = infer_shape(&op, &input_shapes, &input_specs)
            .with_context(|| format!("Shape inference failed for node '{}' ({:?})", raw_node.id, op))?;
        let node_dtype = DataType::F32;

        let new_idx = resolved_graph.add_node(ResolvedNode {
            id: raw_node.id.clone(),
            op,
            shape: node_shape.clone(),
            dtype: node_dtype,
        });

        node_map.insert(old_idx, new_idx);
        shapes.insert(new_idx, node_shape);
    }

    for edge in raw.graph.edge_references() {
        let src_new = node_map.get(&edge.source()).context("Edge source mapping missing")?;
        let dst_new = node_map.get(&edge.target()).context("Edge target mapping missing")?;
        
        resolved_graph.add_edge(*src_new, *dst_new, ResolvedEdge {
            src_port: edge.weight().src_port.clone(),
            dst_port: edge.weight().dst_port.clone(),
        });
    }

    // Заполняем выходы
    let mut outputs = Vec::new();
    let mut out_nodes: Vec<_> = resolved_graph.node_indices()
        .filter(|&idx| matches!(resolved_graph[idx].op, Op::Output { .. }))
        .collect();
    
    // Сортируем по имени выходного порта для детерминизма
    out_nodes.sort_by(|&a, &b| {
        let name_a = if let Op::Output { name } = &resolved_graph[a].op { name } else { "" };
        let name_b = if let Op::Output { name } = &resolved_graph[b].op { name } else { "" };
        name_a.cmp(name_b)
    });

    for idx in out_nodes {
        let node = &resolved_graph[idx];
        if let Op::Output { name } = &node.op {
            let mut incoming = resolved_graph.edges_directed(idx, petgraph::Direction::Incoming);
            if let Some(edge) = incoming.next() {
                let src_node = &resolved_graph[edge.source()];
                outputs.push(Port {
                    name: name.clone(),
                    shape: src_node.shape.clone(),
                    dtype: src_node.dtype,
                });
            }
        }
    }

    Ok(ResolvedIR {
        graph: resolved_graph,
        inputs: raw.inputs.iter().map(|i| {
            input_specs.get(&i.name).cloned().unwrap_or(Port { 
                name: i.name.clone(), 
                shape: Shape { dims: vec![] }, 
                dtype: DataType::F32 
            })
        }).collect(),
        outputs,
    })
}

fn infer_shape(
    op: &Op,
    inputs: &[Shape],
    input_specs: &HashMap<String, Port>,
) -> anyhow::Result<Shape> {
    match op {
        Op::Input { name } => {
            let spec = input_specs.get(name)
                .ok_or_else(|| anyhow!("Missing input spec for '{}' in program interface", name))?;
            Ok(spec.shape.clone())
        }
        Op::Constant { values } => {
            Ok(Shape { dims: vec![Dim::Static(values.len())] })
        }
        Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Min | Op::Max | Op::Pow => {
            if inputs.len() == 2 {
                broadcast_shapes(&inputs[0], &inputs[1])
            } else if inputs.len() == 1 {
                Ok(inputs[0].clone())
            } else {
                Err(anyhow!("Binary op {:?} expects 1 or 2 inputs, found {}", op, inputs.len()))
            }
        }
        Op::Sin | Op::Abs | Op::Sqrt | Op::Square | Op::Exp | Op::Log | Op::Output { .. } => {
            if inputs.is_empty() {
                return Err(anyhow!("Unary/Output op {:?} requires at least 1 input", op));
            }
            Ok(inputs[0].clone())
        }
        Op::Reshape { new_shape } => {
            Ok(Shape { dims: new_shape.clone() })
        }
        Op::Transpose { permutation } => {
            if inputs.is_empty() {
                return Err(anyhow!("Transpose requires 1 input"));
            }
            let input_dims = &inputs[0].dims;
            if input_dims.len() != permutation.len() {
                return Err(anyhow!("Transpose permutation length {} doesn't match input rank {}", permutation.len(), input_dims.len()));
            }
            let mut new_dims = Vec::with_capacity(permutation.len());
            for &axis in permutation {
                if axis >= input_dims.len() {
                    return Err(anyhow!("Transpose axis {} out of bounds for rank {}", axis, input_dims.len()));
                }
                new_dims.push(input_dims[axis].clone());
            }
            Ok(Shape { dims: new_dims })
        }
        Op::ReduceSum { axis } => {
            if inputs.is_empty() { return Err(anyhow!("ReduceSum requires 1 input")); }
            let mut dims = inputs[0].dims.clone();
            if *axis >= dims.len() {
                return Err(anyhow!("ReduceSum axis {} out of bounds for rank {}", axis, dims.len()));
            }
            dims.remove(*axis);
            Ok(Shape { dims })
        }
        Op::Split { axis, parts } => {
            if inputs.is_empty() { return Err(anyhow!("Split requires 1 input")); }
            let mut dims = inputs[0].dims.clone();
            if *axis >= dims.len() {
                return Err(anyhow!("Split axis {} out of bounds for rank {}", axis, dims.len()));
            }
            match &dims[*axis] {
                Dim::Static(val) => {
                    if val % parts != 0 {
                        return Err(anyhow!("Dimension size {} at axis {} is not divisible by {} parts", val, axis, parts));
                    }
                    dims[*axis] = Dim::Static(val / parts);
                }
                Dim::Variable(name) => {
                    // For variables, we might need a synthetic variable for the part size
                    // but for now let's just keep it as is or handle via dynamic expressions
                    dims[*axis] = Dim::Variable(format!("({} / {})", name, parts));
                }
            }
            Ok(Shape { dims })
        }
        Op::MatMul => {
            if inputs.len() != 2 {
                return Err(anyhow!("MatMul requires exactly 2 inputs, found {}", inputs.len()));
            }
            let a = &inputs[0].dims;
            let b = &inputs[1].dims;
            
            if a.len() < 2 || b.len() < 2 {
                return Err(anyhow!("MatMul requires inputs with at least 2 dimensions, found shapes {:?} and {:?}", a, b));
            }

            let m = &a[a.len() - 2];
            let k_a = &a[a.len() - 1];
            let k_b = &b[b.len() - 2];
            let n = &b[b.len() - 1];

            match (k_a, k_b) {
                (Dim::Static(v1), Dim::Static(v2)) if v1 != v2 => {
                    return Err(anyhow!("Incompatible dimensions for MatMul: inner dims {} and {} do not match", v1, v2));
                }
                _ => {}
            }

            let batch_a = Shape { dims: a[..a.len()-2].to_vec() };
            let batch_b = Shape { dims: b[..b.len()-2].to_vec() };
            let mut result_dims = broadcast_shapes(&batch_a, &batch_b)?.dims;
            
            result_dims.push(m.clone());
            result_dims.push(n.clone());
            
            Ok(Shape { dims: result_dims })
        }
    }
}

fn broadcast_shapes(a: &Shape, b: &Shape) -> anyhow::Result<Shape> {
    let mut out_dims = Vec::new();
    let len_a = a.dims.len();
    let len_b = b.dims.len();
    let max_len = std::cmp::max(len_a, len_b);

    for i in 0..max_len {
        let dim_a = if i < (max_len - len_a) { &Dim::Static(1) } else { &a.dims[i - (max_len - len_a)] };
        let dim_b = if i < (max_len - len_b) { &Dim::Static(1) } else { &b.dims[i - (max_len - len_b)] };

        match (dim_a, dim_b) {
            (Dim::Static(va), Dim::Static(vb)) => {
                if *va == *vb { out_dims.push(Dim::Static(*va)); }
                else if *va == 1 { out_dims.push(Dim::Static(*vb)); }
                else if *vb == 1 { out_dims.push(Dim::Static(*va)); }
                else { return Err(anyhow!("Shape mismatch for broadcast: {} and {}", va, vb)); }
            }
            (Dim::Variable(sa), Dim::Variable(sb)) if sa == sb => out_dims.push(Dim::Variable(sa.clone())),
            (Dim::Variable(s), _) | (_, Dim::Variable(s)) => out_dims.push(Dim::Variable(s.clone())),
        }
    }
    Ok(Shape { dims: out_dims })
}