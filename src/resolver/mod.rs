pub mod ir;

use crate::core::types::{Shape, DataType, Dim, Port};
use crate::inliner::raw_ir::{RawIR};
use crate::resolver::ir::{ResolvedIR, ResolvedNode, ResolvedEdge};
use crate::core::op::Op;
use petgraph::algo::toposort;
use petgraph::visit::EdgeRef;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

pub fn resolve_module(
    raw: RawIR,
    input_specs: HashMap<String, Port>,
) -> anyhow::Result<ResolvedIR> {
    let mut resolved_graph = petgraph::graph::DiGraph::new();
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new(); 
    let mut shapes: HashMap<NodeIndex, Shape> = HashMap::new();

    let order = toposort(&raw.graph, None)
        .map_err(|_| anyhow::anyhow!("Cycle detected in module graph"))?;

    for old_idx in order {
        let raw_node = &raw.graph[old_idx];
        
        let op_name = if let Some(s) = raw_node.op.as_str() {
            s
        } else {
            raw_node.op.as_object()
                .and_then(|o| o.keys().next())
                .map(|k| k.as_str())
                .unwrap_or("Unknown")
        };

        let op_params = if raw_node.op.is_string() { 
            serde_json::json!({}) 
        } else { 
            raw_node.op.as_object()
                .and_then(|o| o.values().next().cloned())
                .unwrap_or(serde_json::json!({}))
        };

        let op = Op::from_json(op_name, op_params)?;

        let mut input_shapes = Vec::new();
        let mut incoming_edges: Vec<_> = raw.graph.edges_directed(old_idx, petgraph::Direction::Incoming).collect();
        incoming_edges.sort_by(|a, b| a.weight().dst_port.cmp(&b.weight().dst_port));
        
        for edge in incoming_edges {
            let src_new_idx = node_map[&edge.source()];
            input_shapes.push(shapes[&src_new_idx].clone());
        }

        let node_shape = infer_shape(&op, &input_shapes, &input_specs)?;
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

    let mut outputs = HashMap::new();
    for edge in raw.graph.edge_references() {
        resolved_graph.add_edge(node_map[&edge.source()], node_map[&edge.target()], ResolvedEdge {
            src_port: edge.weight().src_port.clone(),
            dst_port: edge.weight().dst_port.clone(),
        });
    }

    // Заполняем выходы
    for idx in resolved_graph.node_indices() {
        let node = &resolved_graph[idx];
        if let Op::Output { name } = &node.op {
            let mut incoming = resolved_graph.edges_directed(idx, petgraph::Direction::Incoming);
            if let Some(edge) = incoming.next() {
                let src_node = &resolved_graph[edge.source()];
                outputs.insert(name.clone(), src_node.id.clone());
            }
        }
    }

    Ok(ResolvedIR {
        graph: resolved_graph,
        inputs: raw.inputs.iter().map(|i| input_specs.get(&i.name).cloned().unwrap_or(Port { 
            name: i.name.clone(), 
            shape: Shape { dims: vec![] }, 
            dtype: DataType::F32 
        })).collect(),
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
                .ok_or_else(|| anyhow::anyhow!("Missing input spec for {}", name))?;
            Ok(spec.shape.clone())
        }
        Op::Constant { values } => {
            Ok(Shape { dims: vec![Dim::Static(values.len())] })
        }
        Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Min | Op::Max | Op::Pow => {
            if inputs.len() == 2 {
                broadcast_shapes(&inputs[0], &inputs[1])
            } else if !inputs.is_empty() {
                Ok(inputs[0].clone())
            } else {
                Ok(Shape { dims: vec![] })
            }
        }
        Op::Reshape { new_shape } => {
            Ok(Shape { dims: new_shape.clone() })
        }
        Op::ReduceSum { axis } => {
            if inputs.is_empty() { return Ok(Shape { dims: vec![] }); }
            let mut dims = inputs[0].dims.clone();
            if *axis < dims.len() {
                dims.remove(*axis);
            }
            Ok(Shape { dims })
        }
        _ => Ok(inputs.get(0).cloned().unwrap_or(Shape { dims: vec![] })),
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
                else { return Err(anyhow::anyhow!("Shape mismatch for broadcast: {} and {}", va, vb)); }
            }
            (Dim::Variable(sa), Dim::Variable(sb)) if sa == sb => out_dims.push(Dim::Variable(sa.clone())),
            (Dim::Variable(s), _) | (_, Dim::Variable(s)) => out_dims.push(Dim::Variable(s.clone())),
        }
    }
    Ok(Shape { dims: out_dims })
}