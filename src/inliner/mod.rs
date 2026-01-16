pub mod json;
pub mod raw_ir;
pub mod paths;

use crate::inliner::json::{JsonGraph};
use crate::inliner::raw_ir::{RawIR, RawNode, RawEdge};
use crate::inliner::paths::resolve_subgraph_path;
use crate::manifest::Manifest;
use crate::core::op::Op;
use std::collections::HashMap;
use std::path::{Path};
use petgraph::graph::NodeIndex;

#[derive(Default)]
struct InterfaceMapping {
    inputs: HashMap<String, Vec<(NodeIndex, String)>>,
    outputs: HashMap<String, (NodeIndex, String)>,
}

pub fn load_and_inline(
    root_graph: JsonGraph,
    base_path: &Path,
    manifest: &Manifest,
    synthetic_vars: &mut HashMap<String, String>,
) -> anyhow::Result<RawIR> {
    let mut raw_ir = RawIR::new();
    let mapping = inline_recursive_graph(root_graph, base_path, "", &mut raw_ir, manifest, synthetic_vars)?;

    // Bridge top-level inputs to the graph
    for (port_name, consumers) in mapping.inputs {
        let input_node = raw_ir.graph.add_node(RawNode {
            id: format!("inputs.{}", port_name),
            op: Op::Input { name: port_name.clone() },
        });
        for (dst_node, dst_port) in consumers {
            raw_ir.graph.add_edge(input_node, dst_node, RawEdge {
                src_port: "output".to_string(),
                dst_port,
            });
        }
    }

    // Bridge top-level outputs to the graph
    for (port_name, (src_node, src_port)) in mapping.outputs {
        let output_node = raw_ir.graph.add_node(RawNode {
            id: format!("outputs.{}", port_name),
            op: Op::Output { name: port_name.clone() },
        });
        raw_ir.graph.add_edge(src_node, output_node, RawEdge {
            src_port,
            dst_port: "input".to_string(),
        });
    }

    Ok(raw_ir)
}

fn inline_recursive(
    path: &Path,
    prefix: &str,
    raw_ir: &mut RawIR,
    manifest: &Manifest,
    synthetic_vars: &mut HashMap<String, String>,
) -> anyhow::Result<InterfaceMapping> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", path.display(), e))?;
    let graph_def = JsonGraph::from_json(&content)?;
    inline_recursive_graph(graph_def, path, prefix, raw_ir, manifest, synthetic_vars)
}

fn inline_recursive_graph(
    graph_def: JsonGraph,
    path: &Path,
    prefix: &str,
    raw_ir: &mut RawIR,
    manifest: &Manifest,
    synthetic_vars: &mut HashMap<String, String>,
) -> anyhow::Result<InterfaceMapping> {
    if prefix.is_empty() {
        raw_ir.inputs = graph_def.inputs.clone();
        raw_ir.outputs = graph_def.outputs.clone();
    }

    let mut sub_mappings: HashMap<String, InterfaceMapping> = HashMap::new();
    let mut primitive_nodes: HashMap<String, NodeIndex> = HashMap::new();

    for node_def in &graph_def.nodes {
        let full_id = if prefix.is_empty() { node_def.id.clone() } else { format!("{}/{}", prefix, node_def.id) };

        if let Some(sub_path_raw) = &node_def.subgraph {
            let mut actual_path_str = sub_path_raw.clone();
            if let Some(imports) = &graph_def.imports {
                for (key, val) in imports {
                    if sub_path_raw.starts_with(key) {
                        actual_path_str = sub_path_raw.replace(key, val);
                        break;
                    }
                }
            }
            
            let sub_full_path = resolve_subgraph_path(path, &actual_path_str);
            let mapping = inline_recursive(&sub_full_path, &full_id, raw_ir, manifest, synthetic_vars)?;
            sub_mappings.insert(node_def.id.clone(), mapping);
        } else if let Some(op_val) = &node_def.op {
            let mut normalized_json = op_val.clone();
            normalize_op_json(&mut normalized_json, manifest, synthetic_vars);
            
            let op = Op::from_json_value(&normalized_json)?;
            let node_idx = raw_ir.graph.add_node(RawNode {
                id: full_id.clone(),
                op,
            });
            primitive_nodes.insert(node_def.id.clone(), node_idx);
        }
    }

    let mut current_mapping = InterfaceMapping::default();

    for (src_addr, dst_addr) in &graph_def.links {
        let sources = resolve_source(src_addr, &primitive_nodes, &sub_mappings)?;
        let destinations = resolve_destination(dst_addr, &primitive_nodes, &sub_mappings)?;

        for (src_node, src_port) in &sources {
            for (dst_node, dst_port) in &destinations {
                raw_ir.graph.add_edge(*src_node, *dst_node, RawEdge {
                    src_port: src_port.clone(),
                    dst_port: dst_port.clone(),
                });
            }
        }

        update_interface_mapping(src_addr, dst_addr, &sources, &destinations, &mut current_mapping);
    }

    Ok(current_mapping)
}

fn normalize_op_json(
    value: &mut serde_json::Value, 
    manifest: &Manifest,
    synthetic_vars: &mut HashMap<String, String>
) {
    if value.is_object() {
        if let Ok(op) = serde_json::from_value::<crate::inliner::json::JsonDimOp>(value.clone()) {
            let resolved_dim = crate::analyzer::process_json_dim(
                &crate::inliner::json::JsonDim::Op(op), 
                synthetic_vars, 
                manifest
            );
            *value = match resolved_dim {
                crate::core::types::Dim::Variable(var_name) => serde_json::Value::String(var_name),
                crate::core::types::Dim::Static(val) => serde_json::Value::Number(val.into()),
            };
            return;
        }
    }

    if let Some(obj) = value.as_object_mut() {
        for v in obj.values_mut() {
            normalize_op_json(v, manifest, synthetic_vars);
        }
    } else if let Some(arr) = value.as_array_mut() {
        for v in arr {
            normalize_op_json(v, manifest, synthetic_vars);
        }
    }
}

fn resolve_source(
    addr: &str,
    nodes: &HashMap<String, NodeIndex>,
    subgraphs: &HashMap<String, InterfaceMapping>,
) -> anyhow::Result<Vec<(NodeIndex, String)>> {
    if addr.starts_with("inputs.") {
        return Ok(vec![]);
    }
    let (node_id, port) = addr.split_once('.').ok_or_else(|| anyhow::anyhow!("Invalid src: {}", addr))?;
    
    if let Some(&idx) = nodes.get(node_id) {
        return Ok(vec![(idx, port.to_string())]);
    }
    if let Some(mapping) = subgraphs.get(node_id) {
        if let Some(src) = mapping.outputs.get(port) {
            return Ok(vec![src.clone()]);
        }
    }
    Err(anyhow::anyhow!("Source not found: {}", addr))
}

fn resolve_destination(
    addr: &str,
    nodes: &HashMap<String, NodeIndex>,
    subgraphs: &HashMap<String, InterfaceMapping>,
) -> anyhow::Result<Vec<(NodeIndex, String)>> {
    if addr.starts_with("outputs.") {
        return Ok(vec![]);
    }
    let (node_id, port) = addr.split_once('.').ok_or_else(|| anyhow::anyhow!("Invalid dst: {}", addr))?;

    if let Some(&idx) = nodes.get(node_id) {
        return Ok(vec![(idx, port.to_string())]);
    }
    if let Some(mapping) = subgraphs.get(node_id) {
        if let Some(consumers) = mapping.inputs.get(port) {
            return Ok(consumers.clone());
        }
    }
    Err(anyhow::anyhow!("Destination not found: {}", addr))
}

fn update_interface_mapping(
    src_addr: &str,
    dst_addr: &str,
    sources: &[(NodeIndex, String)],
    destinations: &[(NodeIndex, String)],
    mapping: &mut InterfaceMapping,
) {
    if let Some(in_name) = src_addr.strip_prefix("inputs.") {
        mapping.inputs.entry(in_name.to_string()).or_default().extend(destinations.iter().cloned());
    }
    if let Some(out_name) = dst_addr.strip_prefix("outputs.") {
        if let Some(src) = sources.first() {
            mapping.outputs.insert(out_name.to_string(), src.clone());
        }
    }
}
