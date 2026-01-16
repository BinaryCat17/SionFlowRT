use crate::core::types::{Shape, DataType, Port, Dim, WorkspaceSlot};
use crate::manifest::{Manifest, SourceDef};
use crate::inliner::json::JsonGraph;
use std::collections::HashMap;
use petgraph::algo::toposort;
use anyhow::{Context, anyhow};

#[derive(Debug)]
pub struct Resource {
    pub shape: Shape,
    pub dtype: DataType,
}

#[derive(Debug)]
pub struct ProgramInterface {
    pub inputs: HashMap<String, Port>,
    pub outputs: HashMap<String, Port>,
}

#[derive(Debug)]
pub struct ProjectPlan {
    pub resources: HashMap<String, Resource>,
    pub programs: HashMap<String, ProgramInterface>,
    pub execution_order: Vec<String>,
    pub links: Vec<(String, String)>,
    pub synthetic_vars: HashMap<String, String>, // var_name -> C-expression
    pub workspace_info: HashMap<String, Vec<WorkspaceSlot>>, // prog_id -> list of internal buffers
    pub program_graphs: HashMap<String, JsonGraph>, // Store parsed graphs to avoid re-parsing
}

pub fn analyze_project(manifest: &Manifest, base_path: &std::path::Path) -> anyhow::Result<ProjectPlan> {
    let mut resources = HashMap::new();
    let mut programs = HashMap::new();
    let mut synthetic_vars = HashMap::new();
    let mut program_graphs = HashMap::new();
    
    // Default data type if not specified
    let default_dtype = DataType::F32;

    for (name, def) in &manifest.sources {
        let shape = resolve_source_shape(def, manifest, &mut synthetic_vars)?;
        resources.insert(name.clone(), Resource {
            shape,
            dtype: default_dtype,
        });
    }

    // Phase 1: Load interfaces and identify programs
    for prog_def in &manifest.programs {
        let mut path_buf = base_path.to_path_buf();
        let prog_path_raw = if prog_def.path.ends_with(".json") { 
            prog_def.path.clone() 
        } else { 
            format!("{}.json", prog_def.path) 
        };
        path_buf.push(prog_path_raw);
        
        let path = path_buf.to_string_lossy().to_string();
        
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read program graph file: {}", path))?;
        let json_graph: JsonGraph = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse JSON graph: {}", path))?;

        let mut inputs = HashMap::new();
        for p in &json_graph.inputs {
            inputs.insert(p.name.clone(), Port { 
                name: p.name.clone(), 
                shape: Shape { dims: vec![] }, // Will be resolved via links
                dtype: default_dtype 
            });
        }

        let mut outputs = HashMap::new();
        for p in &json_graph.outputs {
            let mut dims = Vec::new();
            if let Some(js_dims) = &p.shape {
                for js_dim in js_dims {
                    dims.push(process_json_dim(js_dim, &mut synthetic_vars, manifest));
                }
            }

            outputs.insert(p.name.clone(), Port { 
                name: p.name.clone(), 
                shape: Shape { dims }, 
                dtype: default_dtype 
            });
        }

        programs.insert(prog_def.id.clone(), ProgramInterface {
            inputs,
            outputs,
        });
        program_graphs.insert(prog_def.id.clone(), json_graph);
    }

    // Phase 2: Resolve links and build dependency graph
    // ... (logic remains the same)
    let mut dep_graph = petgraph::graph::DiGraph::<String, ()>::new();
    let mut node_indices = HashMap::new();

    for prog_id in programs.keys() {
        node_indices.insert(prog_id.clone(), dep_graph.add_node(prog_id.clone()));
    }

    for (src_addr, dst_addr) in &manifest.links {
        let (src_prog, src_port_name, src_is_resource) = if let Some(res_id) = src_addr.strip_prefix("sources.") {
            (res_id.to_string(), res_id.to_string(), true)
        } else if let Some((prog_id, port_name)) = src_addr.split_once('.') {
            (prog_id.to_string(), port_name.to_string(), false)
        } else {
            continue;
        };

        if let Some((dst_prog_id, dst_port_name)) = dst_addr.split_once('.') {
            // Update input shapes/types based on sources
            let src_port = if src_is_resource {
                let res = resources.get(&src_prog)
                    .ok_or_else(|| anyhow!("Resource '{}' not found for link to '{}.{}'", src_prog, dst_prog_id, dst_port_name))?;
                Port { name: src_prog.clone(), shape: res.shape.clone(), dtype: res.dtype }
            } else {
                let prog = programs.get(&src_prog)
                    .ok_or_else(|| anyhow!("Source program '{}' not found in links", src_prog))?;
                prog.outputs.get(&src_port_name)
                    .cloned()
                    .ok_or_else(|| anyhow!("Output '{}' not found in program '{}'", src_port_name, src_prog))?
            };

            if let Some(prog) = programs.get_mut(dst_prog_id) {
                if let Some(target_port) = prog.inputs.get_mut(dst_port_name) {
                    target_port.shape = src_port.shape;
                    target_port.dtype = src_port.dtype;
                }
            }

            // Add edge to dependency graph if it's a program-to-program link
            if !src_is_resource {
                if let (Some(&u), Some(&v)) = (node_indices.get(&src_prog), node_indices.get(dst_prog_id)) {
                    if u != v {
                        dep_graph.add_edge(u, v, ());
                    }
                }
            }
        }
    }

    // Phase 3: Topological sort for execution order
    let order_indices = toposort(&dep_graph, None)
        .map_err(|_| anyhow!("Circular dependency detected between programs in manifest links"))?;
    
    let execution_order = order_indices.into_iter()
        .map(|idx| dep_graph[idx].clone())
        .collect();

    Ok(ProjectPlan {
        resources,
        programs,
        execution_order,
        links: manifest.links.clone(),
        synthetic_vars,
        workspace_info: HashMap::new(),
        program_graphs,
    })
}

fn resolve_source_shape(
    def: &SourceDef, 
    manifest: &Manifest, 
    synthetic_vars: &mut HashMap<String, String>
) -> anyhow::Result<Shape> {
    let mut dims = Vec::new();
    for (i, val) in def.shape.iter().enumerate() {
        let js_dim: crate::inliner::json::JsonDim = if let Some(s) = val.as_str() {
            crate::inliner::json::JsonDim::Symbol(s.to_string())
        } else if let Some(u) = val.as_u64() {
            crate::inliner::json::JsonDim::Value(u as usize)
        } else {
            serde_json::from_value(val.clone())
                .map_err(|_| anyhow!("Invalid shape dimension at index {} for source", i))?
        };
        dims.push(process_json_dim(&js_dim, synthetic_vars, manifest));
    }
    Ok(Shape { dims })
}

pub fn process_json_dim(
    js_dim: &crate::inliner::json::JsonDim,
    synthetic_vars: &mut HashMap<String, String>,
    manifest: &Manifest
) -> Dim {
    use crate::inliner::json::JsonDim::*;
    match js_dim {
        Value(v) => Dim::Static(*v),
        Symbol(s) => {
            if s == "..." || s == "_" {
                Dim::Variable("dynamic".to_string())
            } else {
                if let Some(p_val) = manifest.parameters.as_ref().and_then(|p| p.get(s)) {
                    let is_dynamic = p_val.get("type").and_then(|t| t.as_str()) == Some("dynamic");
                    if is_dynamic {
                        Dim::Variable(s.clone())
                    } else {
                        let actual_val = p_val.get("value").unwrap_or(p_val);
                        if let Some(u) = actual_val.as_u64() {
                            Dim::Static(u as usize)
                        } else {
                            Dim::Variable(s.clone())
                        }
                    }
                } else {
                    Dim::Variable(s.clone())
                }
            }
        }
        Op(op) => {
            let c_expr = json_dim_op_to_c_expr(op);
            let var_name = format!("var_{}", hash_string(&c_expr));
            synthetic_vars.insert(var_name.clone(), c_expr);
            Dim::Variable(var_name)
        }
        _ => Dim::Variable("dynamic".to_string()),
    }
}

fn json_dim_op_to_c_expr(op: &crate::inliner::json::JsonDimOp) -> String {
    use crate::inliner::json::JsonDimOp::*;
    match op {
        Add(a, b) => format!("({} + {})", json_dim_to_c_expr(a), json_dim_to_c_expr(b)),
        Sub(a, b) => format!("({} - {})", json_dim_to_c_expr(a), json_dim_to_c_expr(b)),
        Mul(a, b) => format!("({} * {})", json_dim_to_c_expr(a), json_dim_to_c_expr(b)),
        Div(a, b) => format!("({} / {})", json_dim_to_c_expr(a), json_dim_to_c_expr(b)),
    }
}

fn json_dim_to_c_expr(js_dim: &crate::inliner::json::JsonDim) -> String {
    use crate::inliner::json::JsonDim::*;
    match js_dim {
        Value(v) => v.to_string(),
        Symbol(s) => s.clone(),
        Op(op) => json_dim_op_to_c_expr(op),
        _ => "1".to_string(),
    }
}

fn hash_string(s: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hasher, Hash};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}
