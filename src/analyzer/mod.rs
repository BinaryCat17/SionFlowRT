use crate::core::types::{Shape, DataType, Port, Dim};
use crate::manifest::{Manifest, SourceDef};
use crate::inliner::json::JsonGraph;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Resource {
    pub shape: Shape,
    pub dtype: DataType,
    pub kind: Option<String>,
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
}

pub fn analyze_project(manifest: &Manifest) -> anyhow::Result<ProjectPlan> {
    let mut resources = HashMap::new();
    let mut programs = HashMap::new();
    
    for (name, def) in &manifest.sources {
        let shape = resolve_source_shape(def, manifest)?;
        resources.insert(name.clone(), Resource {
            shape,
            dtype: DataType::F32,
            kind: def.kind.clone(),
        });
    }

    for prog_def in &manifest.programs {
        let path = if prog_def.path.ends_with(".json") { 
            prog_def.path.clone() 
        } else { 
            format!("{}.json", prog_def.path) 
        };
        
        let content = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("Failed to inspect {}: {}", path, e))?;
        let json_graph: JsonGraph = serde_json::from_str(&content)?;

        let mut inputs = HashMap::new();
        for p in json_graph.inputs {
            inputs.insert(p.name.clone(), Port { 
                name: p.name, 
                shape: Shape { dims: vec![] }, 
                dtype: DataType::F32 
            });
        }

        let mut outputs = HashMap::new();
        for p in json_graph.outputs {
            let shape = if let Some(js_dims) = p.shape {
                let dims = js_dims.into_iter().map(|d| match d {
                    crate::inliner::json::JsonDim::Value(v) => Dim::Static(v),
                    crate::inliner::json::JsonDim::Symbol(s) => Dim::Variable(s),
                    _ => Dim::Variable("dynamic".to_string()),
                }).collect();
                Shape { dims }
            } else {
                Shape { dims: vec![] }
            };

            outputs.insert(p.name.clone(), Port { 
                name: p.name, 
                shape, 
                dtype: DataType::F32 
            });
        }

        programs.insert(prog_def.id.clone(), ProgramInterface {
            inputs,
            outputs,
        });
    }

    for (src_addr, dst_addr) in &manifest.links {
        let src_port = if let Some(res_id) = src_addr.strip_prefix("sources.") {
            let res = resources.get(res_id).ok_or_else(|| anyhow::anyhow!("Resource {} not found", res_id))?;
            Port { name: res_id.to_string(), shape: res.shape.clone(), dtype: res.dtype }
        } else if let Some((prog_id, port_name)) = src_addr.split_once('.') {
            let prog = programs.get(prog_id).ok_or_else(|| anyhow::anyhow!("Program {} not found in links", prog_id))?;
            prog.outputs.get(port_name).cloned().ok_or_else(|| anyhow::anyhow!("Output {} not found in program {}", port_name, prog_id))?
        } else {
            continue;
        };

        if let Some((prog_id, port_name)) = dst_addr.split_once('.') {
            if let Some(prog) = programs.get_mut(prog_id) {
                if prog.inputs.contains_key(port_name) {
                    prog.inputs.insert(port_name.to_string(), src_port);
                }
            }
        }
    }

    Ok(ProjectPlan {
        resources,
        programs,
        execution_order: manifest.programs.iter().map(|p| p.id.clone()).collect(),
    })
}

fn resolve_source_shape(def: &SourceDef, manifest: &Manifest) -> anyhow::Result<Shape> {
    let mut dims = Vec::new();
    let params = manifest.parameters.as_ref();
    let dynamic = manifest.dynamic_parameters.as_ref();

    for val in &def.shape {
        if let Some(s) = val.as_str() {
            if dynamic.map_or(false, |d| d.contains(&s.to_string())) {
                dims.push(Dim::Variable(s.to_string()));
            } else if let Some(p_val) = params.and_then(|p| p.get(s)) {
                if let Some(u) = p_val.as_u64() {
                    dims.push(Dim::Static(u as usize));
                } else {
                    dims.push(Dim::Variable(s.to_string()));
                }
            } else {
                dims.push(Dim::Variable(s.to_string()));
            }
        } else if let Some(u) = val.as_u64() {
            dims.push(Dim::Static(u as usize));
        }
    }
    Ok(Shape { dims })
}
