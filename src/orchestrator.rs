use crate::ir_graph::{IRGraph, IRNode};
use crate::manifest::Manifest;
use crate::model::{DataType, TensorShape};
use crate::linear_ir::{LinearIR, LinearNode};
use crate::pipeline::{Stage, CompilerContext, GlobalPassFn};
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use petgraph::algo::toposort;
use std::collections::HashMap;
use serde::Serialize;

#[derive(Debug, Serialize, Clone)]
pub struct GlobalResource {
    pub id: String,
    pub shape: TensorShape,
    pub dtype: DataType,
    pub source_type: Option<String>,
    pub is_state: bool,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResourceBinding {
    pub resource_id: String,
    pub program_port: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ProgramInstance {
    pub id: String,
    pub inputs: Vec<ResourceBinding>,
    pub outputs: Vec<ResourceBinding>,
}

#[derive(Debug, Default, Serialize, Clone)]
pub struct ProjectOrchestration {
    pub resources: HashMap<String, GlobalResource>,
    pub instances: Vec<ProgramInstance>,
    #[serde(skip)]
    pub programs: HashMap<String, LinearIR>,
}

pub struct OrchestrationStage {
    passes: Vec<GlobalPassFn>,
}

impl OrchestrationStage {
    pub fn new() -> Self { Self { passes: Vec::new() } }
    pub fn with_pass(mut self, pass: GlobalPassFn) -> Self {
        self.passes.push(pass);
        self
    }
}

impl Stage for OrchestrationStage {
    fn name(&self) -> &str { "Global Orchestration & Analysis" }
    fn run(&self, ctx: &mut CompilerContext) -> anyhow::Result<()> {
        let manifest = ctx.manifest.as_ref().unwrap();
        
        // 1. Сборка Unified Graph
        let mut unified_graph = Orchestrator::build_unified_graph(manifest, &ctx.ir_graphs)?;
        
        // 2. Запуск глобальных пассов
        for pass in &self.passes {
            pass(&mut unified_graph, &ctx.parameters)?;
        }
        
        ctx.unified_graph = Some(unified_graph);
        Ok(())
    }
}

pub struct Orchestrator;

impl Orchestrator {
    pub fn build_unified_graph(
        manifest: &Manifest,
        ir_graphs: &HashMap<String, IRGraph>
    ) -> anyhow::Result<DiGraph<IRNode, usize>> {
        let mut unified_graph = DiGraph::new();
        let mut node_map = HashMap::new();

        for (prog_id, ir) in ir_graphs {
            for idx in ir.graph.node_indices() {
                let node = &ir.graph[idx];
                let new_idx = unified_graph.add_node(IRNode {
                    id: node.id.clone(),
                    op: node.op.clone(),
                    shape: node.shape.clone(),
                    dtype: node.dtype.clone(),
                    program_id: Some(prog_id.clone()),
                });
                node_map.insert((prog_id.clone(), node.id.clone()), new_idx);
            }
            
            for edge in ir.graph.edge_references() {
                let src = &ir.graph[edge.source()];
                let dst = &ir.graph[edge.target()];
                unified_graph.add_edge(
                    node_map[&(prog_id.clone(), src.id.clone())],
                    node_map[&(prog_id.clone(), dst.id.clone())],
                    *edge.weight()
                );
            }
        }

        for (src_addr, dst_addr) in &manifest.links {
            if src_addr.starts_with("sources.") {
                let res_id = src_addr.strip_prefix("sources.").unwrap();
                if let Some(res_def) = manifest.sources.get(res_id) {
                    let dst_parts: Vec<&str> = dst_addr.split('.').collect();
                    let prog_id = dst_parts[0];
                    let port_name = dst_parts.last().unwrap();
                    if let Some(&idx) = node_map.get(&(prog_id.to_string(), port_name.to_string())) {
                        unified_graph[idx].shape = Some(TensorShape { dims: res_def.shape.clone() });
                    }
                }
                continue;
            }

            if dst_addr.starts_with("sources.") {
                continue;
            }

            let src_parts: Vec<&str> = src_addr.split('.').collect();
            let dst_parts: Vec<&str> = dst_addr.split('.').collect();

            if src_parts.len() == 2 && dst_parts.len() == 2 {
                let (s_prog, s_port) = (src_parts[0], src_parts[1]);
                let (d_prog, d_port) = (dst_parts[0], dst_parts[1]);

                if let Some(s_node_id) = ir_graphs.get(s_prog).and_then(|g| g.outputs.get(s_port)) {
                    if let (Some(&s_idx), Some(&d_idx)) = (node_map.get(&(s_prog.to_string(), s_node_id.clone())), node_map.get(&(d_prog.to_string(), d_port.to_string()))) {
                        let d_ir = &ir_graphs[d_prog];
                        let d_node_local_idx = d_ir.graph.node_indices().find(|&i| d_ir.graph[i].id == d_port).unwrap();
                        let interface = crate::ir_graph::KernelRegistry::get_interface(&d_ir.graph[d_node_local_idx].op);
                        let port_idx = interface.inputs.iter().position(|p| p.name == d_port).unwrap_or(0);

                        unified_graph.add_edge(s_idx, d_idx, port_idx);
                    }
                }
            }
        }

        Ok(unified_graph)
    }

    pub fn compile_to_orchestration(
        manifest: &Manifest,
        unified_graph: &DiGraph<IRNode, usize>
    ) -> anyhow::Result<ProjectOrchestration> {
        let mut resources = HashMap::new();
        let mut instances = Vec::new();
        let mut programs = HashMap::new();

        for (name, def) in &manifest.sources {
            resources.insert(name.clone(), GlobalResource {
                id: name.clone(),
                shape: TensorShape { dims: def.shape.clone() },
                dtype: DataType::F32,
                source_type: Some(def.source_type.clone()),
                is_state: manifest.links.iter().any(|(s, _)| s == &format!("sources.{}", name)) && 
                          manifest.links.iter().any(|(_, d)| d == &format!("sources.{}", name)),
            });
        }

        for prog_entry in &manifest.programs {
            let prog_id = &prog_entry.id;
            let mut linear_nodes = Vec::new();
            let mut outputs = HashMap::new();
            
            let order = toposort(&unified_graph, None).map_err(|_| anyhow::anyhow!("Cycle in Unified Graph"))?;
            
            for &idx in &order {
                let node = &unified_graph[idx];
                if node.program_id.as_ref() == Some(prog_id) {
                    let mut inputs = Vec::new();
                    let mut incoming: Vec<_> = unified_graph.edges_directed(idx, petgraph::Direction::Incoming).collect();
                    incoming.sort_by_key(|e| *e.weight());

                    for edge in incoming {
                        let src_node = &unified_graph[edge.source()];
                        if src_node.program_id.as_ref() == Some(prog_id) {
                            inputs.push(src_node.id.clone());
                        }
                    }

                    linear_nodes.push(LinearNode {
                        id: node.id.clone(),
                        op: node.op.clone(),
                        inputs,
                        shape: node.shape.clone().unwrap_or_else(|| TensorShape { dims: vec![] }),
                        dtype: node.dtype.as_ref().and_then(|t| manifest.type_mapping.as_ref().and_then(|m| m.get(t))).cloned().unwrap_or(DataType::F32),
                    });
                }
            }

            let mut instance = ProgramInstance { id: prog_id.clone(), inputs: Vec::new(), outputs: Vec::new() };
            for (src_addr, dst_addr) in &manifest.links {
                if src_addr.starts_with("sources.") && dst_addr.starts_with(prog_id) {
                    instance.inputs.push(ResourceBinding {
                        resource_id: src_addr.strip_prefix("sources.").unwrap().to_string(),
                        program_port: dst_addr.split('.').last().unwrap().to_string(),
                    });
                }
                if dst_addr.starts_with("sources.") && src_addr.starts_with(prog_id) {
                    let port_name = src_addr.split('.').last().unwrap().to_string();
                    instance.outputs.push(ResourceBinding {
                        resource_id: dst_addr.strip_prefix("sources.").unwrap().to_string(),
                        program_port: port_name.clone(),
                    });
                    outputs.insert(port_name.clone(), port_name);
                }
            }

            programs.insert(prog_id.clone(), LinearIR {
                nodes: linear_nodes,
                outputs,
                groups: Vec::new(),
            });
            instances.push(instance);
        }

        Ok(ProjectOrchestration { resources, instances, programs })
    }
}