use crate::model::{Op};
use crate::shape_engine::ShapeEngine;
use crate::linear_passes::infer_node_shape_generic;
use crate::pipeline::{UnifiedGraph, Parameters};
use petgraph::visit::EdgeRef;

pub struct OrchestrationPasses;

impl OrchestrationPasses {
    /// Глобальный вывод форм на Unified Graph
    pub fn run_shape_inference(graph: &mut UnifiedGraph, parameters: &Parameters) -> anyhow::Result<()> {
        let mut changed = true;
        for _ in 0..30 {
            if !changed { break; }
            changed = false;

            let order = petgraph::algo::toposort(&*graph, None).map_err(|_| anyhow::anyhow!("Cycle in global graph"))?;
            for idx in order {
                let mut input_shapes = Vec::new();
                let mut incoming: Vec<_> = graph.edges_directed(idx, petgraph::Direction::Incoming).collect();
                incoming.sort_by_key(|e| *e.weight());
                
                for edge in incoming {
                    if let Some(s) = &graph[edge.source()].shape {
                        input_shapes.push(s.clone());
                    }
                }

                let current_shape = graph[idx].shape.clone();
                if let Some(new_s) = infer_node_shape_generic(&graph[idx].op, &input_shapes, current_shape.as_ref()) {
                    let mut s = new_s;
                    for d in &mut s.dims { 
                        *d = d.eval(parameters); 
                        *d = ShapeEngine::simplify(d.clone());
                    }
                    if Some(&s) != current_shape.as_ref() {
                        graph[idx].shape = Some(s);
                        changed = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// Глобальное удаление мертвого кода
    pub fn run_dce(graph: &mut UnifiedGraph, _parameters: &Parameters) -> anyhow::Result<()> {
        let mut keep = std::collections::HashSet::new();
        let mut worklist = Vec::new();

        for idx in graph.node_indices() {
            let node = &graph[idx];
            let is_root = matches!(node.op, Op::Output { .. }) || 
                          graph.edges_directed(idx, petgraph::Direction::Outgoing).count() == 0; 
            
            if is_root {
                worklist.push(idx);
                keep.insert(idx);
            }
        }

        while let Some(idx) = worklist.pop() {
            for edge in graph.edges_directed(idx, petgraph::Direction::Incoming) {
                let src = edge.source();
                if keep.insert(src) {
                    worklist.push(src);
                }
            }
        }

        graph.retain_nodes(|_, idx| keep.contains(&idx));
        Ok(())
    }
}