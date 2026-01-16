use crate::analyzer::ProjectPlan;
use crate::manifest::Test;
use crate::core::types::Dim;
use crate::core::utils::sanitize_id;
use std::collections::{HashSet};
use tera::{Tera, Context};

pub fn generate_test_runner(_plan: &ProjectPlan, tests: &[Test]) -> String {
    let mut tera = Tera::default();
    tera.add_raw_template("test_runner", include_str!("../../templates/test_runner.c.tera")).unwrap();

    let mut context = Context::new();
    
    let mut rendered_tests = Vec::new();
    for test in tests {
        let mut inputs = Vec::new();
        for (name, data) in &test.inputs {
            let mut formatted_data = Vec::new();
            for val in data {
                formatted_data.push(if val.fract() == 0.0 { format!("{}.0f", val) } else { format!("{}f", val) });
            }
            inputs.push(serde_json::json!({
                "id": sanitize_id(name),
                "data": formatted_data
            }));
        }

        let mut outputs = Vec::new();
        for (name, expected) in &test.expected {
            let sanitized = sanitize_id(name);
            let buf_name = if name.contains('.') {
                format!("buf_{}", sanitized)
            } else {
                format!("resource_{}", sanitized)
            };
            
            let mut expected_items = Vec::new();
            for (idx, val) in expected.iter().enumerate() {
                expected_items.push(serde_json::json!({
                    "idx": idx,
                    "val": if val.fract() == 0.0 { format!("{}.0f", val) } else { format!("{}f", val) }
                }));
            }

            outputs.push(serde_json::json!({
                "full_name": name,
                "buf_name": buf_name,
                "expected_items": expected_items
            }));
        }

        rendered_tests.push(serde_json::json!({
            "name": test.name,
            "inputs": inputs,
            "outputs": outputs
        }));
    }

    context.insert("tests", &rendered_tests);
    tera.render("test_runner", &context).expect("Failed to render test_runner template")
}

pub fn generate_runtime_c(plan: &ProjectPlan) -> String {
    let mut tera = Tera::default();
    tera.add_raw_template("runtime", include_str!("../../templates/runtime.c.tera")).unwrap();

    let mut context = Context::new();

    // 1. All variables
    let mut all_vars = HashSet::new();
    for interface in plan.programs.values() {
        for port in interface.inputs.values().chain(interface.outputs.values()) {
            for dim in &port.shape.dims {
                if let Dim::Variable(v) = dim {
                    all_vars.insert(v.clone());
                }
            }
        }
    }
    for var in plan.synthetic_vars.keys() {
        all_vars.insert(var.clone());
    }
    let mut sorted_vars: Vec<_> = all_vars.into_iter().collect();
    sorted_vars.sort();
    context.insert("vars", &sorted_vars);

    // 2. Resources
    let mut resources = Vec::new();
    for (id, res) in &plan.resources {
        resources.push(serde_json::json!({
            "id": sanitize_id(id),
            "dtype": res.dtype.to_c_type(),
            "size_expr": res.shape.to_c_size_expr()
        }));
    }
    context.insert("resources", &resources);

    // 3. Programs
    let mut programs = Vec::new();
    for prog_id in &plan.execution_order {
        let interface = &plan.programs[prog_id];
        
        let mut out_ports = Vec::new();
        for (name, port) in &interface.outputs {
            out_ports.push(serde_json::json!({
                "id": sanitize_id(name),
                "dtype": port.dtype.to_c_type(),
                "size_expr": port.shape.to_c_size_expr()
            }));
        }

        let mut workspace_slots = Vec::new();
        if let Some(slots) = plan.workspace_info.get(prog_id) {
            for slot in slots {
                workspace_slots.push(serde_json::json!({
                    "dtype": slot.dtype.to_c_type(),
                    "size_expr": slot.shape.to_c_size_expr()
                }));
            }
        }

        let mut call_args = Vec::new();
        let mut in_names: Vec<_> = interface.inputs.keys().collect();
        in_names.sort();
        for name in &in_names {
            let target_addr = format!("{}.{}", prog_id, name);
            let mut found = false;
            for (src_addr, dst_addr) in &plan.links {
                if dst_addr == &target_addr {
                    if let Some(res_id) = src_addr.strip_prefix("sources.") {
                        call_args.push(format!("resource_{}", sanitize_id(res_id)));
                    } else if let Some((src_p, src_port)) = src_addr.split_once('.') {
                        call_args.push(format!("buf_{}_{}", sanitize_id(src_p), sanitize_id(src_port)));
                    }
                    found = true;
                    break;
                }
            }
            if !found { call_args.push("NULL".to_string()); }
        }
        let mut out_names: Vec<_> = interface.outputs.keys().collect();
        out_names.sort();
        for name in &out_names {
            call_args.push(format!("buf_{}_{}", sanitize_id(prog_id), sanitize_id(name)));
        }

        programs.push(serde_json::json!({
            "id": sanitize_id(prog_id),
            "inputs": in_names,
            "outputs": out_names,
            "outputs_ports": out_ports,
            "workspace_size": workspace_slots.len(),
            "workspace_slots": workspace_slots,
            "call_args": call_args
        }));
    }
    context.insert("programs", &programs);

    // 4. Synthetic Vars
    let mut syn_vars = Vec::new();
    let mut sorted_syn: Vec<_> = plan.synthetic_vars.keys().collect();
    sorted_syn.sort();
    for k in sorted_syn {
        syn_vars.push((k, &plan.synthetic_vars[k]));
    }
    context.insert("synthetic_vars", &syn_vars);

    // 5. Sync Back
    let mut sync_back = Vec::new();
    for (src_addr, dst_addr) in &plan.links {
        if let Some(res_id) = dst_addr.strip_prefix("sources.") {
            if let Some((src_p, src_port)) = src_addr.split_once('.') {
                if src_p != "sources" {
                    let res = &plan.resources[res_id];
                    sync_back.push(serde_json::json!({
                        "res_id": sanitize_id(res_id),
                        "src_prog": sanitize_id(src_p),
                        "src_port": sanitize_id(src_port),
                        "dtype": res.dtype.to_c_type(),
                        "size_expr": res.shape.to_c_size_expr()
                    }));
                }
            }
        }
    }
    context.insert("sync_back", &sync_back);

    tera.render("runtime", &context).expect("Failed to render runtime template")
}