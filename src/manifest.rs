use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::model::{Dimension, DataType};
use crate::pipeline::{Stage, CompilerContext};
use std::fs;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WindowConfig {
    pub title: String,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProgramEntry {
    pub id: String,
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SourceDef {
    #[serde(rename = "type")]
    pub source_type: String,
    pub shape: Vec<Dimension>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Manifest {
    pub window: Option<WindowConfig>,
    pub parameters: Option<HashMap<String, serde_json::Value>>,
    pub type_mapping: Option<HashMap<String, DataType>>,
    pub sources: HashMap<String, SourceDef>,
    pub programs: Vec<ProgramEntry>,
    pub links: Vec<(String, String)>,
}

impl Manifest {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

pub struct LoadManifestStage;
impl Stage for LoadManifestStage {
    fn name(&self) -> &str { "Load Manifest" }
    fn run(&self, ctx: &mut CompilerContext) -> anyhow::Result<()> {
        let content = fs::read_to_string(&ctx.manifest_path)?;
        ctx.manifest = Some(Manifest::from_json(&content)?);
        Ok(())
    }
}

pub struct ResolveParametersStage;
impl Stage for ResolveParametersStage {
    fn name(&self) -> &str { "Resolve Parameters" }
    fn run(&self, ctx: &mut CompilerContext) -> anyhow::Result<()> {
        let manifest = ctx.manifest.as_ref().ok_or_else(|| anyhow::anyhow!("No manifest loaded"))?;
        let mut params = HashMap::new();

        if let Some(raw_params) = &manifest.parameters {
            for (name, val) in raw_params {
                if let Some(s) = val.as_str() {
                    let resolved_val = match s {
                        "window.width" => manifest.window.as_ref().map(|w| w.width).unwrap_or(0),
                        "window.height" => manifest.window.as_ref().map(|w| w.height).unwrap_or(0),
                        _ => 0,
                    };
                    params.insert(name.clone(), resolved_val);
                } else if let Some(n) = val.as_u64() {
                    params.insert(name.clone(), n as usize);
                }
            }
        }
        ctx.parameters = params;
        Ok(())
    }
}