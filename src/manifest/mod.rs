use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SourceDef {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub shape: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProgramDef {
    pub id: String,
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Test {
    pub name: String,
    pub inputs: BTreeMap<String, Vec<f32>>,
    pub expected: BTreeMap<String, Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Manifest {
    pub sources: BTreeMap<String, SourceDef>,
    pub programs: Vec<ProgramDef>,
    pub links: Vec<(String, String)>,
    #[serde(default)]
    pub tests: Vec<Test>,
    #[serde(default)]
    pub parameters: Option<BTreeMap<String, serde_json::Value>>,
}

impl Manifest {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}
