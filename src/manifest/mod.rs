use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::core::types::DataType;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Manifest {
    pub parameters: Option<HashMap<String, serde_json::Value>>,
    pub dynamic_parameters: Option<Vec<String>>,
    pub type_mapping: Option<HashMap<String, DataType>>,
    pub sources: HashMap<String, SourceDef>,
    pub programs: Vec<ProgramDef>,
    pub links: Vec<(String, String)>,
}

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

impl Manifest {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}