use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::model::{Dimension, DataType};

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