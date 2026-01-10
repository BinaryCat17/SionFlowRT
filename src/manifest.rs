use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
#[serde(tag = "type")]
pub enum MappingSource {
    MousePosition,
    MouseButton { button: String },
    ScreenUV,
    Time,
    Display,
    /// Ссылка на выход другого (или этого же) графа: [program_id, tensor_id]
    Link { program: String, output: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Mapping {
    pub program: String,
    pub tensor: String,
    pub source: MappingSource,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Manifest {
    pub window: Option<WindowConfig>,
    pub parameters: Option<HashMap<String, usize>>,
    pub programs: Vec<ProgramEntry>,
    pub mappings: Vec<Mapping>,
}

impl Manifest {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}