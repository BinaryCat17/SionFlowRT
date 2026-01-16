use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum JsonDim {
    Value(usize),
    #[serde(rename = "_")]
    Wildcard,
    #[serde(rename = "...")]
    Ellipsis,
    Symbol(String),
    Op(JsonDimOp),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum JsonDimOp {
    Add(Box<JsonDim>, Box<JsonDim>),
    Sub(Box<JsonDim>, Box<JsonDim>),
    Mul(Box<JsonDim>, Box<JsonDim>),
    Div(Box<JsonDim>, Box<JsonDim>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonPort {
    pub name: String,
    pub dtype: Option<String>,
    pub shape: Option<Vec<JsonDim>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonNode {
    pub id: String,
    pub op: Option<serde_json::Value>, // Мы десериализуем операцию позже, когда поймем её тип
    pub subgraph: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonGraph {
    pub imports: Option<HashMap<String, String>>,
    pub inputs: Vec<JsonPort>,
    pub outputs: Vec<JsonPort>,
    pub nodes: Vec<JsonNode>,
    pub links: Vec<(String, String)>,
}

impl JsonGraph {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}
