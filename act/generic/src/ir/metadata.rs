use std::fs::File;
use std::io::Write;

use crate::ir::dtype::Dtype;

#[derive(Debug, Clone, serde::Serialize)]
pub struct MetaDataInfo {
    pub addr: i32,
    pub shape: Vec<i32>,
    pub dtype: Dtype,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MetaData {
    pub module_name: String,
    pub input: Vec<MetaDataInfo>,
    pub output: Vec<MetaDataInfo>,
}

impl MetaData {
    pub fn save(&self, path: &std::path::PathBuf) {
        let mut file = File::create(path).expect("Unable to create file");
        let json = serde_json::to_string_pretty(&self).expect("Unable to serialize data");
        write!(file, "{}", json).expect("Unable to write data");
    }
}

impl Default for MetaData {
    fn default() -> Self {
        MetaData {
            module_name: "xla_computation_unknown".to_string(),
            input: vec![],
            output: vec![],
        }
    }
}
