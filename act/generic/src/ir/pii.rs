use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use crate::ir::buffer::Buffer;
use crate::ir::egraph::{TensorInfo, TensorOp};

use crate::ir::buffer::buffer_assignment as get_bufs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PiiId(pub usize, pub usize);

impl std::fmt::Display for PiiId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t{}_{}", self.0, self.1)
    }
}

#[derive(Debug, Clone)]
pub struct PiiNode {
    pub id: PiiId,
    pub op: TensorOp,
    pub info: TensorInfo,
    pub buffer: Buffer,
    pub hbm_offset: Option<i32>,
    pub children: Vec<PiiId>,
}

#[derive(Debug, Clone)]
pub struct PiiGraph {
    pub nodes: HashMap<PiiId, PiiNode>,
    pub root: PiiId,
}

impl Default for PiiGraph {
    fn default() -> Self {
        PiiGraph {
            nodes: HashMap::new(),
            root: PiiId(0, 0),
        }
    }
}

impl PiiGraph {
    pub fn add_node(
        &mut self,
        op: TensorOp,
        info: TensorInfo,
        children: Vec<PiiId>,
        hbm_offset: Option<i32>,
    ) -> PiiId {
        let mut buffer = match get_bufs(&op) {
            Some(bufs) => bufs[0],
            None => panic!("Not a valid pii node operation: {:?}", op),
        };
        if buffer == Buffer::ANY {
            // If all children have the same buffer, we use that buffer else panic
            let child_buffers: Vec<Buffer> = children
                .iter()
                .map(|c| self.nodes.get(c).expect("child not found").buffer)
                .collect();
            if child_buffers.is_empty() {
                panic!("ANY buffer requires at least one child to infer buffer type");
            }
            let first_buf = child_buffers[0];
            if child_buffers.iter().all(|&b| b == first_buf) {
                buffer = first_buf;
            } else {
                panic!("Cannot infer buffer: children have different buffers");
            }
        }

        let mut info = info;
        if buffer != Buffer::HBM && info.shape.len() == 1 {
            info.shape.push(1);
        }

        let idx = self.nodes.len();
        let id = PiiId(idx, 0);
        self.nodes.insert(
            id,
            PiiNode {
                id,
                op,
                info,
                children,
                buffer,
                hbm_offset,
            },
        );
        id
    }

    fn shape_to_string(shape: &Vec<i32>) -> String {
        let s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("[{}]", s.join(","))
    }

    pub fn save(&self, path: &std::path::PathBuf) {
        let mut file = File::create(path).expect("Unable to create file");
        write!(file, "{}", self).expect("Unable to write data");
    }
}

impl std::fmt::Display for PiiGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut keys: Vec<&PiiId> = self.nodes.keys().collect();
        keys.sort();
        for key in keys {
            let n = &self.nodes[key];
            let dtype = n.info.dtype.to_string();
            let shape = Self::shape_to_string(&n.info.shape);
            let op_str = n.op.to_string();
            let children_str: Vec<String> = n.children.iter().map(|c| c.to_string()).collect();
            writeln!(
                f,
                "{}: {}[{}] = {}{} {}({})",
                n.id,
                n.buffer,
                n.hbm_offset.unwrap_or(-1),
                dtype,
                shape,
                op_str,
                children_str.join(", ")
            )?;
        }
        Ok(())
    }
}
