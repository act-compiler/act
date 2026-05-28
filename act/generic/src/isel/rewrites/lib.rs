use std::fs::read_to_string;
use std::path::PathBuf;

use egg::Rewrite;

use crate::ir::egraph::{TensorInfo, TensorOp};
use crate::isel::rewrites::applier::get_applier;

pub fn get_rewrites() -> Vec<Rewrite<TensorOp, TensorInfo>> {
    let this_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(file!());
    let pwd = this_file_path
        .parent()
        .expect("Failed to get rewrites/ directory");

    let ir2ir = read_to_string(pwd.join("ir2ir_rewrites.txt"))
        .expect("Could not open file ir2ir_rewrites.txt");
    let ir2ir: Vec<&str> = ir2ir.split('\n').filter(|s| !s.is_empty()).collect();

    let ir2isa = read_to_string(pwd.join("ir2isa_rewrites.txt"))
        .expect("Could not open file ir2isa_rewrites.txt");
    let ir2isa: Vec<&str> = ir2isa.split('\n').filter(|s| !s.is_empty()).collect();

    let mut rewrites: Vec<Rewrite<TensorOp, TensorInfo>> = vec![];
    for rewrite_str in [ir2ir, ir2isa].concat().iter() {
        rewrites.push(get_applier(rewrite_str));
    }
    rewrites
}
