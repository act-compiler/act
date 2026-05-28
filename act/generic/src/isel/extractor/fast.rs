use egg::*;
use std::collections::{HashMap, HashSet};

use crate::ir::buffer::Buffer;
use crate::ir::egraph::{TensorInfo, TensorOp};
use crate::ir::pii::{PiiGraph, PiiId};

use crate::ir::buffer::buffer_assignment as get_bufs;
use crate::isel::extractor::utils::{annotate_constants, detect_constants, get_hbm_offset};

/// Phase 3: Graph Extractor
/// Extraction phase to select instructions from the final e-graph.

/// Fast linear-time extraction algorithm. Computes one pii graph by visiting each e-class (at most
/// once) and adds selected enodes to a `selected` HashSet.
pub fn extract_fast(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    root: Id,
    inputs: &HashSet<Id>,
    hbm_offsets: &Vec<(Option<Id>, i32)>,
) -> Vec<PiiGraph> {
    // Bottom-up traversal to detect compile-time constants
    let constants = detect_constants(egraph, inputs);
    // Annotate the egraph with DetectedConst nodes and mark analysis data.
    annotate_constants(egraph, &constants);

    // Top-down traversal to perform extraction
    let mut selected: HashMap<TensorOp, (Id, TensorInfo)> = HashMap::new();
    let ret = get_pii_graph(
        egraph,
        root,
        Buffer::HBM,
        inputs,
        &mut HashSet::new(),
        &mut selected,
    );
    if ret {
        vec![select2pii_graph(&selected, hbm_offsets)]
    } else {
        vec![]
    }
}

/// Pre-order traversal algorithm for extraction.
/// Returns true if there exists an extraction rooted at eclass.
fn get_pii_graph(
    egraph: &EGraph<TensorOp, TensorInfo>,
    eclass: Id,
    target_buf: Buffer,
    inputs: &HashSet<Id>,
    path: &mut HashSet<Id>,
    selected: &mut HashMap<TensorOp, (Id, TensorInfo)>,
) -> bool {
    if path.contains(&eclass) {
        // no cycles allowed
        return false;
    }

    for en in &egraph[eclass].nodes {
        if let Some(bufs) = get_bufs(en) {
            // memoization case: previously selected enode
            // This is where we lose completeness for fast extraction.
            if selected.get(en).is_some() && target_buf == bufs[0] {
                return true;
            }
        }
    }

    'en_loop: for en in &egraph[eclass].nodes {
        // snapshot selected so we can rollback any provisional selections if this enode
        // attempt fails (backtracking). This prevents nodes selected while exploring
        // a branch that later fails from remaining in `selected`.
        let snapshot = selected.clone();
        // base case: input enode -- TODO: this assumes only 1 enode in the eclass
        if inputs.contains(&eclass) && target_buf == Buffer::HBM {
            selected.insert(en.clone(), (eclass, egraph[eclass].data.clone()));
            return true;
        }
        // base case: detected constant enode
        if egraph[eclass].data.is_const && en.is_detected_const() && target_buf == Buffer::HBM {
            selected.insert(en.clone(), (eclass, egraph[eclass].data.clone()));
            return true;
        }

        // only consider instruction enodes
        if let Some(bufs) = get_bufs(en) {
            if bufs[0] != Buffer::ANY && bufs[0] != target_buf {
                continue;
            }

            path.insert(eclass);
            for (i, child_ec) in en.children().iter().enumerate() {
                let out_buf = if bufs[0] != Buffer::ANY {
                    bufs[1 + i]
                } else {
                    target_buf
                };
                if !get_pii_graph(egraph, *child_ec, out_buf, inputs, path, selected) {
                    // rollback any selections that happened while attempting this enode
                    selected.clear();
                    for (k, v) in snapshot.iter() {
                        selected.insert(k.clone(), v.clone());
                    }
                    // remove current eclass from path before trying next enode
                    path.remove(&eclass);
                    continue 'en_loop;
                }
            }
            path.remove(&eclass);

            selected.insert(en.clone(), (eclass, egraph[eclass].data.clone()));
            return true;
        }
    }
    false
}

/// Convert a HashSet of selected e-nodes into a pii graph.
/// Algorithm builds the pii graph bottom-up in topological order.
fn select2pii_graph(
    selected: &HashMap<TensorOp, (Id, TensorInfo)>,
    hbm_offsets: &Vec<(Option<Id>, i32)>,
) -> PiiGraph {
    let mut pii_graph = PiiGraph::default();

    let mut to_remove: Vec<TensorOp> = vec![];
    // map selected enode's eclass -> pii node id in pii graph
    let mut eclass_map: HashMap<Id, PiiId> = HashMap::new();

    // Add leaf enodes to the pii graph
    for (en, (ec_id, info)) in selected.iter() {
        if en.is_leaf() {
            let new_id = pii_graph.add_node(
                en.clone(),
                info.clone(),
                vec![],
                get_hbm_offset(hbm_offsets, *ec_id),
            );
            eclass_map.insert(*ec_id, new_id);
            to_remove.push(en.clone());
        }
    }

    // Add enodes in topological order
    while to_remove.len() < selected.len() {
        let mut progress = false;
        for (en, (ec_id, info)) in selected.iter() {
            if to_remove.contains(en) {
                continue;
            }
            if en
                .children()
                .iter()
                .all(|child_ec| eclass_map.contains_key(child_ec))
            {
                let children: Vec<PiiId> = en
                    .children()
                    .iter()
                    .map(|child_ec| *eclass_map.get(child_ec).unwrap())
                    .collect();
                let new_id = pii_graph.add_node(
                    en.clone(),
                    info.clone(),
                    children,
                    get_hbm_offset(hbm_offsets, *ec_id),
                );
                eclass_map.insert(*ec_id, new_id);
                pii_graph.root = new_id;
                to_remove.push(en.clone());
                progress = true;
                break;
            }
        }
        if !progress {
            // This should not happen: `selected` is expected to be topologically closed
            // (all children of selected nodes are also selected). Treat this as a bug.
            debug_assert!(false, "select2pii_graph: possible infinite loop");
            break;
        }
    }

    // O(N^2) — see inplace_fix.rs for both implementations
    // use crate::isel::extractor::inplace_fix::fix_inplace_dfs;
    // fix_inplace_dfs(&mut pii_graph);
    // use crate::isel::extractor::inplace_fix::fix_inplace_fixpoint;
    // fix_inplace_fixpoint(&mut pii_graph);

    pii_graph
}
