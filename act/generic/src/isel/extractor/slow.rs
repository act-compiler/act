use egg::*;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};

use crate::ir::buffer::Buffer;
use crate::ir::egraph::{TensorInfo, TensorOp};
use crate::ir::pii::{PiiGraph, PiiId};

use crate::ir::buffer::buffer_assignment as get_bufs;
use crate::isel::extractor::utils::{annotate_constants, detect_constants, get_hbm_offset};

/// Phase 3: Graph Extractor
/// Extraction phase to select instructions from the final e-graph.

/// Slow exponential-time extraction algorithm. Computes all pii graphs with up to `limit` e-nodes
/// by obtaining full extraction trees and collapsing them with GVN.
pub fn extract_slow(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    root: Id,
    inputs: &HashSet<Id>,
    hbm_offsets: &Vec<(Option<Id>, i32)>,
    limit: usize,
) -> Vec<PiiGraph> {
    // Bottom-up traversal to detect compile-time constants
    let constants = detect_constants(egraph, inputs);
    // Annotate the egraph with DetectedConst nodes and mark analysis data.
    annotate_constants(egraph, &constants);

    // Top-down traversal to perform extraction
    let trees = get_extraction_trees(egraph, root, Buffer::HBM, limit, inputs);
    let mut piis = vec![];
    if !trees.is_empty() {
        for tree in trees.iter() {
            let (mut child_map, mut parent_map) = get_edges(tree);
            let extractions = get_extractions(tree, &mut child_map, &mut parent_map);
            for extraction in extractions.iter() {
                // convert an Extraction into a PiiGraph
                piis.push(extraction_to_piigraph(extraction, hbm_offsets, egraph));
            }
        }
        piis
    } else {
        vec![]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Extraction {
    tree: Vec<TensorOp>,
    children: HashMap<usize, Vec<usize>>,
    parent: HashMap<usize, Option<usize>>,
}

/// Pre-order traversal algorithm for extraction.
/// Returns a list of extractions (set of enodes).
fn get_extraction_trees(
    egraph: &EGraph<TensorOp, TensorInfo>,
    eclass: Id,
    target_buf: Buffer,
    limit: usize,
    inputs: &HashSet<Id>,
) -> Vec<Vec<TensorOp>> {
    let mut trees = vec![];
    if limit == 0 {
        return trees;
    }

    for en in &egraph[eclass].nodes {
        // base case: input enode -- TODO: this assumes only 1 enode in the eclass
        if inputs.contains(&eclass) && target_buf == Buffer::HBM {
            trees.push(vec![en.clone()]);
            break;
        }
        // base case: detected constant enode
        if egraph[eclass].data.is_const && en.is_detected_const() && target_buf == Buffer::HBM {
            trees.push(vec![en.clone()]);
            break;
        }
        // only consider instruction enodes
        if let Some(bufs) = get_bufs(en) {
            if bufs[0] != Buffer::ANY && bufs[0] != target_buf {
                continue;
            }

            let mut child_trees = vec![];
            for (i, child_ec) in en.children().iter().enumerate() {
                let out_buf = if bufs[0] != Buffer::ANY {
                    bufs[1 + i]
                } else {
                    target_buf
                };
                if limit >= en.num_children() {
                    child_trees.push(get_extraction_trees(
                        egraph,
                        *child_ec,
                        out_buf,
                        limit - en.num_children(),
                        inputs,
                    ));
                } else {
                    child_trees = vec![vec![]];
                    break;
                }
            }
            // merge all combinations of child extractions and add en
            if child_trees.iter().all(|x| !x.is_empty()) {
                for combination in child_trees.into_iter().multi_cartesian_product() {
                    let child_trees: Vec<TensorOp> = combination.into_iter().flatten().collect();
                    let mut tree = vec![en.clone()];
                    tree.extend(child_trees);
                    if tree.len() <= limit {
                        trees.push(tree);
                    }
                }
            }
        }
    }
    trees
}

/// Given a preorder extraction tree, recover the parent/child edges between nodes.
fn get_edges(tree: &Vec<TensorOp>) -> (HashMap<usize, Vec<usize>>, HashMap<usize, Option<usize>>) {
    let mut children = HashMap::new();
    let mut parent = HashMap::new();
    let mut stack = vec![];

    for i in 0..tree.len() {
        children.insert(i, vec![]);
        parent.insert(i, None);
    }

    for i in 0..tree.len() {
        if let Some(top) = stack.last() {
            children.get_mut(top).unwrap().push(i);
            *parent.get_mut(&i).unwrap() = Some(*top);
        }
        stack.push(i);
        if tree[i].is_leaf() {
            loop {
                if let Some(top) = stack.last() {
                    if children[top].len() == tree[*top].num_children() {
                        stack.pop();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }
    (children, parent)
}

/// Perform GVN on an extraction tree to collapse equivalent subtrees.
fn get_extractions(
    tree: &Vec<TensorOp>,
    child_map: &mut HashMap<usize, Vec<usize>>,
    parent_map: &mut HashMap<usize, Option<usize>>,
) -> Vec<Extraction> {
    let mut extractions = vec![];
    let mut gvn_map = HashMap::new(); // node -> num
    let mut idx_map = HashMap::new(); // node -> index in tree
    let mut num = 0;
    let mut processed = vec![false; tree.len()];

    // all duplicate leaf nodes must be collapsed
    for (i, node) in tree.iter().enumerate() {
        if node.is_leaf() {
            if !gvn_map.contains_key(node) {
                gvn_map.insert(node.clone(), num);
                num += 1;
                idx_map.insert(node.clone(), i);
            } else {
                if let Some(parent) = parent_map[&i] {
                    let siblings = child_map.get_mut(&parent).unwrap();
                    let child_idx = siblings.iter_mut().position(|x| *x == i).unwrap();
                    siblings[child_idx] = idx_map[node];
                    parent_map.insert(i, None);
                }
            }
            processed[i] = true;
        }
    }
    extractions.push(Extraction {
        tree: tree.clone(),
        parent: parent_map.clone(),
        children: child_map.clone(),
    });

    while processed.iter().any(|&x| !x) {
        for (i, node) in tree.iter().enumerate() {
            // process node when all of its children have been processed
            let children = &child_map[&i];
            if children.iter().all(|&x| processed[x]) && !processed[i] {
                if !gvn_map.contains_key(node) {
                    gvn_map.insert(node.clone(), num);
                    num += 1;
                    idx_map.insert(node.clone(), i);
                } else {
                    // found duplicate node, so collapse
                    if let Some(parent) = parent_map[&i] {
                        let siblings = child_map.get_mut(&parent).unwrap();
                        let child_idx = siblings.iter_mut().position(|x| *x == i).unwrap();
                        siblings[child_idx] = idx_map[node];
                        parent_map.insert(i, None);
                        extractions.push(Extraction {
                            tree: tree.clone(),
                            parent: parent_map.clone(),
                            children: child_map.clone(),
                        });
                    }
                }
                processed[i] = true;
            }
        }
    }

    extractions
}

// Convert an Extraction into a PiiGraph.
fn extraction_to_piigraph(
    extraction: &Extraction,
    hbm_offsets: &Vec<(Option<Id>, i32)>,
    egraph: &EGraph<TensorOp, TensorInfo>,
) -> PiiGraph {
    let mut pii = PiiGraph::default();

    // Build a postorder traversal starting from root index 0 using the extraction's children map.
    let mut postorder: Vec<usize> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    fn dfs(
        idx: usize,
        extraction: &Extraction,
        visited: &mut HashSet<usize>,
        post: &mut Vec<usize>,
    ) {
        if visited.contains(&idx) {
            return;
        }
        visited.insert(idx);
        if let Some(children) = extraction.children.get(&idx) {
            for &c in children.iter() {
                dfs(c, extraction, visited, post);
            }
        }
        post.push(idx);
    }

    if !extraction.tree.is_empty() {
        dfs(0, extraction, &mut visited, &mut postorder);
    }

    // Map tree index -> Pii node id
    let mut idx_map: HashMap<usize, PiiId> = HashMap::new();

    for idx in postorder.into_iter() {
        let op = extraction.tree[idx].clone();
        let child_tree_idxs = extraction.children.get(&idx).cloned().unwrap_or_default();
        let mut child_node_ids: Vec<PiiId> = Vec::with_capacity(child_tree_idxs.len());
        for ti in child_tree_idxs.iter() {
            if let Some(&nid) = idx_map.get(ti) {
                child_node_ids.push(nid);
            }
        }

        let ec_id = egraph.lookup(op.clone()).unwrap();
        let info = egraph[ec_id].data.clone();
        let node_id = pii.add_node(op, info, child_node_ids, get_hbm_offset(hbm_offsets, ec_id));
        pii.root = node_id;
        idx_map.insert(idx, node_id);
    }

    pii
}
