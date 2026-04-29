use std::collections::HashSet;

use crate::ir::pii::{PiiGraph, PiiId, PiiNode};

/// Top-down DFS approach: traverse from root, copy in-place children on the way down.
/// Single pass, no fixed point needed, but requires recursion.
pub fn fix_inplace_dfs(pii: &mut PiiGraph) {
    let root_id = pii.root;
    let mut visited: HashSet<PiiId> = HashSet::new();
    dfs(pii, root_id, &mut visited);
    remove_dead_nodes(pii);
}

fn count_users(pii: &PiiGraph, target: PiiId) -> usize {
    pii.nodes.values()
        .filter(|n| n.children.contains(&target))
        .count()
}

fn dfs(pii: &mut PiiGraph, node_id: PiiId, visited: &mut HashSet<PiiId>) {
    if visited.contains(&node_id) {
        return;
    }
    visited.insert(node_id);

    let inplace = pii.nodes[&node_id].op.inplace_children();
    let children = pii.nodes[&node_id].children.clone();

    for (i, &child_id) in children.iter().enumerate() {
        if inplace[i] && count_users(pii, child_id) > 1 {
            let child = pii.nodes[&child_id].clone();
            let next_copy = pii.nodes.keys()
                .filter(|k| k.0 == child_id.0)
                .map(|k| k.1)
                .max()
                .unwrap() + 1;
            let new_id = PiiId(child_id.0, next_copy);
            pii.nodes.insert(new_id, PiiNode {
                id: new_id,
                op: child.op,
                info: child.info,
                buffer: child.buffer,
                hbm_offset: child.hbm_offset,
                children: child.children,
            });
            pii.nodes.get_mut(&node_id).unwrap().children[i] = new_id;
            dfs(pii, new_id, visited);
        } else {
            dfs(pii, child_id, visited);
        }
    }
}

/// Fixed-point approach: scan all nodes, find a conflict, copy, repeat.
/// O(N^2) per iteration — not scalable for large PII graphs.
pub fn fix_inplace_fixpoint(pii: &mut PiiGraph) {
    loop {
        let mut conflict: Option<(PiiId, usize)> = None;
        'outer: for (_, node) in pii.nodes.iter() {
            let inplace = node.op.inplace_children();
            for (child_pos, &is_inplace) in inplace.iter().enumerate() {
                if !is_inplace {
                    continue;
                }
                let child_id = node.children[child_pos];
                for (_, other) in pii.nodes.iter() {
                    if other.id == node.id {
                        continue;
                    }
                    if other.children.contains(&child_id) {
                        conflict = Some((node.id, child_pos));
                        break 'outer;
                    }
                }
            }
        }

        match conflict {
            Some((instr_id, child_pos)) => {
                let child_id = pii.nodes[&instr_id].children[child_pos];
                let child = pii.nodes[&child_id].clone();
                let next_copy = pii.nodes.keys()
                    .filter(|k| k.0 == child_id.0)
                    .map(|k| k.1)
                    .max()
                    .unwrap() + 1;
                let new_id = PiiId(child_id.0, next_copy);
                pii.nodes.insert(new_id, PiiNode {
                    id: new_id,
                    op: child.op,
                    info: child.info,
                    buffer: child.buffer,
                    hbm_offset: child.hbm_offset,
                    children: child.children,
                });
                pii.nodes.get_mut(&instr_id).unwrap().children[child_pos] = new_id;
            }
            None => break,
        }
    }

    remove_dead_nodes(pii);
}

fn remove_dead_nodes(pii: &mut PiiGraph) {
    let root_id = pii.root;
    loop {
        let mut used: HashSet<PiiId> = HashSet::new();
        used.insert(root_id);
        for (_, node) in pii.nodes.iter() {
            for &child in &node.children {
                used.insert(child);
            }
        }
        let before = pii.nodes.len();
        pii.nodes.retain(|id, _| used.contains(id));
        if pii.nodes.len() == before {
            break;
        }
    }
}
