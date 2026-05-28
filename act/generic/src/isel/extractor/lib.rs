use std::collections::HashSet;
use std::time::Instant;

use egg::{EGraph, Id};

use crate::ir::egraph::{TensorInfo, TensorOp};
use crate::ir::pii::PiiGraph;

use crate::isel::extractor::fast::extract_fast;
use crate::isel::extractor::slow::extract_slow;

use crate::SLOW_LIMIT_CUTOFF;

pub fn extract(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    root: Id,
    inputs: &HashSet<Id>,
    hbm_offsets: &Vec<(Option<Id>, i32)>,
    limit: usize,
    fast_only: bool,
) -> Vec<PiiGraph> {
    let nodes = egraph.total_number_of_nodes();

    let start = Instant::now();
    let mut piis_fast = extract_fast(egraph, root, &inputs, &hbm_offsets);

    log!("Fast Extractor Algorithm over #nodes={}", nodes);
    log!("Number of PII graphs extracted: {}", piis_fast.len());
    log!("Extraction time: {:?}", start.elapsed());

    log!();

    if fast_only {
        return piis_fast;
    }

    let mut piis_slow: Vec<PiiGraph> = vec![];
    if limit <= SLOW_LIMIT_CUTOFF {
        let start = Instant::now();
        piis_slow = extract_slow(egraph, root, &inputs, &hbm_offsets, limit);

        log!("Slow Extractor Algorithm over #nodes={}", nodes);
        log!("Limit used: {}", limit);
        log!("Number of PII graphs extracted: {}", piis_slow.len());
        log!("Extraction time: {:?}", start.elapsed());
    } else {
        log!(
            "Skipping Slow Extractor Algorithm (limit = {} > {})",
            limit, SLOW_LIMIT_CUTOFF
        );
    }

    log!();

    piis_fast.append(&mut piis_slow);

    piis_fast
}
