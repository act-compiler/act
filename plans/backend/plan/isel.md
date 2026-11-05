# Instruction selection, e-graph and rewrites

Rewrites are text lines `N: lhs => rhs` compiled into the binary (`generic/src/isel/rewrites/lib.rs:6-19`), parsed at start-up, and bound by number to three Rust functions each (`applier.rs:15-58`).
The 10 IR-to-IR rewrite rules are hand-written as `precond_0` to `set_shapes_9` (`ir2ir_rewrites.rs`, 680 lines); the IR-to-ISA rewrite rules are generated the same way (`isel_ir2isa_rewrites_rs_generator.py`, 795 lines).
Shapes are not inferred by the e-graph analysis (`ir/egraph.rs:396-401` makes empty class data); each rewrite sets them in `set_shapes_N`.
Standard egg practice is a `Rewrite` built from a pattern and a typed `Applier` or `ConditionalApplier`, and an `Analysis` whose `make` computes the class data.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B6 | One Rust value per rewrite: pattern, condition, result data | B1, B3 | M | Q1 |
| B7 | Shape and element type inferred in `Analysis::make` | B6 | L | Q2 |
| B8 | Name and test each IR-to-IR rewrite rule | B6 | S | Q1 |
| B9 | Extraction without cloning or mutating the e-graph | B7 | M | Q2 |

## B6 One Rust value per rewrite

- What: generate `rewrite!`-style constructors holding the pattern and three closures or a struct implementing `Applier`; drop the text files and the number-keyed `match` arms in `applier.rs:15-58`.
- Preserves: every pattern and every condition body verbatim; `applier.rs`'s matching logic (`find_matches`, `apply_pat`).
- Benefits: a missing or renumbered function is a compile error, not a start-up panic (D6); patterns are checked at build time.
- Caveats: `?`-suffixed op words such as `dot_?` (`ir2ir_rewrites.txt` line 0) match any payload; they need a typed wildcard in `FromOp` (`ir/egraph.rs:222-335`).
- Fail points: a pattern that parses differently as Rust than as text; noticed by goldens.
- Order and dependencies: after B1, B3; before B7, B8, B12.
- Verification: goldens byte-identical; `ir2ir_rewrites.txt` and `ir2isa_rewrites.txt` deleted.

## B7 Shape and element type inferred in `Analysis::make`

- What: compute each class's shape and element type from its e-node in `make`, so rewrites stop writing shapes (`set_shapes_N`) and `merge` checks agreement (`ir/egraph.rs:403-415`).
- Preserves: `TensorOp` variants and the `merge` agreement check (`ir/egraph.rs:403-410`).
- Benefits: deletes every `set_shapes_N` and most `metadata_N`; fixes the rank-0 gap of D8; matches egg's intended use.
- Caveats: needs each op's shape rule in Rust; payload must carry what the rule needs.
- Fail points: a rule that disagrees with the reader's shapes; noticed by the `merge` panic on the examples.
- Order and dependencies: after B6; L-sized, Q2.
- Verification: goldens byte-identical; no `set_shapes_` in the crate.

## B8 Name and test each IR-to-IR rewrite rule

- What: give each of the 10 IR-to-IR rewrite rules a name for what it does, and one unit test that applies it to a small e-graph.
- Preserves: each IR-to-IR rewrite rule's pattern and condition.
- Benefits: a reader can tell `(slice_? (convert_? ?x)) => (convert_? (slice_? ?x))` from rewrite 1 by name.
- Caveats: none beyond test upkeep.
- Fail points: a test that pins the e-graph's internal ids; assert on extracted terms instead.
- Order and dependencies: after B6.
- Verification: `cargo test` covers every IR-to-IR rewrite rule.

## B9 Extraction without cloning or mutating the e-graph

- What: today each pass clones the e-graph (`pipeline.rs:234-240`) because extraction adds `DetectedConst` nodes; compute constant-ness in the analysis instead.
- Preserves: both extraction strategies and their candidate order.
- Benefits: removes a full e-graph copy per pass and a hidden mutation.
- Caveats: depends on B7's analysis.
- Fail points: a constant no longer folded; noticed by goldens.
- Order and dependencies: after B7.
- Verification: `extract` takes `&EGraph`; goldens byte-identical.
