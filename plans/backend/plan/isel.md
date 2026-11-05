# Instruction selection, e-graph and rewrites

Rewrites are text lines `N: lhs => rhs` compiled into the binary (`generic/src/isel/rewrites/lib.rs:6-19`), parsed at start-up, and bound by number to three Rust functions each (`applier.rs:15-58`).
Payloads are strings (`OpDot(String, ..)`, `ir/egraph.rs:10-28`) re-split at every condition by `payload_ints` (`egraph.rs:63-94`), and `Language::matches` compares only the discriminant (`egraph.rs:163-165`), so a payload in a pattern is never matched.
Shapes are not inferred by the analysis (`egraph.rs:396-401`); each rewrite sets them in `set_shapes_N`, and conditions index pattern positions (`rhs_metadata[1]`, `ir2ir_rewrites.rs:78-85,103-135`).
Standard egg practice is an `Analysis` whose `make` computes class data, and a `Rewrite` whose condition reads bound variables.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B31 | Typed payloads in `TensorOp` | B1 | M | Q1 |
| B7 | Shape and element type inferred in `Analysis::make` | B31 | L | Q1 |
| B9 | Extraction without cloning or mutating the e-graph | B7 | S | Q1 |
| B6 | One rewrite value per rule, variables bound by name | B7, B9 | M | Q1 |
| B8 | Name and test each IR-to-IR rewrite rule | B6 | S | Q1 |
| B33 | Target constants out of IR-to-IR rewrite 0 | B6 | S | Q2 |

## B31 Typed payloads in `TensorOp`

- What: each variant carries typed fields (dims as integer vectors, `Dtype`) instead of a `String`; text exists only in `Display` for `--log`.
- Preserves: variant set; `--log` text.
- Benefits: removes `payload_ints` and `set_metadata`'s `expect` (`egraph.rs:137-150`); B7 rules and B2 structs read fields directly.
- Caveats: `FromOp` (`egraph.rs:222-335`) parses `dot_?` by splitting at `_`; the typed form states that payloads never match in patterns and conditions carry them.
- Fail points: a payload parsed with a different field order; noticed by the B3 gate.
- Order and dependencies: first e-graph item, after B1.
- Verification: B3 gate identical; no `payload_ints` in the crate.

## B7 Shape and element type inferred in `Analysis::make`

- What: compute each class's shape and element type from its e-node and children in `make`; shape is `Option`; `merge` asserts agreement.
- Preserves: `TensorOp` variants and the `merge` agreement check (`egraph.rs:403-410`).
- Benefits: deletes every `set_shapes_N`, most `metadata_N`, and the reader's 17 `set_analysis_data` calls; fixes D8.
- Caveats: needs each operation's shape rule in Rust, sharing a table with the front end where one exists.
- Fail points: a rule that disagrees with the reader's shapes; noticed by the `merge` panic on the examples.
- Order and dependencies: after B31; before B6 so `set_shapes_N` is never regenerated.
- Verification: B3 gate identical and oracle run equal; no `set_shapes_` in the crate.

## B9 Extraction without cloning or mutating the e-graph

- What: constant-ness and the folded value live in class data computed by `make`; candidate building emits the constant node from class data, so no `DetectedConst` node enters the e-graph.
- Preserves: the folded value the allocator evaluates (`act_malloc.cc:74-78`).
- Benefits: removes the e-graph clone per pass (`pipeline.rs:234-240`) and the mutation (`slow.rs:24`).
- Caveats: adding the node in `Analysis::modify` would grow the e-graph during saturation and count against `NODE_LIMIT`.
- Fail points: a constant no longer folded; noticed by the B3 gate.
- Order and dependencies: after B7; before B10, since both touch `annotate_constants`.
- Verification: `extract` takes `&EGraph`; B3 gate identical.

## B6 One rewrite value per rule

- What: each rewrite is one value holding its pattern, a condition over named pattern variables, and result attributes; the text files and number-keyed arms of `applier.rs:15-58` go.
- What: open for the panel: generated Rust per rule, or data per instruction read by one generic applier.
- Preserves: each condition's logic; the re-match in `applier.rs:106-113,168-215` only where a condition reads an unbound inner e-node.
- Benefits: a missing rule is a compile error, not a start-up panic (D6); conditions stop indexing positions.
- Caveats: rule 6 (`convert_s32 (convert_s8 ?x)`) is enforced only by `precond_6` (`ir2ir_rewrites.rs:514-526`); its condition must carry both payloads.
- Fail points: a condition bound to the wrong variable; noticed by B8 tests and the B3 gate.
- Order and dependencies: after B7 and B9; before B8, B12, B33.
- Verification: B3 gate identical; `ir2ir_rewrites.txt` and `ir2isa_rewrites.txt` deleted.

## B8 Name and test each IR-to-IR rewrite rule

- What: give each of the 10 IR-to-IR rules a name and two unit tests: it fires on a minimal e-graph and does not fire on a near miss.
- Preserves: each rule's pattern and condition.
- Benefits: rule 6's payload check and rule 0's target condition are pinned by tests.
- Caveats: tests assert on extracted terms, not on e-graph ids.
- Fail points: a test that pins internal ids; avoided by asserting terms.
- Order and dependencies: after B6.
- Verification: `cargo test` covers every IR-to-IR rule.

## B33 Target constants out of IR-to-IR rewrite 0

- What: rewrite 0 hard-codes tile size `16` and `Dtype::I32` (`ir2ir_rewrites.rs:20-27,100`) inside `generic/`; generate those values from the spec's instructions, or drop the rule where no instruction needs it.
- Preserves: rule 0's result on the examples.
- Benefits: the `generic/` crate holds no target constant.
- Caveats: the generator must say which instruction the tile comes from.
- Fail points: a spec with another tile size loses the split; noticed by a test spec with tile 32.
- Order and dependencies: after B6.
- Verification: no literal `16` tied to a `Dtype` in `generic/src/`; B3 gate identical.
