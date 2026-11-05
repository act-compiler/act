# Instruction selection, e-graph and rewrites

Rewrites are text lines `N: lhs => rhs` compiled into the binary (`generic/src/isel/rewrites/lib.rs:6-19`), parsed at start-up, and bound by number to three Rust functions each (`applier.rs:15-58`).
Payloads are strings (`OpDot(String, ..)`, `ir/egraph.rs:10-28`) re-split at every condition by `payload_ints` (`egraph.rs:63-94`), and `Language::matches` compares only the discriminant (`egraph.rs:163-165`), so a payload in a pattern is never matched.
Shapes are not inferred by the analysis (`egraph.rs:396-401`); each rewrite sets them in `set_shapes_N`, and conditions index pattern positions (`rhs_metadata[1]`, `ir2ir_rewrites.rs:78-85,103-135`).
The order is typed payloads, shape in `make`, no clone, then rewrite values (Q4.3).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B31 | Typed payloads in `TensorOp` | B1, B3 | M | Q1 |
| B7 | Shape and element type inferred in `Analysis::make` | B31, B3 | L | Q1 |
| B57 | Class data record and `merge` change report for B49 | B7 | S | later |
| B9 | Extraction without cloning or mutating the e-graph | B7, B3 | S | Q1 |
| B6 | One rewrite value per rule, variables bound by name | B7, B9, B3 | M | Q1 |
| B8 | Name and test each IR-to-IR rewrite rule | B6 | S | Q1 |
| B35 | Overlapping IR-to-ISA rules: declaration-index tie-break, duplicate warning | B6, B29 | S | Q1 |
| B33 | Target constants out of IR-to-IR rewrite 0 | B6 | S | Q2 |

## B31 Typed payloads in `TensorOp`

- What: each variant carries typed fields (dims as integer vectors, `Dtype`) instead of a `String`; text exists only in `Display` for `--log`.
- What: the payload is plain integers and a dtype enum with a size and offset test; it becomes B49's e-node payload and hash-cons key unchanged (Q12.4; needed now).
- Preserves: variant set; `--log` text.
- Benefits: removes `payload_ints` and `set_metadata`'s `expect` (`egraph.rs:137-150`); B7 rules and B2 structs read fields directly.
- Caveats: `FromOp` (`egraph.rs:222-335`) parses `dot_?` by splitting at `_`; payloads never match in patterns, and conditions carry them.
- Fail points: a payload parsed with a different field order; noticed by the B3 gate.
- Order and dependencies: first e-graph item, after B1.
- Verification: B3 gate identical; no `payload_ints` in the crate.

## B7 Shape and element type inferred in `Analysis::make`

- What: compute each class's shape and element type from its e-node and children in `make`; shape is `Option`; `make` asserts that the inferred shape equals the annotation (Q4.8).
- What: B7 is the shape rules only and closes at M1; the class data record is B57, which waits for B49.
- What: the Rust shape rules cover the words the backend supports and the nodes rewrites create (`slice_?`, `concat_0`); U4's table is the source, generated from it once the language track owns one table (Q4.8).
- Preserves: `TensorOp` variants and the `merge` agreement check, which B26 turns into a coded fault.
- Benefits: deletes every `set_shapes_N`, most `metadata_N`, and the reader's 17 `set_analysis_data` calls; fixes D8.
- Caveats: until U4's table is generated into Rust, the Rust subset can drift from it; the oracle run catches a drift on the examples.
- Fail points: a rule that disagrees with the reader's shapes; noticed by the coded shape fault on the examples.
- What: after the engine swap (B52), the taidl shape module stays the authoritative copy; the Rust table and the library's class data are generated from it (K3).
- What: a second engineer is paired on B7; B7 open at wave 2 end trips risk K27.
- Order and dependencies: after B31 and the B3 gate; before B6 so `set_shapes_N` is never regenerated.
- Verification: B3 gate identical and oracle run equal; no `set_shapes_` in the crate.

## B57 Class data record for the e-graph library

- What: class data is the record (shape, dtype, constant value); `merge` reports change for both sides per egg's `DidMerge` contract (Q12.4; waits for B49).
- Preserves: B7's shape rules and the coded shape fault of B26.
- Benefits: the record is B49's class data unchanged, so the swap (B52) moves no analysis logic.
- Caveats: not needed for M1; it lands before B49 starts.
- Fail points: a `merge` that reports no change after a change; noticed by egg's rebuild missing a parent update on the examples.
- Order and dependencies: after B7; before B49.
- Verification: B3 gate identical; a unit test merges two classes and asserts the reported change per side.

## B9 Extraction without cloning or mutating the e-graph

- What: constant-ness and the folded value live in class data computed by `make`; candidate building emits the constant node from class data, so no `DetectedConst` node enters the e-graph.
- Preserves: the folded value the allocator evaluates (`act_malloc.cc:74-78`).
- Benefits: removes the e-graph clone per pass (`pipeline.rs:234-240`) and the mutation (`slow.rs:24`).
- Caveats: adding the node in `Analysis::modify` would grow the e-graph during saturation and count against `NODE_LIMIT`.
- Fail points: a constant no longer folded; noticed by the B3 gate.
- Order and dependencies: after B7; before B10, since both touch `annotate_constants`.
- Verification: `extract` takes `&EGraph`; B3 gate identical.

## B6 One rewrite value per rule

- What: IR-to-IR rules are hand-written named Rust; IR-to-ISA rules are generated Rust with variables bound by name (Q4.4).
- What: a rule is a language-neutral record (name, pattern, guard tree over bound names, right-hand side) with no engine type in any signature; a Rust emitter writes it now, and B51 adds a C++ emitter (Q12.4; record form needed now, C++ emitter waits for B49).
- What: one shared re-match helper (`applier.rs:106-113,168-215`) serves every rule, since egg's `Subst` holds no payloads.
- What: the first two rules (`add_in_place` and one `dot` rule) are written as a spike of the data form with one generic applier; it replaces generated Rust only if it lands under about 300 lines with typed errors.
- Preserves: each condition's logic; `ConditionWriter` and `CheckedWriter`'s checked i64 operators.
- Benefits: a missing rule is a compile error, not a start-up panic (D6); conditions stop indexing positions; the number-keyed arms of `applier.rs:15-58` and the text files go.
- Caveats: rule 6 (`convert_s32 (convert_s8 ?x)`) is enforced only by `precond_6` (`ir2ir_rewrites.rs:514-526`); its condition must carry both payloads.
- Fail points: a condition bound to the wrong variable; noticed by B8 tests and the B3 gate.
- Order and dependencies: after B7 and B9; the spike decides the form before the third rule; before B8, B12, B33, B35.
- Verification: B3 gate identical; `ir2ir_rewrites.txt` and `ir2isa_rewrites.txt` deleted.

## B8 Name and test each IR-to-IR rewrite rule

- What: give each of the 10 IR-to-IR rules a name and two unit tests: it fires on a minimal e-graph and does not fire on a near miss.
- Preserves: each rule's pattern and condition.
- Benefits: rule 6's payload check and rule 0's target condition are pinned by tests.
- Caveats: tests assert on extracted terms, not on e-graph ids.
- Fail points: a test that pins internal ids; avoided by asserting terms.
- Order and dependencies: after B6.
- Verification: `cargo test` covers every IR-to-IR rule.

## B35 Overlapping IR-to-ISA rules

- What: overlapping rules stay alternatives the cost chooses; ties break on (cost, declaration index), the index stored on the ISA e-node and compared last (Q4.1).
- What: two instructions with identical pattern and conditions raise a generation warning.
- Preserves: the fixture's choice of the first-declared instruction (`tests/test_extraction.py:405-414`).
- Benefits: selection no longer depends on e-node order in a class (`fast.rs:24`), so a tie is defined (owner decision 16).
- Caveats: a name order would tie results to renames, so the index is the declaration position.
- Fail points: a golden that moves; noticed by the B3 gate.
- Order and dependencies: after B6 and B29.
- Verification: a tie test with two equal-cost instructions picks the first declared; duplicate warning test.

## B33 Target constants out of IR-to-IR rewrite 0

- What: rewrite 0 hard-codes tile size `16` and `Dtype::I32` (`ir2ir_rewrites.rs:20-27,100`) inside `generic/`; generate those values from the ISA specification's instructions, or drop the rule where no instruction needs it (Q4.10).
- Preserves: rule 0's result on the examples.
- Benefits: the `generic/` crate holds no target constant.
- Caveats: the generator must say which instruction the tile comes from.
- Fail points: an ISA specification with another tile size loses the split; noticed by a test with tile 32.
- Order and dependencies: after B6.
- Verification: no literal `16` tied to a `Dtype` in `generic/src/`; B3 gate identical.
