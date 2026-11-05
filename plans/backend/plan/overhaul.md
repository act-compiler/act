# Overhaul: target structure from today's code

The target keeps the four stages, egg, and the C++ allocator, and replaces the string boundaries between them.
Two pieces are replaced outright: the number-keyed rewrite dispatch (`applier.rs:15-58`) and both extractors (`fast.rs`, `slow.rs`).
Every structural step passes the B3 gate (identical chosen assembly and `--log` candidate files in B29's deterministic mode); every behavioral step passes the oracle run (Q4.2).

## Per module

| Module today | Action | Target | What is preserved |
| --- | --- | --- | --- |
| `generic/src/isel/initializer/` | reshape | `kernel/` reader that checks annotations (B5) | error messages, `add_<op>` builders, `shapes.rs` |
| `generic/src/ir/egraph.rs` | reshape | `egraph/`: typed payloads, shape and type in `make`, coded shape fault in `merge` (B31, B7, B26) | `TensorOp` variants, `merge` check |
| `generic/src/ir/pii.rs`, `metadata.rs`, `buffer.rs` | reshape | `extraction/pii.rs` | node fields, `Buffer` enum |
| `generic/src/isel/rewrites/*.txt`, `applier.rs`, `lib.rs` | replace | generated IR-to-ISA rules bound by name, one re-match helper (B6) | each condition's logic |
| `generic/src/isel/rewrites/ir2ir_rewrites.rs` | reshape | named IR-to-IR rules with guards (B8, B32) | each condition's logic |
| `generic/src/isel/extractor/slow.rs`, `fast.rs`, `utils.rs`, `lib.rs` | replace | `extraction/`: one k-best extractor; enumerator test-only (B37, B10) | (class, buffer) state, drop reasons |
| `generic/src/isel/extractor/inplace_fix.rs` | reshape | step of candidate building (B11) | copy rule and its two errors |
| `generic/src/pipeline.rs` | reshape | `main.rs` plus `search.rs`; cost failure drops the candidate (B29, B41) | CLI, limits, stop reasons, `--log` |
| `generic/src/malloc/cpp.rs` | reshape | typed FFI with kernel facts; C header by cbindgen (B2) | reason buffer |
| `generic/cpp/malloc/src/parser.cc` | replace | `create_instruction`, `hold`, `hold_tensors` kept (B4) | those three |
| `generic/cpp/malloc/src/constraint.cc` | reshape | pair relations at the touch point; no tie exemption (B30) | every other edge |
| `generic/cpp/malloc/src/` other files | keep | as is | all |
| `act_backend/spec_checks.py` | reshape | `checks/` after `validate()`, building the instruction model (B21, B14) | every backend-limit rejection and message |
| `act_backend/*_generator.py` | reshape | `emit/`, one per generated file, reading the model (B13) | generated code's behavior |
| `act_backend/template_loader.py`, `templates/` | replace | one template per fragment; rustfmt at generation (B12) | fragment text |
| `act_backend/expression_writer.py`, `operand_sizes.py` | keep | as is | all |

## Ordered steps

Two backend lanes: A is the Rust core, B is the FFI, allocator and generators.
Sizes are engineer-weeks (ew); Q1 is weeks 1-13.

| Step | Items | Lane | Kind | Size | Gate |
| --- | --- | --- | --- | --- | --- |
| 1 | B29, B3 | A | test infrastructure | 3 ew | two runs, two machines identical |
| 2 | B1, B28 | A | structural: moves and renames | 1 ew | B3 gate |
| 3 | B37 | A | structural: one extraction module, no re-extraction after saturation | 1 ew | B3 gate |
| 4 | B31 | A | structural: typed payloads | 3 ew | B3 gate |
| 5 | B7, B26 | A | behavioral: shapes in the analysis | 5 ew | B3 gate; oracle run; no coded shape fault on the examples |
| 6 | B9 | A | structural: no e-graph clone | 1 ew | B3 gate |
| 7 | B6, B8, B35 | A | structural: rewrite values; spike on the first two rules | 5 ew | B3 gate; text rules deleted |
| 8 | B25, B38 | B | C++ error returns; attribute domains from the width layer | 2 ew | `grep 'assert(false'` empty; negative-attribute and `u8` kernel tests |
| 9 | B13 | B | structural: instruction model | 4 ew | generated crates identical |
| 10 | B2, B4 | B | structural: typed FFI | 4 ew | B3 gate; no file read in C++ without `--log` |
| 11 | B39, B30, B40 | B | behavioral: pair relations in the allocator | 4 ew | oracle test per relation; equal offsets for a tie |
| 12 | B21, B24, B14, B27 | B | `validate()` first, checks and breaking rejections | 3 ew | `test_refusals.py`; new rejection tests |
| 12a | B41, B42 | A and B | behavioral: dropped candidates, input validation | 4 ew | unexpected-drop test; mutated-kernel test |
| 13 | B10, B34, B36, B11 | A | behavioral: one extractor; copy as a step of candidate building | 6 ew | equal or lower cost; oracle run |
| 14 | B32, B33 | A | behavioral: rule guards, target constants | 3 ew | equal or lower cost |
| 15 | B12, B5 | B | generators and reader | 3 ew | `tests/lint.sh`; `test_reader.py` |
| 16 | B15 to B20, B43 | A and B | coverage; input format readers | 12-17 ew | per-row tests flip; per-format equal assembly |
| 17 | B44, B45 | B | behavioral: derived attributes, then the assembler unit, last of v3 | 6 ew | round-trip corpus; neutrality test |

- Steps 1-7 (lane A) and 8-12 (lane B) run in parallel in Q1; steps 13-15 open Q2; step 16 fills Q2.
- Step 12 waits for U3; until U3 ships, the literal count check stays.
- Step 17 lands last as one unit and never ships in part (panel 10, Q10.4); the date moves, not the scope.
- Total about 69-74 ew: above two backend engineers for two quarters, so the roadmap places the excess (B41 to B45) with the other tracks.
- Every breaking item ships in act-backend 2.0 with the v3 release (owner decision 26).
