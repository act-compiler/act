# Overhaul: target structure from today's code

The target keeps the four stages, egg, and the C++ allocator, and replaces the string boundaries between them.
Two pieces are replaced outright: the number-keyed rewrite dispatch (`applier.rs:15-58`) and both extractors (`fast.rs`, `slow.rs`).
Every structural step passes the B3 gate (identical chosen assembly, candidate files and goldens in B29's deterministic mode); every behavioral step passes the oracle run.

## Per module

| Module today | Action | Target | What is preserved |
| --- | --- | --- | --- |
| `generic/src/isel/initializer/` | reshape | `kernel/` reader that checks annotations (B5) | error messages, `add_<op>` builders, `shapes.rs` |
| `generic/src/ir/egraph.rs` | reshape | `egraph/`: typed payloads, shape and type in `make` | `TensorOp` variants, `merge` check |
| `generic/src/ir/pii.rs`, `metadata.rs`, `buffer.rs` | reshape | `extraction/pii.rs` | node fields, `Buffer` enum |
| `generic/src/isel/rewrites/*.txt`, `applier.rs`, `lib.rs` | replace | one rewrite value per rule (B6) | each condition's logic |
| `generic/src/isel/rewrites/ir2ir_rewrites.rs` | reshape | named IR-to-IR rules in phases (B8, B32) | each condition's logic |
| `generic/src/isel/extractor/slow.rs`, `fast.rs`, `utils.rs`, `lib.rs` | replace | `extraction/`: one cost-guided extractor (B10) | (class, buffer) state, drop reasons |
| `generic/src/isel/extractor/inplace_fix.rs` | reshape | step of candidate building (B11) | copy rule and its two errors |
| `generic/src/pipeline.rs` | reshape | `main.rs` plus `search.rs` (B29) | CLI, limits, stop reasons, `--log` |
| `generic/src/malloc/cpp.rs` | reshape | typed FFI with kernel facts (B2) | reason buffer |
| `generic/cpp/malloc/src/parser.cc` | replace | `create_instruction`, `hold`, `hold_tensors` kept (B4) | those three |
| `generic/cpp/malloc/src/constraint.cc` | reshape | in-place equality constraint (B30) | every other edge |
| `generic/cpp/malloc/src/` other files | keep | as is | all |
| `act_backend/spec_checks.py` | reshape | `checks/` building the instruction model (B14) | every rejection and message |
| `act_backend/*_generator.py` | reshape | `emit/`, one per generated file, reading the model (B13) | generated code's behavior |
| `act_backend/template_loader.py`, `templates/` | replace | one template per fragment (B12) | fragment text |
| `act_backend/expression_writer.py`, `operand_sizes.py` | keep | as is | all |

## Ordered steps

Two backend lanes: A is the Rust core, B is the FFI, allocator and generators.
Sizes are engineer-weeks (ew); Q1 is weeks 1-13.

| Step | Items | Lane | Kind | Size | Gate |
| --- | --- | --- | --- | --- | --- |
| 1 | B29, B3 | A | test infrastructure | 3 ew | two runs, two machines identical |
| 2 | B1, B28 | A | structural: moves and renames | 1 ew | B3 gate |
| 3 | B31 | A | structural: typed payloads | 3 ew | B3 gate |
| 4 | B7, B26 | A | behavioral: shapes in the analysis | 5 ew | B3 gate; oracle run; `merge` silent |
| 5 | B9 | A | structural: no e-graph clone | 1 ew | B3 gate |
| 6 | B6, B8 | A | structural: rewrite values | 4 ew | B3 gate; text rules deleted |
| 7 | B13 | B | structural: instruction model | 4 ew | generated crates identical |
| 8 | B2, B4, B25 | B | structural: typed FFI | 4 ew | B3 gate; no file read in C++ without `--log` |
| 9 | B30 | B | behavioral: in-place storage reuse | 2 ew | oracle test; equal offsets |
| 10 | B14, B21, B22, B24, B27 | B | checks and breaking rejections | 3 ew | `test_refusals.py`; new rejection tests |
| 11 | B10, B11, B34 | A | behavioral: one extractor | 5 ew | equal or lower cost; oracle run |
| 12 | B32, B33 | A | behavioral: rule phases, target constants | 3 ew | equal or lower cost |
| 13 | B12, B5, B23 | B | generators and reader | 4 ew | `tests/lint.sh`; `test_reader.py` |
| 14 | B15 to B20 | A and B | coverage | 10-14 ew | per-row tests flip |

- Steps 1-6 (lane A) and 7-10 (lane B) run in parallel in Q1; steps 11-13 open Q2; step 14 fills Q2.
- Total about 52-56 ew: two backend engineers for two quarters.
