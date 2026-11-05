# Overhaul: target structure from today's code

The target keeps the four stages and the C++ allocator and replaces only the string boundaries between them.
No module is rewritten from scratch; the replaced pieces keep their role, and the decision record gives each replacement's reason.
Each step leaves the golden kernels byte-identical unless it says otherwise.

## Per module

| Module today | Action | Target | What is preserved |
| --- | --- | --- | --- |
| `generic/src/isel/initializer/` | reshape | `kernel/` reader: tokenizer, line parser, builder | error messages, `add_<op>` builders, `shapes.rs` as is |
| `generic/src/ir/egraph.rs` | reshape | `egraph/`: language and class data with shape rules in `make` | `TensorOp` variants, `merge` check |
| `generic/src/ir/pii.rs`, `metadata.rs`, `buffer.rs` | reshape | `extraction/pii.rs` | node fields, data model enum |
| `generic/src/isel/rewrites/*.txt`, `applier.rs`, `lib.rs` | replace | one Rust value per rewrite | every pattern and condition, verbatim |
| `generic/src/isel/rewrites/ir2ir_rewrites.rs` | reshape | named IR-to-IR rewrite rules | each condition's logic |
| `generic/src/isel/extractor/slow.rs`, `fast.rs`, `utils.rs`, `lib.rs` | merge | `extraction/`: shared core, fast and slow extractor | candidate order and limits |
| `generic/src/isel/extractor/inplace_fix.rs` | reshape | step of candidate building | copy rule and its two errors |
| `generic/src/pipeline.rs` | reshape | `main.rs` plus `search.rs` | CLI, limits, stop reasons, `--log` |
| `generic/src/malloc/cpp.rs` | reshape | typed FFI | reason buffer |
| `generic/cpp/malloc/src/parser.cc` | replace | struct reader, `create_instruction` kept | `create_instruction` |
| `generic/cpp/malloc/src/` other files | keep | as is | all |
| `act_backend/spec_checks.py` | reshape | `checks/` building the instruction model | every rejection and message |
| `act_backend/*_generator.py` | reshape | `emit/`, one per generated file, reading the model | generated code's behavior |
| `act_backend/template_loader.py`, `templates/` | replace | one template per fragment, formatter run after | fragment text |
| `act_backend/expression_writer.py`, `operand_sizes.py` | keep | as is | all |

## Ordered steps

| Step | Items | Kind | Size | Risk | Verify |
| --- | --- | --- | --- | --- | --- |
| 1 | B3 | test infrastructure | M | low | CI and local run agree on unchanged code |
| 2 | B1, B28 | structural: moves and renames | S | low | renames only in diff; goldens identical |
| 3 | B13 | structural: one instruction model | M | low | generated crates identical |
| 4 | B6, B8 | structural: typed rewrites | M | medium | goldens identical; text rewrite files gone |
| 5 | B2, B4, B25 | structural: typed FFI | M | medium | goldens identical; no `.pii` read in C++ |
| 6 | B10 | structural: one extractor module | M | medium | same candidate counts in `--verbose` |
| 7 | B12, B14 | structural: templates and checks | M | medium | `tests/lint.sh`, `test_refusals.py` pass |
| 8 | B7, B9, B26 | behavioral: shapes in the analysis | L | high | goldens identical; `merge` silent on examples |
| 9 | B21 to B24, B27 | behavioral, breaking | S each | medium | new rejection tests |
| 10 | B11, B15 to B20 | behavioral: coverage | M to L | high | per-row tests flip |
