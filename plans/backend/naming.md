# act-backend naming todo

Names and words to change, aligned to the glossary of the TAIDL language specification.
None is applied: the scope forbids code edits, so every row is a plan item (B27, B28).
`new` is a proposal; rows marked `?` need an owner call.

| old | new | where | applied? |
| --- | --- | --- | --- |
| `honoured` | `is_supported_constraint` | `act_backend/spec_checks.py:331,355` | no |
| data model (one declared memory) | buffer | `act_backend/spec_checks.py:297-305` messages, docstrings | no |
| refuse, refusal, `start_refusal`, `division_refusal` | reject, rejection, `start_rejection`, `division_rejection` | `act_backend/spec_checks.py:1,243,258` and docstrings; `tests/test_refusals.py` | no |
| address attribute | addressing attribute | `act_backend/spec_checks.py:204,278` | no |
| machine state | state / state update | `act_backend/spec_checks.py:371,376` | no |
| `beyond_allocator` | `reads_address_or_state` | `act_backend/spec_checks.py:277` | no |
| operand (for a region) | region | `act_backend/spec_checks.py:308-325` messages "its operand on" | no |
| `TensorInfo` | `ClassData` | `generic/src/ir/egraph.rs:371`, all users | no |
| `MetaData`, `MetaDataInfo` | `KernelRegions`, `KernelRegion` ? | `generic/src/ir/metadata.rs:7,18-19` | no |
| `pii` undefined in code | keep `PiiNode`, `PiiGraph`; define pii once as partially instantiated instruction | `generic/src/ir/pii.rs:10,20`, extractors, `pipeline.rs` | no |
| `Buffer` enum | keep: it names a buffer (owner decision 19) | `generic/src/ir/buffer.rs` | no |
| `DetectedConst` | `FoldedConstant` | `generic/src/ir/egraph.rs:398`, extractors | no |
| `EggSymbolTable` | `ValueIds` | `generic/src/isel/initializer/lib.rs:23` | no |
| `HLOExpr` | `KernelExpr` | `generic/src/isel/initializer/expression.rs` | no |
| `slow`, `fast` extractors | gone: one k-best extractor; the enumerator kept test-only is named `enumerate` (B10) | `generic/src/isel/extractor/slow.rs:13`, `fast.rs:12` | no |
| `inplace_fix`, `fix_inplace` | `copy_overwritten`, `copy_overwritten_inputs` | `generic/src/isel/extractor/inplace_fix.rs:9` | no |
| `precond_N`, `metadata_N`, `set_shapes_N` | named per rewrite, `condition`, `result_types`, `result_shapes` | `generic/src/isel/rewrites/ir2ir_rewrites.rs`, `applier.rs:15-58` | no |
| `ir2ir`, `ir2isa` in prose | IR-to-IR rewrite rule, IR-to-ISA rewrite rule | file names under `generic/src/isel/rewrites/`, generator module names | no |
| `Info:` log prefix | no prefix | `generic/src/pipeline.rs:521,537,548` | no |
| comments `// Process input file`, `// Process output file`, `// Process log directory`, `// Start processing the input file` | `// Read the kernel`, `// Check the output path`, `// Make the log directory`, `// Build the e-graph` | `generic/src/pipeline.rs:300,326,352,435` | no |
| C++ lambdas named `panic`, comment `// Panic with an error message` | `fail` returning an error | `cpp/malloc/src/parser.cc:185-190`, `constraint.cc:112-117` | no |
| `// trim` | `// Strip surrounding whitespace` | `cpp/malloc/src/act_malloc.cc:46` | no |
| `generate_helpers` | `generate_rewrite_functions` | `act_backend/isel_ir2isa_rewrites_rs_generator.py:762` | no |
| `get_instruction_metadata`, `InstructionMetadata`, `instruction_info.py` | `selectable_instructions`, `SelectableInstruction`, `selectable.py` | `act_backend/generator.py:49`, `act_backend/instruction_info.py` | no |
| `get_next_variable`, `substitute_variables`, `variable_counter` | `fresh_variable`, `rename_variables`, `next_variable` | `act_backend/isel_ir2isa_rewrites_txt_generator.py:39-54` | no |
| `COMPARED` | `PAYLOAD_OPERATIONS` | `act_backend/isel_ir2isa_rewrites_rs_generator.py:41` | no |
| `TWINS` | gone with the layout twins (B12) | `act_backend/template_loader.py:14` | no |
| spec (the user's file) | ISA specification (owner decision 32) | `act_backend/` messages and docstrings, `README.md` (act-backend) | no |
| TAIDL spec (the language document) | TAIDL language specification (owner decision 32) | `docs/coverage.md`, `README.md` (act-backend) | no |
| dtype (in messages and docs) | element type | Rust `Dtype` may stay as a type name | no |
| refuses | rejects | `README.md:48` (act-backend) | no |
| machine code, assembly | compiled kernel | `README.md:16` and the usage section (act-backend) | no |
| seamless support | name what works, such as runs inside XLA | `README.md:17` (act-backend) | no |
| kernel (unqualified) | input kernel or compiled kernel | `README.md`, `generic/src/pipeline.rs` messages | no |
