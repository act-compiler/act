# Review of the backend as it stands

The backend works end to end and its rejections are careful; its weakness is the plumbing between stages, not the stages' ideas.
Verdicts: keep (leave as is), reshape (same role, restructured), replace (role kept, code rewritten).
Evidence is file:line from the code; nothing was run.

## What is good and should be kept

- Every rejection is a `ValueError` naming the instruction, raised before anything is written (`act_backend/generator.py:124-131`).
- 64-bit overflow is handled exactly: integer-only arithmetic is evaluated and checked (`act_backend/spec_checks.py:147-166`), and sizes that may overflow at selection get checked Rust (`act_backend/operand_sizes.py:1-10`).
- Floor division matches the oracle in all three places it is written: Python-emitted Rust and C++ (`act_backend/expression_writer.py:10,74`) and the solver (`cpp/malloc/include/solver.h:26-63`).
- Standard egg use for saturation: `Runner` with node, iteration and time limits (`generic/src/pipeline.rs:491-495`).
- The analysis `merge` turns a shape-changing rewrite into a loud internal error, not a wrong result (`generic/src/ir/egraph.rs:403-410`); a panic is not a defined error under owner decision 40, so B26 makes it the shared coded shape fault and B41 drops the candidate (Q3.6).
- The allocator drops a candidate with a reason instead of failing the compile (its asserts and the cost-model exit are the exceptions, D17 and D20) (`cpp/malloc/src/act_malloc.cc:52-64`, `generic/src/malloc/cpp.rs:12`).
- The cost model is a one-function C ABI, a clean extension point (`README.md` cost section).
- A large test suite pins behavior: `tests/test_matching.py` (1,479 lines), `test_refusals.py`, `test_reader.py`, goldens.

## Verdict per component

| Component | Verdict | Good | Weak |
| --- | --- | --- | --- |
| `generator.py` | keep | one entry, checks before writes (`:119-131`) | drops semantics-less instructions unchecked (D4) |
| `spec_checks.py` | reshape | precise named rejections | imports emitter internals (`:9-18`), re-runs an emitter visitor (`:213-215`) |
| `isel_ir2isa_rewrites_rs_generator.py` | reshape | exact payload comparison (`:250-273`) | 795 lines, half is layout choice (`:369-399`) |
| `isel_ir2isa_rewrites_txt_generator.py` | replace | small op table (`:11-30`) | text patterns parsed again at run time (D6) |
| `isel_applier_rs_generator.py` | replace | small | emits number-keyed `match` arms (D6) |
| `ir_egraph_rs_generator.py`, `ir_buffer_rs_generator.py` | keep | generate the language enum per spec | layout twins |
| `malloc_*_generator.py` (4) | reshape | per-instruction allocator classes | `malloc_parser_cc_generator.py` serves the round-trip parser (D10) |
| `expression_writer.py`, `operand_sizes.py` | keep | one writer for Rust and C++ | none found |
| `semantics_visitor.py`, `instruction_info.py` | reshape | shared visitor | several visitors re-walk the same trees |
| `template_loader.py` and 79 templates | replace | output passes the formatters | twin layouts per width (`template_loader.py:14`) |
| `initializer/parser.rs` | reshape | names errors | mixes tokenizing and building |
| `initializer/nodes.rs` | reshape | one `add_<op>` per op | 668 lines of building plus checks |
| `initializer/expression.rs` | reshape | | 471 lines, parsing mixed in |
| `initializer/shapes.rs` | keep | small, focused | |
| `ir/egraph.rs` | reshape | standard `Analysis` | `make` computes no shape (`:396-401`); empty shape means unset (D8) |
| `rewrites/ir2ir_rewrites.rs` | reshape | correct, hand-checked | numbered functions, no names (D6) |
| `rewrites/applier.rs` | replace | custom applier works | string dispatch (`:15-58`) |
| `extractor/slow.rs`, `fast.rs` | merge | two useful strategies | duplicated constant pass (`slow.rs:21-23`, `fast.rs:19-21`) |
| `extractor/inplace_fix.rs` | reshape | clear errors (`:6-9`) | post-hoc fix-up |
| `pipeline.rs` | reshape | clear stop reasons | 568 lines; clones the e-graph per pass (D11); stage-number logs |
| `cpp/malloc/src/parser.cc` | replace | | re-parses Rust's own text (D10), `assert` errors (D5) |
| C++ allocator (`constraint.cc`, `scheduler.cc`, `emitter.cc`, `act_malloc.cc`) | keep | OR-Tools model, reasons returned | `assert` error paths (D5) |
