# Parser consolidation

The backend has two parsers that do different jobs.
The Rust reader under `generic/src/isel/initializer/` (about 2,100 lines) reads the XLA-HLO kernel into the e-graph; its messages are pinned by `tests/test_reader.py`.
The C++ `cpp/malloc/src/parser.cc` (369 lines) reads back the candidate text Rust wrote (`act_malloc.cc:45-66`); its `hold` and `hold_tensors` (`parser.cc:146,335`) check that each value fits its storage, which is allocator logic.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B4 | Remove the C++ candidate parser | B2 | M | Q1 |
| B5 | Reader checks HLO annotations against the inferred shapes | B7 | S | Q2 |

## B4 Remove the C++ candidate parser

- What: `create_instruction` takes the B2 struct; delete `parse_line`, `ParsedLine` (`cpp/malloc/include/parser.h:16-20`) and `to_int` (`parser.cc:9-16`); move `hold` and `hold_tensors` beside `create_instruction`.
- Preserves: `create_instruction` (`parser.cc:173`), `hold`, `hold_tensors`, and the allocator classes they build.
- Benefits: one parser fewer; removes the `assert(false)` error paths of D5 in that file.
- Caveats: the generated `malloc_parser_cc_generator.py` shrinks to a table of op ids.
- Fail points: a payload field read in the wrong order; noticed by the B3 gate.
- Order and dependencies: after B2.
- Verification: B3 gate identical; `grep parse_line cpp/` empty; `hold_tensors` still called.

## B5 Reader checks annotations against the inferred shapes

- What: after B7, the reader stops setting class data by hand and checks each HLO shape annotation against the shape `make` inferred, raising a named error on a mismatch.
- Preserves: every reader message pinned by `tests/test_reader.py`.
- Benefits: `nodes.rs` (668 lines) loses its 17 `set_analysis_data` calls.
- Caveats: a new mismatch message needs a test of its own.
- Fail points: an accepted kernel newly rejected; noticed by `tests/test_reader.py` and the B3 gate.
- Order and dependencies: after B7.
- Verification: `tests/test_reader.py` passes unchanged.
