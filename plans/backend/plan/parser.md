# Parser consolidation

The backend has two parsers that do different jobs.
The Rust reader under `generic/src/isel/initializer/` (`parser.rs`, `line.rs`, `expression.rs`, `shapes.rs`, `nodes.rs`, `lib.rs`, about 2,100 lines) reads the XLA-HLO kernel into the e-graph.
The C++ `cpp/malloc/src/parser.cc` (369 lines) reads back the candidate text that Rust itself wrote (`act_malloc.cc:45-66`), so it is an internal round trip, not a second input language.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B4 | Remove the C++ candidate parser | B2 | M | Q1 |
| B5 | One kernel reader: tokenizer, line parser, e-graph builder | B1, B3 | M | Q2 |

## B4 Remove the C++ candidate parser

- What: `create_instruction` takes the B2 struct; delete `parse_line`, `ParsedLine` (`cpp/malloc/include/parser.h:16-20`) and `to_int` (`parser.cc:9-16`).
- Preserves: `create_instruction` (`cpp/malloc/src/parser.cc:173`) and the allocator classes it builds.
- Benefits: one parser fewer; removes the `assert(false)` error paths of D5 in that file.
- Caveats: the generated `malloc_parser_cc_generator.py` emits per-instruction parsing; it shrinks to a table of op ids.
- Fail points: a payload field read in the wrong order; noticed by golden mismatch.
- Order and dependencies: after B2.
- Verification: goldens byte-identical; `grep parse_line cpp/` empty.

## B5 One kernel reader

- What: split the reader into a tokenizer, a line parser producing a typed `KernelLine`, and an e-graph builder; one `add_<op>` per supported op stays in the builder (`initializer/nodes.rs`).
- Preserves: the `add_<op>` builders of `initializer/nodes.rs`, `shapes.rs` as is, every error message.
- Benefits: each layer is testable alone; the 668-line `nodes.rs` and 471-line `expression.rs` lose their mixed parsing and building.
- Caveats: the reader's error messages are pinned by `tests/test_reader.py`; keep them word for word.
- Fail points: an accepted kernel newly rejected; noticed by `tests/test_reader.py` and goldens.
- Order and dependencies: after B1 and B3; independent of B2.
- Verification: `tests/test_reader.py` passes unchanged; goldens byte-identical.
