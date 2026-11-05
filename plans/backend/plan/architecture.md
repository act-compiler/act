# Backend architecture and module layout

The generated compiler runs four stages: read the kernel into an e-graph, saturate it with rewrites, enumerate candidate instruction graphs, then allocate and cost each candidate in C++.
Today each stage boundary is a file or a string: candidates cross to C++ as text (`generic/src/pipeline.rs:249-253`), rewrites are text parsed at start-up (`generic/src/isel/rewrites/lib.rs:6-19`).
These items set the target layout and replace the string boundaries with typed ones; later plan files build on them.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B1 | Target module layout and stage names | -- | S | Q1 |
| B2 | Typed candidate handoff from Rust to the C++ allocator | B1 | M | Q1 |
| B3 | End-to-end golden harness that runs without docker | T | M | Q1 |

## B1 Target module layout and stage names

- What: one crate layout `kernel/` (reader), `egraph/` (language, class data), `rewrites/` (IR-to-IR and IR-to-ISA rewrite rules), `extraction/` (pii graphs), `allocate/` (FFI to C++), `cost/`, `main.rs`; Python side `checks/`, `model/` (one selectable-instruction model), `emit/` (one module per generated file).
- Preserves: every file's code; only paths and module names change.
- Benefits: names match what each stage does; replaces `isel/initializer`, `isel/extractor`, `ir/pii.rs` and the "Phase 1 Module 3" log labels (`pipeline.rs:229`).
- Caveats: pure moves touch every `use` line; the generator copies `generic/` verbatim, so paths in `generator.py` and `build_rs_generator.py` move too.
- Fail points: a missed path breaks generation; noticed by the first generation test.
- Order and dependencies: first structural commit; no behavior change; T must agree on test file moves.
- Verification: `git diff --stat` shows renames only; golden kernels byte-identical (B3).

## B2 Typed candidate handoff from Rust to the C++ allocator

- What: pass each candidate to `act_malloc` as a flat `#[repr(C)]` node array (op id, payload ints, children, data model, off-chip offset) instead of writing `<n>.pii` and re-parsing it in `cpp/malloc/src/act_malloc.cc:45-66`.
- Preserves: `act_malloc`'s signature and reason buffer (`generic/src/malloc/cpp.rs:15-24`); `--log` candidate files, written from Rust.
- Benefits: deletes `PiiGraph`'s `Display` (`ir/pii.rs:75-93`) and most of `cpp/malloc/src/parser.cc`; one source of truth for the format; no temp-file I/O per candidate.
- Caveats: `--log` still wants candidates on disk; keep a debug printer on the Rust side only.
- Fail points: ABI mismatch between Rust and C++ structs; noticed as wrong allocations, so add a round-trip test that prints the struct from C++.
- Order and dependencies: after B1; enables B4; independent of O.
- Verification: goldens byte-identical; `parser.cc` reduced to `create_instruction`; no `.pii` read in C++.

## B3 End-to-end golden harness that runs without docker

- What: a test entry that generates, builds and compiles the examples using a local toolchain and OR-Tools path, sharing `tests/conftest.py` logic.
- Preserves: `tests/conftest.py` logic and the goldens.
- Benefits: every overhaul step below is verified by byte-identical goldens, the only safety net for refactors of this size.
- Caveats: needs OR-Tools and nlohmann/json installed; T track owns test infrastructure.
- Fail points: goldens depend on platform (`README.md` testing section); a mismatch from platform drift looks like a regression, so record platform with each run.
- Order and dependencies: before any behavioral item; owned jointly with T.
- Verification: CI and a local run both pass on unchanged code.
