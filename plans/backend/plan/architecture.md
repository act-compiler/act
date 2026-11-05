# Backend architecture and module layout

The generated compiler runs four stages: read the kernel into an e-graph, saturate it with rewrites, enumerate candidate instruction graphs, then allocate and cost each candidate in C++.
Each stage boundary is a file or a string: candidates cross to C++ as text (`generic/src/pipeline.rs:249-253`), kernel facts as `metadata.json` (`pipeline.rs:457-469`, `act_malloc.cc:147-153`), rewrites as text parsed at start-up (`generic/src/isel/rewrites/lib.rs:6-19`).
These items set the target layout, the test gate, and the typed candidate boundary.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B1 | Target module layout and stage names | B29 | S | Q1 |
| B2 | Typed candidate and kernel-fact handoff to the C++ allocator | B1, B13 | M | Q1 |
| B3 | End-to-end gate: goldens, chosen assembly, candidate logs, oracle run | B29, T | M | Q1 |

## B1 Target module layout and stage names

- What: one crate layout `kernel/` (reader), `egraph/` (language, class data), `rewrites/` (IR-to-IR and IR-to-ISA rewrite rules), `extraction/` (pii graphs), `allocate/` (FFI to C++), `cost/`, `main.rs`; Python side `checks/`, `model/`, `emit/`.
- Preserves: every file's code; only paths and module names change.
- Benefits: names match what each stage does; replaces `isel/initializer`, `isel/extractor`, `ir/pii.rs` and the "Phase 1 Module 3" log labels (`pipeline.rs:229`).
- Caveats: pure moves touch every `use` line; the generator copies `generic/` verbatim, so paths in `generator.py` and `build_rs_generator.py` move too.
- Fail points: a missed path breaks generation; noticed by the first generation test.
- Order and dependencies: right after B29, before the lanes split; with B28 so each file moves and renames once.
- Verification: `git diff --stat` shows renames only; B3 gate identical.

## B2 Typed candidate and kernel-fact handoff

- What: pass each candidate to `act_malloc` as a flat `#[repr(C)]` node array (op id, typed payload, children, buffer, hbm offset) plus the kernel facts in the same call, instead of `<n>.pii` and `metadata.json`.
- What: the struct is emitted for Rust and C++ from the B13 model, so one schema exists.
- Preserves: `act_malloc`'s reason buffer (`generic/src/malloc/cpp.rs:15-24`); `--log` candidate files, written from Rust in today's exact line format.
- Benefits: deletes `PiiGraph`'s `Display` round trip (`ir/pii.rs:75-93`) and the `metadata.json` read; no temp-file I/O per candidate.
- Caveats: `tests/test_extraction.py:109,135,496` read `--log` candidate files, so the Rust printer keeps their line format.
- Fail points: ABI mismatch between Rust and C++ structs; noticed as wrong allocations, so add a round-trip test that prints the struct from C++.
- Order and dependencies: after B1 and B13; enables B4; payload fields from B31.
- Verification: B3 gate identical; no file read in C++ without `--log`.

## B3 End-to-end gate

- What: a test entry that generates, builds and compiles the examples with a local toolchain, in B29's deterministic mode.
- What: a structural step passes when the chosen assembly, the `--log` candidate files and the goldens are identical.
- What: a behavioral step passes an oracle-differential run over the specs of `tests/test_extraction.py` and `tests/test_matching.py`.
- Preserves: `tests/conftest.py` logic and the goldens.
- Benefits: goldens are simulated outputs (`README.md:109`), so equal goldens do not prove an equal selection; the assembly and candidate checks do.
- Caveats: needs OR-Tools and nlohmann/json installed; T owns test infrastructure.
- Fail points: platform drift looks like a regression; record platform with each run.
- Order and dependencies: with B29, before any other item; owned jointly with T.
- Verification: two runs on two machines give identical assembly and candidate files on unchanged code.
