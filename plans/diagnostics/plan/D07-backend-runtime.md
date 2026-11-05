# D7 Backend runtime diagnostics (Rust and C++)

- What: one `Diagnostic { code, object, line, message }` in Rust and a C++ twin; `Result<_, Diagnostic>` replaces `Result<_, String>` in the reader, extractor and bridges.
- What: the reader keeps a line number per kernel line; kernel faults print `kernel.hlo:12` (E9).
- What: replace every `assert(false)` and `debug_assert!` that stands for a check with a returned `I` code (E3, E4); keep asserts only for invariants and build the allocator with an explicit build type.
- What: a panic hook prints `error[IB000]` and exits 3; exit 1 for `K` and `X`, 2 for `U` (E10, E11).
- What: write the aggregated no-assembly error with the count and the first three codes on stderr; send drops and warnings to stderr too (E19).
- What: make constant messages whole sentences (E12).
- What: replace the `merge` panic (act-backend/act_backend/generic/src/ir/egraph.rs:403-409) with the shared fault `VF010` at selection; exit status 1.
- Benefits: closes E3, E4, E9 to E12, E19, E23; users see a code and a kernel line for every failure.
- Caveats: the largest item; the candidate handoff changes in B2, and the C++ parser is removed in B4, so do the C++ part after B4.
- Fail points: a diagnostic lost across the FFI reason buffer (`act_malloc.h:13-15`, 1024 bytes); noticed by a test with a long reason.
- Order and dependencies: D1, B4, B5, B25, B26.
- Verification: `tests/test_endings.py` asserts code, exit status and stream for each ending; a Release build of the allocator passes the same suite.
