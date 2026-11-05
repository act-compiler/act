# Wave 2

Normative rules, the typed contract, the width layer, and the backend shape chain; 42 items, weight 78.
Wave 2 ends at M1.

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| L4 | Check generators against the stated reduce identities and `constant` fill | L2 | S | -- |
| L15 | Rename confusing internal names | L25 | S | -- |
| L16 | `check` reports a range or type error at the declaring call's site | L3 | S | -- |
| L17 | Element types `f4e2m1fn` and `f8e8m0fnu` | L2, L3 | S | -- |
| L26 | Apply the glossary to the taidl docs and error messages | L25 | M | -- |
| L31 | Float accuracy bounds for `dot`, `reduce_add`, `reduce_multiply` | L2 | S | -- |
| L32 | Defined-behavior norms: pre-state issue, unwritten reads, numerics, fault guarantees | L1, L2 | M | -- |
| L46 | Declared initial value per buffer: `initial_value` on `acc.buffer` | L16, L32 | S | O26 and B36 read it |
| L34 | Rename "refus" to "reject" in messages, docstrings, READMEs, and identifiers | L30 | S | -- |
| L27 | Apply the glossary to B, O, and A docs | L25 | M | later item; scheduled because B27 and O15 depend on it (Q8.1) |
| B24 | Reject instructions with no semantics, exclude state-only ones (D4) | L5 | S | -- |
| B27 | Glossary and owner vocabulary in messages, docs and Python names | L25, L27 | S | -- |
| B28 | Rust and C++ identifier and log renames | B1 | S | -- |
| B31 | Typed payloads in `TensorOp` | B1, B3 | M | -- |
| B2 | Typed candidate and kernel-fact handoff to the C++ allocator | B1, B25, B31 | M | -- |
| B7 | Shape and element type inferred in `Analysis::make` | B3, B31 | L | names its rule source; agreement test with the taidl module (Q8.2); open at wave 2 end trips K27 |
| B37 | One extraction module and no re-extraction after saturation | B1, B29 | S | -- |
| B39 | Pair-relation record in the typed handoff | B2, L35 | S | -- |
| O6 | Spec errors raise `ValueError`; S8.3 table from the front end | L3 | S | -- |
| O9 | `triangular_solve` at `f16`: measure, then widen only where needed | L3 | S | -- |
| O11 | Fallback only if O19 fails: keep `while` with an equivalence test; reject wrapping starts | O19 | S | -- |
| O14 | Fold `docs/operations.md` into `docs/coverage.md` and README | L1, L7 | S | -- |
| O17 | Keep `docs/coverage.md` true by test | O14 | S | -- |
| O22 | Enforce the numerics norm table: a guard per row and a golden per row | L6 | M | -- |
| O25 | Expressions read the pre-state; updates commit with the outputs | L5 | S | -- |
| O10 | Overlap constraints and in-place marks: check the list before any load, value from the pre-state | L5, O25 | S | reinstated (Q8.6) |
| O27 | Widen narrow float `dot` and `reduce` to `f32` in the module | L6 | S | -- |
| O30 | Overlap test matrix: declarations by placements | O10 | S | -- |
| T8 | Operations corpus with reference values (S8) | L3, T5 | L | -- |
| T10 | Real ISA specifications as the end-to-end list | T5 | S | -- |
| T14 | One shared end-to-end suite in act | B29, T10, T3 | M | -- |
| T17 | Split `act-oracle/tests/test_kernel.py` by theme | T1 | M | -- |
| T19 | Decision: corpus lives in taidl, consumed through the pin | T5 | S | -- |
| A1 | Apply glossary terms and plain words in README.md | A8, L26 | S | -- |
| A7 | ASCII and one-sentence-per-line gate for text files | A11 | S | -- |
| C2 | Strided regions: evidence that admits L19 | C11 | S | -- |
| P7 | Executable examples per spec row | P1, P2 | M | -- |
| U2 | Typed front-end contract with a version | L15 | M | -- |
| L37 | Width layer: a type per attribute, `s64` by default | L1, U2 | M | width layer first (Q10.4) |
| D1 | Registry, classes and message format | L1, L25 | M | -- |
| T11 | Expectations table per generator, coverage.md generated from it | D1, T5 | M | -- |
| D14 | Fault table and run-time norm | L5, L6 | S | -- |
