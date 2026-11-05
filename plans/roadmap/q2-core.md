# Quarter 2, core tracks

Plan items scheduled for quarter 2 in the core tracks, by track.

## language

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| L3 | Normative shape and element type rules per operation | L2, O6, O7 | L | R62 |
| L8 | Spec consistency checks in CI | L1 | M | -- |
| L9 | Worked examples per section | L3 | M | -- |
| L12 | Shape and element type checker | L3, O7 | L | R62 |
| L15 | Rename confusing internal names | L25 | M | -- |
| L16 | Range-check `constant` literals by element type | L3 | S | -- |
| L17 | Low-precision and block-scaled element types, `scaled_dot` | L2, L3 | M | -- |
| L18 | Write masks on output regions | L24, L3 | M | -- |
| L19 | Strided regions | L5 | M | -- |
| L20 | `convolution` with window attributes | L3 | M | -- |
| L21 | Rounding and saturation attributes on `convert` | L3 | S | -- |
| L22 | Sub-computations for `reduce`, `scatter`, `sort`, `map` | L3 | L | -- |
| L23 | Indirect regions: starts read from a data model | L19 | M | -- |
| L28 | Banned-synonym lint | L26 | S | -- |

## backend

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| B5 | One kernel reader: tokenizer, line parser, e-graph builder | B1, B3 | M | -- |
| B11 | In-place copies as a candidate rewrite, not a fix-up | B10 | M | -- |
| B14 | Checks as one pass producing that model | B13 | M | -- |
| B15 | Keep `docs/coverage.md` in sync by a test | B14, L7 | S | -- |
| B16 | State variables in the backend | B13, L1 | L | -- |
| B17 | Operations beyond the 18 | B6, B7 | M | -- |
| B18 | Element types `s2`, `s4`, `u2`, `u4`, `f8e4m3fn`, `f8e5m2` | B7, L17 | S | -- |
| B19 | Several output regions | B11, B2 | L | -- |
| B20 | `slice` stride and `constant` expressions | B6 | S | -- |
| B23 | Check shape and element-type agreement of semantics lines (D3) | B13, L12, L3 | M | -- |
| B26 | Extraction errors as dropped candidates, unset shape as `Option` (D7, D8) | B7 | S | -- |

## oracle

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| O3 | Replace string templates by emitter calls | O2, O4 | L | R30 |
| O6 | Spec errors raise `ValueError`; S8.3 table from the front end | L3 | S | R62 |
| O7 | Move `shapes.py` rules to the front end, keep call-time evaluation | B23, L12, O6 | L | R31, R62 |
| O8 | Round float literals once, from the decimal | L16 | S | -- |
| O9 | `triangular_solve` at `f16`: decide and narrow the widening | L3 | S | -- |
| O12 | Scope jax configuration to the oracle | -- | M | R32 |
| O13 | NumPy reference interpreter for differential tests | O2 | L | -- |

## tests

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| T1 | Tag every surviving test with its spec section | S1, S10 | M | -- |
| T4 | Delete tests that pin generated text or internal names | T11, T13, T14 | S | -- |
| T5 | Corpus format, location, and naming | S1, S9 | S | -- |
| T6 | Front-end verdict runner in taidl | T5 | M | -- |
| T7 | Seed the corpus from existing front-end tests (S2 to S7) | T1, T5, T6 | L | -- |
| T8 | Operations corpus with reference values (S8) | S8, T5 | L | -- |
| T9 | Property-based tests for shapes, element types, and names | T22, T6, T8 | M | -- |
| T10 | Real ISA descriptions as the end-to-end list | T5 | S | -- |
| T11 | Expectations table per generator, coverage.md generated from it | T5 | M | -- |
| T12 | Expectation runner: supported or rejected with the documented error | T11, T6 | M | -- |
| T13 | Differential tests oracle vs backend on the supported intersection | T12, T3, T8 | L | -- |
| T14 | One shared end-to-end suite in act | T10, T3 | M | -- |
| T15 | Golden policy: one regenerate, one record, one tolerance | T14 | S | -- |
| T16 | CI wiring: corpus travels with the pinned taidl release | T12, T5 | S | -- |
| T17 | Split `act-oracle/tests/test_kernel.py` by theme | T1 | M | -- |
| T18 | Final deletion pass of PR-defense tests | T13, T14, T20, T4 | M | -- |
| T19 | Decision: corpus lives in taidl, consumed through the pin | T5 | S | -- |
| T20 | Decision: impl tests survive with a header, PR-defense tests do not | T18 | S | -- |
| T21 | Decision: expectations table is the source, coverage.md is rendered | T11 | S | -- |

## act

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| A6 | Comment pass on docker/ scripts | -- | S | -- |
| A11 | Replace the three copies by one tutorial plus per-event pages | A10, A9 | L | R50, R54 |
| A13 | Make exercise 4 runnable like exercises 1-3 | A11 | S | -- |
| A17 | Check sibling pins and dependency floors against act | A15 | M | -- |
| A18 | Add a scheduled compatibility build | A16 | M | -- |
| A19 | Cover state variables and constraints in the end-to-end test | -- | M | -- |
| A21 | Run the current tutorial's solutions in CI | A10, A11 | M | -- |
| A22 | Decide the tutorials' lint scope | A11 | S | -- |
