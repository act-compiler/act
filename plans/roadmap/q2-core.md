# Quarter 2, core tracks

Plan items scheduled for quarter 2 in the core tracks, by track.

## language

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| L3 | Normative shape and element type rules per operation | L2, O6, O7 | L | R62 |
| L8 | Spec consistency checks in CI | L1 | S | -- |
| L12 | Shape and element type checker | L3 | L | R62 |
| L15 | Rename confusing internal names | L25 | S | -- |
| L16 | Range and type checks at the declaring call | L3 | S | -- |
| L17 | Low-precision and block-scaled element types, `scaled_dot` | L2, L3 | M | -- |

## backend

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| B5 | Reader checks HLO annotations against the inferred shapes | B7 | S | -- |
| B10 | One cost-guided extractor keeping the best few candidates | B9 | M | -- |
| B11 | In-place copies as a step of candidate building | B10 | M | -- |
| B12 | Emit plain code without layout twins | B6 | S | R19 |
| B16 | State in the backend | B13, L1 | L | -- |
| B17 | Operations beyond the 18, `concatenate` of any arity | B31, B6, B7 | M | -- |
| B18 | Element types `s2`, `s4`, `u2`, `u4`, `f8e4m3fn`, `f8e5m2` | B7, L17 | S | -- |
| B19 | Several output regions | B11, B2 | L | -- |
| B20 | `slice` stride, `constant` expressions, float literal range | B6 | S | -- |
| B23 | Check shape and element-type agreement of semantics lines (D3) | B13, B7, L12, L3 | M | -- |
| B32 | IR-to-IR rule phases | B29, B8 | M | -- |
| B33 | Target constants out of IR-to-IR rewrite 0 | B6 | S | -- |
| B34 | Search budget measurement and candidate deduplication | B10, B29 | S | -- |

## oracle

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| O1 | Runtime code out of `API_FILE` into `runtime/`; split what remains of `utils.py` | O20 | M | R28 |
| O2 | One instruction record from the visitor, consumed by checks and emission | O1, O5 | M | R30 |
| O3 | Emitter calls in generic form instead of string templates; one word table | O2 | M | R30 |
| O6 | Spec errors raise `ValueError`; S8.3 table from the front end | L3 | S | R62 |
| O7 | Move static checks to the front end: shapes, region values, constant ranges, `compare` type | B23, L12, O6 | M | R31, R62 |
| O8 | Round float literals once, from the decimal | L16 | S | -- |
| O9 | `triangular_solve` at `f16`: measure, then widen only where needed | L3 | S | -- |
| O11 | Reject a loop-expression start whose 64-bit steps overflow; only if O20 does not land | O19 | S | R28 |
| O12 | Drop the matmul-precision flag; transfer buffers without `jax.device_put` | -- | S | R32 |
| O13 | Differential reference over the instruction record | O2 | M | -- |
| O15 | Apply the naming table: identifiers, comments, docs | L25, L27, O1 | M | -- |
| O19 | Spike: unroll every loop at trace time, measure compile time | O21, O8 | S | -- |
| O20 | Unroll every loop; delete loop algebra, `Unroll`, `while` emission | O19 | M | -- |
| O22 | Enforce the numerics norm table: a guard column and a golden per row | L6, O3 | M | -- |
| O23 | Detect data-dependent errors inside the module | O22, O3 | M | -- |
| O24 | Call-time errors: checked 64-bit values, overlapping outputs, constraint overflow | L5, L6, O20 | S | -- |

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
