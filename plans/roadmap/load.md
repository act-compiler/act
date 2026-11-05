# Load and critical path

Size weights: S = 1, M = 3, L = 6, in unassisted engineer-weeks, to compare items.
The plan assumes Claude-assisted development, so weights rank work and do not cap it.

| Track | Q1 items | Q1 weight | Q2 items | Q2 weight | Later |
| ----- | -------- | --------- | -------- | --------- | ----- |
| language | 28 | 48 | 6 | 12 | 11 |
| backend | 27 | 52 | 17 | 46 | 2 |
| oracle | 24 | 36 | 8 | 18 | 1 |
| tests | 25 | 65 | 13 | 37 | 0 |
| act | 14 | 24 | 9 | 26 | 3 |
| casestudies | 9 | 11 | 4 | 6 | 0 |
| priorart | 7 | 15 | 4 | 8 | 1 |
| shared | 4 | 10 | 9 | 27 | 1 |
| diagnostics | 8 | 20 | 9 | 23 | 1 |
| semantics | 5 | 9 | 2 | 9 | 2 |
| total | 151 | 290 | 81 | 212 | 22 |

## Longest dependency chain

Weight 35, by size weights.

- B29: Deterministic search mode and `search` as a library function (M, Q1)
- B1: Target module layout and stage names (S, Q1)
- B31: Typed payloads in `TensorOp` (M, Q1)
- B7: Shape and element type inferred in `Analysis::make` (L, Q1)
- B26: Extraction errors as dropped candidates, unset shape as `Option`, shape disagreement as a coded fault (D7, D8) (S, Q1)
- D7: Backend runtime diagnostics (Rust and C++) (L, Q2)
- D16: Failure policy (decision 40) (M, Q2)
- T32: Width layer cases (range faults, `u64` rejection) (M, Q2)
- T33: Encoding layer cases and assembler round trip (L, Q2)
- T34: Derived attribute cases (closed-form links, rejected links) (M, Q2)
