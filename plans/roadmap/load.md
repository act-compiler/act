# Load and critical path

Size weights: S = 1, M = 3, L = 6, in unassisted engineer-weeks, to compare items.
The plan assumes Claude-assisted development, so weights rank work and do not cap it.

| Track | Q1 items | Q1 weight | Q2 items | Q2 weight | Later |
| ----- | -------- | --------- | -------- | --------- | ----- |
| language | 27 | 40 | 4 | 8 | 10 |
| backend | 26 | 51 | 17 | 46 | 2 |
| oracle | 24 | 36 | 8 | 18 | 1 |
| tests | 23 | 59 | 13 | 37 | 0 |
| act | 13 | 21 | 14 | 33 | 0 |
| casestudies | 9 | 11 | 4 | 6 | 0 |
| priorart | 7 | 15 | 4 | 8 | 1 |
| shared | 4 | 10 | 7 | 18 | 0 |
| diagnostics | 8 | 20 | 8 | 22 | 0 |
| semantics | 6 | 12 | 3 | 12 | 0 |
| total | 147 | 275 | 82 | 208 | 14 |

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
