# Load and critical path

Size weights: S = 1, M = 3, L = 6, in unassisted engineer-weeks, to compare items.
The plan assumes Claude-assisted development, so weights rank work and do not cap it.

| Track | Q1 items | Q1 weight | Q2 items | Q2 weight | Later |
| ----- | -------- | --------- | -------- | --------- | ----- |
| language | 16 | 18 | 6 | 18 | 9 |
| backend | 21 | 44 | 13 | 33 | 0 |
| oracle | 7 | 7 | 16 | 34 | 1 |
| tests | 3 | 7 | 19 | 50 | 0 |
| act | 14 | 22 | 8 | 21 | 0 |
| casestudies | 8 | 21 | 5 | 19 | 0 |
| priorart | 7 | 15 | 5 | 19 | 0 |
| shared | 0 | 0 | 9 | 24 | 0 |
| diagnostics | 6 | 14 | 8 | 24 | 0 |
| semantics | 6 | 15 | 3 | 12 | 0 |
| total | 88 | 163 | 92 | 254 | 10 |

## Longest dependency chain

Weight 59, by size weights.

- B29: Deterministic search mode and `search` as a library function (M, Q1)
- B1: Target module layout and stage names (S, Q1)
- B31: Typed payloads in `TensorOp` (M, Q1)
- B7: Shape and element type inferred in `Analysis::make` (L, Q1)
- B23: Check shape and element-type agreement of semantics lines (D3) (M, Q2)
- O7: Move static checks to the front end: shapes, region values, constant ranges, `compare` type (M, Q2)
- L3: Normative shape and element type rules per operation (L, Q2)
- L16: Range and type checks at the declaring call (S, Q2)
- O8: Round float literals once, from the decimal (S, Q2)
- O19: Spike: unroll every loop at trace time, measure compile time (S, Q2)
- O20: Unroll every loop; delete loop algebra, `Unroll`, `while` emission (M, Q2)
- O1: Runtime code out of `API_FILE` into `runtime/`; split what remains of `utils.py` (M, Q2)
- O2: One instruction record from the visitor, consumed by checks and emission (M, Q2)
- O3: Emitter calls in generic form instead of string templates; one word table (M, Q2)
- S1: Close the two emitter divergences (S, Q2)
- T5: Corpus format, location, and naming (S, Q2)
- T11: Expectations table per generator, coverage.md generated from it (M, Q2)
- T12: Expectation runner: supported or rejected with the documented error (M, Q2)
- T13: Differential tests oracle vs backend on the supported intersection (L, Q2)
- T4: Delete tests that pin generated text or internal names (S, Q2)
- T18: Final deletion pass of PR-defense tests (M, Q2)
- T20: Decision: impl tests survive with a header, PR-defense tests do not (S, Q2)
