# Load and critical path

Size weights: S = 1, M = 3, L = 6, in unassisted engineer-weeks, to compare items.
The plan assumes Claude-assisted development, so weights rank work and do not cap it.

| Track | Q1 items | Q1 weight | Q2 items | Q2 weight | Later |
| ----- | -------- | --------- | -------- | --------- | ----- |
| language | 14 | 22 | 14 | 45 | 0 |
| backend | 17 | 38 | 11 | 31 | 0 |
| oracle | 11 | 17 | 7 | 24 | 0 |
| tests | 3 | 7 | 19 | 50 | 0 |
| act | 14 | 22 | 8 | 21 | 0 |
| casestudies | 8 | 21 | 5 | 19 | 0 |
| priorart | 7 | 15 | 5 | 19 | 0 |
| shared | 0 | 0 | 9 | 24 | 0 |
| diagnostics | 4 | 10 | 10 | 28 | 0 |
| semantics | 6 | 15 | 3 | 12 | 0 |
| total | 84 | 167 | 91 | 273 | 0 |

## Longest dependency chain

Weight 47, by size weights.

- L12: Shape and element type checker (L, Q2)
- B23: Check shape and element-type agreement of semantics lines (D3) (M, Q2)
- O7: Move `shapes.py` rules to the front end, keep call-time evaluation (L, Q2)
- L3: Normative shape and element type rules per operation (L, Q2)
- S4: Shape and per-word element-type rules in the front end (L, Q2)
- S8: Coordinated v3 release train (M, Q2)
- T8: Operations corpus with reference values (S8) (L, Q2)
- T13: Differential tests oracle vs backend on the supported intersection (L, Q2)
- T4: Delete tests that pin generated text or internal names (S, Q2)
- T18: Final deletion pass of PR-defense tests (M, Q2)
- T20: Decision: impl tests survive with a header, PR-defense tests do not (S, Q2)
