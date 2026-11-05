# Load and critical path

Size weights: S = 1, M = 3, L = 6, in unassisted engineer-weeks, to compare items.
The plan assumes Claude-assisted development, so weights rank work and do not cap it.

| Track | Q1 items | Q1 weight | Q2 items | Q2 weight | Later |
| ----- | -------- | --------- | -------- | --------- | ----- |
| language | 27 | 42 | 5 | 9 | 14 |
| backend | 27 | 52 | 17 | 41 | 13 |
| oracle | 24 | 36 | 8 | 18 | 4 |
| tests | 25 | 65 | 13 | 37 | 3 |
| act | 14 | 22 | 11 | 30 | 5 |
| casestudies | 9 | 11 | 4 | 6 | 0 |
| priorart | 7 | 15 | 4 | 8 | 1 |
| shared | 4 | 10 | 9 | 27 | 2 |
| diagnostics | 8 | 20 | 9 | 23 | 1 |
| semantics | 5 | 9 | 2 | 9 | 2 |
| total | 150 | 282 | 82 | 208 | 45 |

## Longest dependency chain per milestone

The longest weighted chain that ends at an item of the milestone's waves, over the wave files, by size weights.
It is recomputed at each regeneration.

| Milestone | Weight | Chain |
| --------- | ------ | ----- |
| M1 | 15 | B29 -> B3 -> B31 -> B7 |
| M2 | 22 | B29 -> B3 -> B31 -> B7 -> B26 -> D7 |
| M3 | 28 | B29 -> B3 -> B31 -> B7 -> B9 -> B10 -> B11 -> B19 |
| M4 | 44 | B29 -> B3 -> B31 -> B7 -> B26 -> D7 -> D16 -> T32 -> T33 -> T34 -> A31 -> U8 -> A23 |
| M5 | 53 | the M4 chain -> A24 -> A21 |
| M6 | 64 | the M4 chain to U8 -> U9 -> U15 -> B48 -> B50 -> B54 -> B55 -> B56 -> T41 |

- The backend shape chain (B29 to B7) heads every chain, so B7 carries the K27 tripwire.
- Ties at the same weight: B5 can stand for B26, D5 or T32 can end the M3 chain, and A12 can end the M5 chain.
- From M4 on, the release gate A31 and U8 sit on every chain, and A23 on the M4 and M5 chains.
