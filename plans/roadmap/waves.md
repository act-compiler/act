# Waves

Every item of the quarter tables sits in one wave, or in one merged or dropped row below.
The per-item rows live in wave-1.md, wave-2.md, wave-3.md, and wave-4-7.md.

## Rules

- The base is the regenerated tables: 254 IDs, of which 247 are scheduled, 3 are merged, and 4 are dropped by decision.
- A14 and A22 no longer sit in the tables; they are merged rows here, outside the 254.
- An item's wave is never earlier than the wave of any item it depends on; a dependency inside the same wave sits in an earlier row.
- The default wave follows the item's dependency depth; an item moves later only to keep wave 1 measurable or to follow the release order.
- The front-end migration follows the Q11.4 order: `check` on the string front end, generators on the `Model`, tracer, converter, equivalence, deletion.
- The width layer lands first; the encoding unit (L38, L39, B44, B45, T33, T34) lands whole as the last work before U8 (Q10.4).
- A later item of other-core.md or other-supporting.md enters a wave before M5 only when a scheduled item depends on it (Q8.1).
- Weights use S = 1, M = 3, L = 6 unassisted engineer-weeks; they rank work and cap nothing.
- No date or quarter label is set before M1; the M1 rebaseline sets dates from wave 1 measurements (Q8.9).
- A measured assist factor below the need moves dates, never scope.

## Waves

| Wave | Items | Weight | Ends at | Outcome |
| ---- | ----- | ------ | ------- | ------- |
| 1 | 53 | 81 | -- | Foundations, pair relations, gates decided, assist factor measured |
| 2 | 42 | 83 | M1 | Normative rules, typed contract, width layer, backend shape chain |
| 3 | 59 | 128 | M2 | `taidl.check` and the `Model` for every generator; traced surface; generator metadata |
| 4 | 50 | 130 | M3 | Generators read the checked model; exclusion report; failure policy; converter in the last 2.x release |
| 5 | 27 | 74 | M4 | Equivalence, string surface deleted, encoding unit, then v3 released on released pins |
| 6 | 5 | 14 | M5 | Tutorial on the release; cold-start trial |
| 7 | 11 | 31 | -- | Follow-ons after M5 |
| total | 247 | 541 | -- | -- |

## Wave 1 gates

Each gate decides later scope, so all of them close in wave 1 (Q8.10).

| Gate | Item | Named outcome |
| ---- | ---- | ------------- |
| Test-suite answers | T22 | Answers in force, cited by T-items |
| Test policy | T20 | Survival header rule that T4 and T18 apply |
| `hypothesis` (R45) | -- | Settled by owner 28: allowed when T9 lands |
| Interpreter (R29) | -- | Owner 28: no StableHLO interpreter; F7 and F9 are cut (Q6.6) |
| Unroll bound (R28) | O19 | Bound set by the Q5.1 resolution; the spike passes it or O11 lands |
| Tutorial syntax | A10 | List of syntax that fails on the v3 API |
| Release order | A16 | Order and checklist in force before taidl 3.0 |
| Ascend study (R15) | -- | Case-studies track adds it as an item in wave 1 or 2, before R59 applies |

## Release order

Releases follow the act 2.x structure: released pins only, no pre-release tags, no commit or branch pins in public (owner 33).

- Development runs in the private repos; private CI may consume unreleased heads.
- The last taidl 2.x release ships the converter script (A28, L44); taidl 3.0 holds no string surface (L45, U14).
- The encoding unit is complete before the release train starts (Q10.4).
- taidl 3.0 is released first, by U8.
- taidl-to 3.0 and act-backend 2.0 then pin the released taidl 3.0 and are released.
- act 3.0 is released last by A23, pinning the released generators.
- T16 and T19 consume the corpus through the released taidl 3.0 pin.
- The new tutorial (wave 6) is written against the released act 3.0.

## Merged and dropped rows

Each merged row keeps its exit as a named sub-item of its absorber (Q8.6); a row dropped by decision has no exit.

| ID | Disposition | Absorber | Decision |
| -- | ----------- | -------- | -------- |
| B22 | merged into | B21 | Q8.7: B22 lands with B21 |
| L12 | merged into | U4 | Q8.4 step b: the shape module lives in taidl |
| O4 | merged into | O3 | Q8.6: O4 folds into O3, as the table states |
| A14 | merged into | A24 | Q7.1: paper titles and setup text belong to the one new tutorial |
| A22 | merged into | A11 | Q7.1: the tutorial lint scope is one statement of the archive item |
| B23 | dropped by decision | -- | Cut: line agreement (D3) is U4's |
| L13 | dropped by decision | -- | Q11.4: no grammar remains to lex after L45 |
| F7 | dropped by decision | -- | Cut: carried by T7, T8, T26, T29 (Q6.6) |
| F9 | dropped by decision | -- | Cut: carried by T8; no interpreter is built (Q6.6, owner 28) |

## Held items

- Held until their rulings: F3 (R56), P6 (R20).
- These keep their waves; a ruling changes their text, not their place.

## Check

A script over the six tables, the four wave files, and the merged and dropped rows gives this result.

- 254 table IDs listed exactly once: 247 in waves, 3 merged, 4 dropped by decision.
- A14 and A22 are outside the tables and sit in merged rows only.
- 0 missing, 0 extra table IDs, 0 listed twice.
- Every row's item text, dependencies, and size match its table row.
- Every table row of size `-` has a merged or dropped row, and no other table ID has one.
- 430 dependency edges checked: 0 unknown IDs, 0 items before a dependency, counting row order inside a wave.
- The Q11.4 order holds by row: U2, U3, L43, U10, L42, U12, L44, U13, A28, T37, L45, U14, the encoding unit, U8.
- Wave totals sum to 247 items and weight 541, matching load.md's 290 plus 212 plus 39 for the fifteen scheduled later items.
