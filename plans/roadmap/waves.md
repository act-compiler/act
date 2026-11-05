# Waves

Every item of the quarter tables sits in one wave, or in one merged or dropped row below.
The per-item rows live in wave-1.md, wave-2.md, wave-3.md, wave-4-7.md, and wave-8-10.md.

## Rules

- The base is the regenerated tables: 277 IDs, of which 266 are scheduled, 6 are merged, and 5 are dropped by decision.
- A14 and A22 no longer sit in the tables; they are merged rows here, outside the 277.
- An item's wave is never earlier than the wave of any item it depends on; a dependency inside the same wave sits in an earlier row.
- A dependency written "X or Y" holds when either item is in an earlier wave or an earlier row.
- The default wave follows the item's dependency depth; an item moves later only to keep wave 1 measurable or to follow the release order.
- The front-end migration follows the Q11.4 order: `check` on the string front end, generators on the `Model`, tracer, converter, equivalence, deletion.
- The width layer lands first.
- The encoding unit (L38, L39, B44, B45, T33, T34) is developed from wave 3 on the private heads and merges whole after the deletion pass, as the last work before U8 (Q10.4).
- A later item of other-core.md or other-supporting.md enters a wave before M5 only when a scheduled item depends on it (Q8.1).
- Weights use S = 1, M = 3, L = 6 unassisted engineer-weeks; they rank work and cap nothing.
- No date or quarter label is set before M1; the M1 rebaseline sets dates from wave 1 measurements (Q8.9).
- At the M1 rebaseline, every wave 1 gate and every size-L item gets a named owner and a close date.
- Wave 1 holds one size-L item (L3), so the assist factor is measured on each size class.
- A measured assist factor below the need moves dates, never scope.

## Waves

| Wave | Items | Weight | Ends at | Outcome |
| ---- | ----- | ------ | ------- | ------- |
| 1 | 55 | 88 | -- | Foundations, pair relations, gates decided, assist factor measured on S, M, and L items |
| 2 | 42 | 78 | M1 | Normative rules, typed contract, width layer, backend shape chain |
| 3 | 58 | 122 | M2 | `taidl.check` and the `Model` for every generator; traced surface; generator metadata |
| 4 | 49 | 122 | M3 | Generators read the checked model; exclusion report; failure policy; converter in the last 2.x release |
| 5 | 27 | 74 | M4 | Equivalence, string surface deleted, encoding unit merged, release gate, then v3 released on released pins |
| 6 | 5 | 14 | M5 | Tutorial on the release; cold-start trial |
| 7 | 10 | 25 | -- | Follow-ons after M5 |
| 8 | 8 | 24 | -- | Toolchain pin set, StableHLO importer, e-graph library core, fsim steps 1 and 2 |
| 9 | 7 | 24 | -- | Dialect, matcher, engine swap, torch adapter and entry points |
| 10 | 5 | 18 | M6 | Typed import, one C++ driver, partition opt-in, platform gate |
| total | 266 | 589 | -- | -- |

Through M5 (waves 1 to 6): 236 items, weight 498.

## Wave 1 gates

Each gate decides later scope, so all of them close in wave 1 (Q8.10).
Owners are roles until the M1 rebaseline names a person; every gate closes by the end of wave 1.

| Gate | Item | Named outcome | Owner role |
| ---- | ---- | ------------- | ---------- |
| Test-suite answers | T22 | Answers in force, cited by T-items | tests lead |
| Test policy | T20 | Survival header rule that T4 and T18 apply | tests lead |
| `hypothesis` (R45) | -- | Settled by owner 28: allowed when T9 lands | -- |
| Interpreter (R29) | -- | Owner 28: no StableHLO interpreter; F7 and F9 are cut (Q6.6) | -- |
| Unroll bound (R28) | O19 | Bound set by the Q5.1 resolution; the spike passes it or O11 lands | oracle lead |
| Tutorial syntax | A10 | List of syntax that fails on the v3 API | act lead |
| Release order | A16 | Order and checklist in force before taidl 3.0 | act lead |
| Private development | A32 | Repo topology, branching, private-to-public sync, private CI owner, and the 2.x maintenance policy | act lead |

## Release order

Releases follow the act 2.x structure: released pins only, no pre-release tags, no commit or branch pins in public (owner 33).

- Development runs in the private repos; private CI may consume unreleased heads (A32).
- Before v3, a non-breaking fix ships publicly as a 2.x patch release with released pins; a breaking change waits for v3 (owner 47).
- The last taidl 2.x release ships the converter script (A28, U13); taidl 3.0 holds no string surface (U14).
- The encoding unit is complete before the release train starts (Q10.4).
- The release gate (A31) passes on the M4 evidence before any v3 tag.
- taidl 3.0 is released first, by U8.
- taidl-to 3.0 and act-backend 2.0 then pin the released taidl 3.0 and are released.
- act 3.0 is released last by A23, after U8, pinning the released generators.
- T16 and T19 consume the corpus through the released taidl 3.0 pin.
- The new tutorial (wave 6) is drafted on the private heads before U8 and finished against the released act 3.0.

## Merged and dropped rows

Each merged row keeps its exit as a named sub-item of its absorber (Q8.6); a row dropped by decision has no exit.

| ID | Disposition | Absorber | Decision |
| -- | ----------- | -------- | -------- |
| B22 | merged into | B21 | Q8.7: B22 lands with B21 |
| L12 | merged into | U4 | Q8.4 step b: the shape module lives in taidl |
| O4 | merged into | O3 | Q8.6: O4 folds into O3, as the table states |
| L42 | merged into | U12 | One tracer item with one owner; L42's exit is a sub-item of U12 |
| L44 | merged into | U13 | One converter item with one owner; L44's exit is a sub-item of U13 |
| L45 | merged into | U14 | One deletion item with one owner; L45's exit is a sub-item of U14 |
| A14 | merged into | A24 | Q7.1: paper titles and setup text belong to the one new tutorial |
| A22 | merged into | A11 | Q7.1: the tutorial lint scope is one statement of the archive item |
| B23 | dropped by decision | -- | Cut: line agreement (D3) is U4's |
| L13 | dropped by decision | -- | Q11.4: no grammar remains to lex after U14 |
| L22 | dropped by decision | -- | Cut in the language plan |
| F7 | dropped by decision | -- | Cut: carried by T7, T8, T26, T29 (Q6.6) |
| F9 | dropped by decision | -- | Cut: carried by T8; no interpreter is built (Q6.6, owner 28) |

## Held items

- Held until their rulings: F3 (R56), P6 (R20).
- These keep their waves; a ruling changes their text, not their place.

## Quarter labels

The wave files are the schedule, and a quarter label in the tables sets no date.
The tables put Q1 items in waves 1 to 4, Q2 items in waves 3 to 6, and later items in waves 6 to 10, except the rows below.
Each listed item keeps its wave.

| ID | Tables | Wave | Why the wave holds |
| -- | ------ | ---- | ------------------ |
| L27 | later | 2 | B27 and O15 depend on it (Q8.1) |
| A6 | after v3 | 3 | D12 depends on it (Q8.1) |
| A18 | after v3 | 3 | The M4 release criterion needs it (Q8.7) |
| T12 | Q1 | 5 | After T6 and D6 of wave 4; the M4 criteria need it (Q8.7) |
| T16 | Q1 | 5 | After T12 |
| T27 | Q1 | 5 | After T12 |
| T35 | Q1 | 5 | After T12 |
| T38 | Q1 | 5 | After T12 |
| T37 | Q1 | 5 | After T38 |

## Check

A script over the six tables, the five wave files, and the merged and dropped rows gives this result once the tables are regenerated from the track plans.

- 277 table IDs listed exactly once: 266 in waves, 6 merged, 5 dropped by decision.
- A14 and A22 are outside the tables and sit in merged rows only.
- Every row's item text, dependencies, and size match its table row.
- Every table row of size `-` has a merged or dropped row, and no other table ID has one.
- Every dependency edge names a known ID, and no item sits before a dependency, counting row order inside a wave.
- The Q11.4 order holds by row: U2, U3, L43, U10, U12, U13, A28, T37, U14, the encoding unit, A31, U8, A23.
- Wave totals sum to 266 items and weight 589.
