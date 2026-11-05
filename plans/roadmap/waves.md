# Waves

Every item of the quarter tables sits in one wave, or in one merged or dropped row below.
The per-item rows live in wave-1.md, wave-2.md, wave-3.md, and wave-4-7.md.

## Rules

- The base is the regenerated tables: 243 IDs, of which 239 are scheduled and 4 are merged.
- An item's wave is never earlier than the wave of any item it depends on; a dependency inside the same wave sits in an earlier row.
- The default wave follows the item's dependency depth; an item moves later only to keep wave 1 measurable or to follow the release order.
- The width layer lands first; the encoding unit (L38, L39, B44, B45, T33, T34) lands whole as the last work before U8 (Q10.4).
- An other-core.md later item enters a wave before M5 only when a scheduled item depends on it (Q8.1).
- Weights use S = 1, M = 3, L = 6 unassisted engineer-weeks; they rank work and cap nothing.
- No date or quarter label is set before M1; the M1 rebaseline sets dates from wave 1 measurements (Q8.9).
- A measured assist factor below the need moves dates, never scope.

## Waves

| Wave | Items | Weight | Ends at | Outcome |
| ---- | ----- | ------ | ------- | ------- |
| 1 | 55 | 83 | -- | Foundations, pair relations, gates decided, assist factor measured |
| 2 | 42 | 83 | M1 | Normative rules, typed contract, width layer, backend shape chain |
| 3 | 60 | 132 | M2 | One front end validates for every generator; generator metadata; failure policy |
| 4 | 42 | 105 | M3 | Generators read the front end's checks and codes |
| 5 | 23 | 66 | M4 | Encoding unit, then v3 released on released pins |
| 6 | 7 | 16 | M5 | Tutorial on the release; cold-start trial |
| 7 | 10 | 30 | -- | Follow-ons after M5 |
| total | 239 | 515 | -- | -- |

## Wave 1 gates

Each gate decides later scope, so all of them close in wave 1 (Q8.10).

| Gate | Item | Named outcome |
| ---- | ---- | ------------- |
| Test-suite answers | T22 | Answers in force, cited by T-items |
| Test policy | T20 | Survival header rule that T4 and T18 apply |
| `hypothesis` (R45) | -- | Settled by owner 28: allowed when T9 lands |
| Interpreter (R29) | -- | Owner 28: no StableHLO interpreter; F9 stays held until R29 is ruled |
| Unroll bound (R28) | O19 | Owner sets the bound before the spike runs |
| Tutorial syntax | A10 | List of syntax that fails on the v3 API |
| Release order | A16 | Order and checklist in force before taidl 3.0 |
| Ascend study (R15) | -- | Case-studies track adds it as an item in wave 1 or 2, before R59 applies |

## Release order

Releases follow the act 2.x structure: released pins only, no pre-release tags, no commit or branch pins in public (owner 33).

- Development runs in the private repos; private CI may consume unreleased heads.
- The encoding unit is complete before the release train starts (Q10.4).
- taidl 3.0 is released first, by U8.
- taidl-to 3.0 and act-backend 2.0 then pin the released taidl 3.0 and are released.
- act 3.0 is released last by A23, pinning the released generators.
- T16 and T19 consume the corpus through the released taidl 3.0 pin.
- The new tutorial (wave 6) is written against the released act 3.0.

## Merged and dropped rows

Each merged row keeps its exit as a named sub-item of its absorber (Q8.6).

| ID | Disposition | Absorber | Decision |
| -- | ----------- | -------- | -------- |
| B22 | merged into | B21 | Q8.7: B22 lands with B21 |
| B23 | merged into | D5 | Q8.4 step c: consumer deletion; line agreement is U4's |
| L12 | merged into | U4 | Q8.4 step b: the shape module lives in taidl |
| O4 | merged into | O3 | Q8.6: O4 folds into O3, as the table states |

## Held items

- Held until their rulings: F3 (R56), F9 (R29), P6 (R20).
- These keep their waves; a ruling changes their text, not their place.

## Check

- Check (a script over the six tables and the four wave files): 243 table IDs listed exactly once (239 in waves, 4 merged), 0 missing, 0 extra, 0 listed twice; every row's dependencies and size match its table row; 397 dependency edges checked, 0 unknown IDs, 0 items before a dependency, counting row order inside a wave.
