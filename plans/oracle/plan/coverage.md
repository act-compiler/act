# act-oracle coverage and docs

`act-oracle/docs/coverage.md` lists each language feature with its status, citing the language spec's section ids.
`act-oracle/docs/operations.md` and `README.md` still restate parts of the language, such as the S8.3 element-type table and the operand rules, which the spec owns.
This plan makes the generator's docs state only the oracle's deviations and its own API.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O14 | Fold `docs/operations.md` into `docs/coverage.md` and README | L1, L7 | S | Q1 |
| O17 | Keep `docs/coverage.md` true by test | O14, T | S | Q1 |

## O14 Fold operations.md

- Breaking: yes.
- Changes: `docs/operations.md`, `docs/coverage.md`, `README.md`; preserves the oracle-only rows and the test bounds.
- What: `docs/operations.md:14-97` repeats the spec's element-type table (S8.3) row by row, with "lowers to" a direct name in all but a few rows.
- What: keep only what is the oracle's: the rows whose reference operation is not the same-named StableHLO one (`chlo.*`, `copy` as `reshape`, `parameter` as `dynamic_slice`, `reduce_*`), the `dot` widening of 2-bit integers, the test bounds table (`docs/operations.md:139-156`).
- What: move those into `docs/coverage.md` as notes and a "Test bounds" section; delete `docs/operations.md`; point README at the spec for the language and at `coverage.md` for the oracle.
- What: add a normative-platform statement: the CPU is the reference, the GPU is held to the bound table (`README.md:163`), and every GPU deviation (O9) is listed.
- What: README sections "What an oracle refuses" and "The names an oracle reserves" (`README.md:54-82`) restate rejections; keep the reserved names, which are the oracle's, and point to `coverage.md` for the rest.
- Benefits: one place per fact; the spec can change S8.3 without a stale copy here.
- Caveats: the reserved-name table also appears in S4.1; L27 should replace it with a pointer to the oracle's README.
- Fail points: a link from another repo to `docs/operations.md` breaks; grep `act`, `act-backend`, `taidl` before deleting.
- Order and dependencies: after L1 and L7; A8 updates any umbrella links.
- Verification: no word-by-word copy of S8.3 remains in act-oracle; `git ls-files | xargs grep -n operations.md` lists only intended links.

## O17 Coverage kept true by test

- Breaking: no.
- Changes: `tests/`; preserves `docs/coverage.md` as written.
- What: the shipped table keeps only `partial` and `rejected` rows, the S1.2 status vocabulary.
- What: a test reads the `partial` and `rejected` rows of `docs/coverage.md` and, for each `rejected` row, generates a spec that uses the feature and asserts the documented exception class.
- Benefits: the doc cannot drift from the code silently.
- Caveats: `partial` rows need one hand-written case each.
- Fail points: a test that parses markdown is brittle; keep the table format fixed and fail loudly on a parse error.
- Order and dependencies: after O14 and O6, since O6 changes exception classes.
- Verification: removing a rejection from the code fails the test.
