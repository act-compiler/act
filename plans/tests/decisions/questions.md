# Answers in force

## T22. Answers in force for the test-suite questions

| question | answer | items |
| --- | --- | --- |
| Where does the shared test kit live | split: `taidl.testing` for the corpus loader and tolerances, `act/testkit/` for goldens and oracle import | T3 |
| Do messages carry the clause id | yes, with the diagnostic `code`; cases match on `code` and `object`; the clause id sits on line 1 until the owner decides otherwise | T6, T12 |
| Is `hypothesis` allowed | yes, when T9 lands | T9, T24 |
| Is `mutmut` allowed | yes, when T25 lands | T25 |
| Is a StableHLO interpreter added | no; expected values come from NumPy or closed forms | T8 |
| Does each generator repo keep a `mini_trn` smoke test | yes, one per repo | T14 |
| Which ISA specification joins the T10 list | one GPU tensor-core specification (the fixed Hopper sketch), written by A | T10 |
| Is the GPU a CI target for goldens | no; GPU checks are manual and use the reference tolerance | T14, T15 |
| When does the differential tier run | nightly and on release; the per-pull-request slow tier of T27 runs beside it | T13, T27 |
| Backend search limit in tests | deterministic mode; the wall-clock bound stays the user default | T13, T27 |
| Public pins | released versions only; no pre-release tag, no commit or branch pin | T16 |

- What: the table above, the single record of the answers.
- Benefits: no test item waits on an owner question.
- Caveats: the clause id placement on line 1 changes with the owner's later word on diagnostics.
- Fail points: an answer applied differently in two repos; the answer is recorded here once.
- Order and dependencies: none.
- Verification: each answer names its items.
- Breaking: no.
- Files: none.
