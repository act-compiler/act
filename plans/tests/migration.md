# Migration plan

The migration moves from the per-PR suites to the spec-driven suite in PR-sized steps, each deleting what it replaces in the same PR.
The order keeps every repo green at every step and never deletes a test before its replacement runs in CI.
T16 to T18 are the steps that are not already an item elsewhere.

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| T16 | CI wiring: corpus travels with the pinned taidl release | T5, T12 | S | Q1 |
| T17 | Split `act-oracle/tests/test_kernel.py` by theme | T1 | M | Q1 |
| T18 | Final deletion pass of PR-defense tests | T4, T13, T14, T20 | M | Q2 |

## Ordered steps

| step | item | repo | deletes | quarter |
| --- | --- | --- | --- | --- |
| 1 | T2 conftest decoupling | act-backend, act-oracle | nothing | Q1 |
| 2 | T1 spec tags on kept tests | all | nothing | Q1 |
| 3 | T5 corpus format and loader | taidl | nothing | Q1 |
| 4 | T6 front-end verdict runner | taidl | nothing | Q1 |
| 5 | T7 seed from grammar tests, one outline file per PR | taidl | the replaced parametrize blocks of `tests/test_grammar.py` | Q1 |
| 6 | T11 expectations tables and coverage.md renderer | act-backend, act-oracle | `conftest.py:75-155` `LOWERED` in act-oracle | Q1 |
| 7 | T16 CI wiring through the pin | act-backend, act-oracle | nothing | Q1 |
| 8 | T12 expectation runners | act-backend, act-oracle | `test_refusals.py` word and type loops in both repos | Q1 |
| 9 | T3 shared test kit | act and taidl (T22 question 1) | duplicated conftest helpers | Q1 |
| 10 | T10 accelerator list | taidl | `EXAMPLES` in `taidl/tests/test_examples.py` | Q1 |
| 11 | T17 split test_kernel.py | act-oracle | nothing | Q1 |
| 12 | T8 operations corpus | taidl, act-oracle | the per-word loops move, the files stay as runners | Q1 |
| 13 | T14 shared end-to-end | act | `test_mini_trn*.py` in both generator repos | Q1 |
| 14 | T15 golden policy | act | the two `regenerate.py` | Q2 |
| 15 | T13 differential tests | act | selection-fact tests of `test_matching.py` and `test_constructs.py` | Q2 |
| 16 | T9 property-based tests | taidl, act-oracle, act-backend | nothing | Q2 |
| 17 | T4 and T18 deletion pass | all | internal-text pins and remaining PR-defense tests | Q2 |

## T16. CI wiring: corpus travels with the pinned taidl release

- What: the taidl wheel ships `taidl/conformance/` as package data; a generator's CI reads the corpus from the installed taidl at the pin in `pyproject.toml:16`, with an override path for a local taidl checkout.
- Benefits: a generator tests against the language version it claims; a pin bump is the moment new cases arrive and new rows are needed.
- Caveats: the corpus grows the wheel; cases are small Python and TOML, so the size stays in kilobytes, but `inputs.npz` files for T13 stay out of the wheel and live in act.
- Fail points: a generator CI runs against an unpinned taidl main; `act/tests/check_versions.py` already checks pins agree and gains a check that the corpus version matches.
- Order and dependencies: after T5; before T12 runs in CI.
- Verification: `python -c "import taidl.conformance"` works in each generator's image; `check_versions.py` passes.
- Breaking: no.
- Files: `act/tests/check_versions.py` extended; CI config of both generators.

## T17. Split `act-oracle/tests/test_kernel.py` by theme

- What: split the 1744-line file into kernel API tests (modes :46-57, debug :129-198, loops :216-440, declarations :671-935, compile and run :952-1091) and spec-level tests (looped attributes and region shapes :1287-1728) that T7 and T13 absorb.
- Benefits: a reviewer can read a file; the spec-level half gets S5.2 tags and leaves the API half untouched.
- Caveats: O owns the tree; the split is mechanical and lands as one PR with no behaviour change.
- Fail points: a module-level fixture shared across the split files breaks; noticed at collection.
- Order and dependencies: after T1; before T7 takes the spec-level half.
- Verification: no test file over 800 lines in act-oracle; test count unchanged.
- Breaking: no.
- Files: `act-oracle/tests/test_kernel.py`; every test preserved, moved by theme.

## T18. Final deletion pass of PR-defense tests

- What: once T13 and T14 are green, delete the remaining bespoke-description tests of `act-backend/tests/test_matching.py` and `test_constructs.py` whose fact is now a corpus case, and the assertion-free tests TB1 to TB5 that T1 did not repair.
- Benefits: the suites that remain are spec conformance, generator expectations, differential, end-to-end, and a small set of impl tests that say so in their file header.
- Caveats: each deleted test must name the corpus case that replaces it in the PR description; a test with no replacement stays and gets a header saying it is an impl test.
- Fail points: a deleted test guarded a search invariant no result check sees (as `test_extraction.py` does); those stay as impl tests by rule T20.
- Order and dependencies: last step; Q2.
- Verification: every test file header names its style from the inventory key; `grep -rn 'request.module' */tests/` finds nothing.
- Breaking: no.
- Files: remaining bespoke tests of `test_matching.py` and `test_constructs.py`; each deletion names its corpus case.
