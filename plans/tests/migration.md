# Migration plan

The migration moves from the per-PR suites to the language-specification-driven suite in PR-sized steps, each deleting what it replaces in the same PR.
The order keeps every repo green at every step and never deletes a test before its replacement runs in CI.
T16, T17, T18, T28, and T37 are the steps that are not already an item elsewhere.
T37 follows the order of work of Q11.4: operation table and node model, `check` over the string front end, generators on `Model`, tracer, converter, equivalence tests, deletion of the string surface.

## Migration rules

- A test is deleted only in the step whose runner covers its fact at the same cadence.
- The S8 word and type cases land before T11.
- Deletions are named by function in the PR description.
- T28 fails a PR in which a test id disappears without a mapped successor.
- Tests of removed constructs become `rejected` cases naming the replacement, so the error is the migration aid.
- `LOWERED` is replaced by a corpus runner over S8 words, not by a lowering table.
- Marker tests (`set_inplace`, `set_apart`, and the others, spelled as `write` keywords in v3) and overlap tests move into the T30 matrix and the T31 rows; the markers stay, so no marker test is deleted unmapped.
- A string-form test is deleted only after its ISA specification is converted and the T37 equivalence test passes; the string front end, `taidl/antlr4/`, the grammar tests, and the parser dependency are deleted in the final step (Q11.4).

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| T16 | CI wiring: corpus travels with the pinned taidl release | T5, T12 | S | Q1 |
| T17 | Split `act-oracle/tests/test_kernel.py` by theme | T1 | M | Q1 |
| T18 | Final deletion pass of PR-defense tests | T4, T13, T14, T20 | M | Q2 |
| T28 | Mapped-deletion check | - | S | Q1 |
| T37 | Converter equivalence tests per ISA specification file | T6, T38, U3 | M | Q1 |

## Ordered steps

| step | item | repo | deletes | quarter |
| --- | --- | --- | --- | --- |
| 1 | T28 mapped-deletion check | act | nothing | Q1 |
| 2 | T2 conftest decoupling | act-backend, act-oracle | nothing | Q1 |
| 3 | T1 section tags on kept tests | all | nothing | Q1 |
| 4 | T5 corpus format and loader | taidl | nothing | Q1 |
| 5 | T6 front-end verdict runner | taidl | nothing | Q1 |
| 6 | T26 norm catalogue and coverage report, T29 program runner | taidl, act-oracle | nothing | Q1 |
| 6a | T37 converter run on the corpus and equivalence test per file | taidl, act-oracle, act-backend, act | nothing | Q1 |
| 7 | T7 seed from grammar tests, one outline file per PR | taidl | the replaced parametrize blocks of `tests/test_grammar.py` | Q1 |
| 8 | T8 operations corpus | taidl, act-oracle | the per-word loops move, the files stay as runners | Q1 |
| 9 | T11 expectations tables and coverage.md renderer | act-backend, act-oracle | `conftest.py:75-155` `LOWERED` in act-oracle | Q1 |
| 10 | T16 CI wiring through the pin | act-backend, act-oracle | nothing | Q1 |
| 11 | T12 expectation runners | act-backend, act-oracle | `test_refusals.py` word and type loops in both repos | Q1 |
| 12 | T27 CI tiers | all | nothing | Q1 |
| 13 | T3 shared test kit | act and taidl | duplicated conftest helpers | Q1 |
| 14 | T10 accelerator list | taidl | `EXAMPLES` in `taidl/tests/test_examples.py` | Q1 |
| 15 | T17 split test_kernel.py | act-oracle | nothing | Q1 |
| 16 | T14 shared end-to-end | act | `test_mini_trn*.py` in both generator repos | Q1 |
| 17 | T15 golden policy | act | the two `regenerate.py` | Q2 |
| 18 | T13 differential tests | act | selection-fact tests of `test_matching.py` and `test_constructs.py` | Q2 |
| 19 | T9, T24, T25 property tests, fuzzer, mutation pilot | taidl, act-oracle, act-backend | nothing | Q2 |
| 20 | T23 shape module check | act-oracle | nothing | Q2 |
| 21 | T4 and T18 deletion pass | all | internal-text pins and remaining PR-defense tests | Q2 |
| 22 | T37 teardown with the string surface | taidl, act-oracle, act-backend | the equivalence tests, the grammar tests, `taidl/antlr4/` | Q2 |

## T16. CI wiring: corpus travels with the pinned taidl release

- What: the taidl wheel ships `taidl/conformance/` as package data; a generator's CI reads the corpus from the installed taidl at the pin in `pyproject.toml:16`.
- What: public CI pins released versions only: no pre-release tag, no commit or branch pin; a local taidl checkout override serves private development only.
- What: `check_versions.py` expects taidl, taidl-to, and act at 3.0 and act-backend at 2.0, each pinning released dependencies, and prints the `CHANGES.md` delta of the corpus on a pin bump.
- Benefits: a generator tests against the language version it claims; a pin bump is the moment new cases arrive and new rows are needed.
- Caveats: the corpus grows the wheel; cases are small Python and TOML, but `inputs.npz` files for T13 stay out of the wheel and live in act.
- Fail points: a generator CI runs against an unreleased taidl; `check_versions.py` fails on a pre-release, commit, or branch pin.
- Order and dependencies: after T5 and T12; before T12 runs in CI.
- Verification: `python -c "import taidl.conformance"` works in each generator's image; `check_versions.py` passes.
- Breaking: no.
- Files: `act/tests/check_versions.py` extended; CI config of both generators.

## T17. Split `act-oracle/tests/test_kernel.py` by theme

- What: split the 1744-line file into kernel API tests (modes :46-57, debug :129-198, loops :216-440, declarations :671-935, compile and run :952-1091) and language-level tests (looped attributes and region shapes :1287-1728) that T7 and T29 absorb.
- Benefits: a reviewer can read a file; the language-level half gets S5.2 tags and leaves the API half untouched.
- Caveats: O owns the tree; the split is mechanical and lands as one PR with no behaviour change.
- Fail points: a module-level fixture shared across the split files breaks; noticed at collection.
- Order and dependencies: after T1; before T7 takes the language-level half.
- Verification: no test file over 800 lines in act-oracle; test count unchanged.
- Breaking: no.
- Files: `act-oracle/tests/test_kernel.py`; every test preserved, moved by theme.

## T18. Final deletion pass of PR-defense tests

- What: once T13 and T14 are green, delete the remaining bespoke-specification tests of `act-backend/tests/test_matching.py` and `test_constructs.py` whose fact is now a corpus case, and the assertion-free tests TB1 to TB5 that T1 did not repair.
- Benefits: the suites that remain are conformance, generator expectations, differential, end-to-end, and a small set of impl tests that say so in their file header.
- Caveats: each deleted test names the corpus case that replaces it, found by `origin = "seeded:<old test path>"` in `case.toml`; a test with no replacement stays with an `impl` header.
- Fail points: a deleted test guarded a search invariant no result check sees (as `test_extraction.py` does); those stay as impl tests by rule T20.
- Order and dependencies: last step; T20 has no dependency and comes first.
- Verification: every test file header names its style from the inventory key; `grep -rn 'request.module' */tests/` finds nothing.
- Breaking: no.
- Files: remaining bespoke tests of `test_matching.py` and `test_constructs.py`; each deletion names its corpus case.

## T28. Mapped-deletion check

- What: a script in act compares the test ids of a PR's base and head and fails when an id disappears without a mapped successor (a corpus case, an expectations row, or a differential case).
- What: the PR description lists the mapping, keyed by function name.
- Benefits: a deletion cannot lose a fact unseen; the check holds at the cadence of the PR.
- Caveats: ids move with file splits (T17), so the mapping lists renames as successors.
- Fail points: a mapping that names a successor that does not run in CI; the script checks the successor exists.
- Order and dependencies: first step; no dependency.
- Verification: a test PR that drops an id without a mapping fails in CI.
- Breaking: no.
- Files: new script under `act/tests/`; no existing file changed.

## T37. Converter equivalence tests per ISA specification file

- What: the converter is a script on the taidl 2.x line, run under taidl 2.x, that reads loaded v2 `Instruction` objects and prints the authoring surface; it is not a `taidl` subcommand (Q11.4).
- What: the script runs once over every ISA specification file of the four repos: the Python files of taidl, act-oracle, and act-backend (25 counted on the planning day) and the files of act (`tests/test_end_to_end.py`; the archived tutorials are not converted).
- What: each converted file is committed beside the corpus seed it feeds (T7) or replaces its original in the step that deletes the string front end.
- What: one test per file builds the `Model` from the v2 string form through `check` and the `Model` traced from the converted file, and asserts equality up to call sites.
- What: a test that assembles semantics text at run time is rewritten by hand from the HLO-text printer's output, and its description names the printer output it started from.
- What: the converter's own cases are the T38 trace rows read backward: every v2 construct has a converted spelling or a coded rejection naming the route (Q3.12).
- What: the v3 removed-names rejection of `add_semantics` names the converter and its release; a `rejected` case holds that row.
- Benefits: equality of `Model` tests the contract itself, not text, so a converter bug fails a named file.
- Caveats: the tests run only on the branch where both front ends exist, and they are deleted with the string surface; the converted files stay.
- Fail points: a `Model` field outside equality (a call site compared by accident); the equality definition excludes call sites by name and a test checks one site differs.
- What: A28 runs this per-file check on act's own converted files before the last 2.x release; this item extends it to every corpus file.
- Order and dependencies: after T6, T38, and U3; the tracer and the `Model` over the string front end exist first; before T7 seeds from converted files; before U14 and the release gate A31.
- Verification: one test per converted file and a count equal to the file count; no v2 file remains unconverted at step 22.
- Breaking: no.
- Files: new converter script and equivalence tests on the 2.x line; converted files replace the originals as T7 and step 22 delete them.
