# Review of the current test suites

The four suites hold 508 tests: 131 in taidl, 223 in act-backend, 153 in act-oracle, 1 in act.
Much of it is good: the oracle's reference tests, the front-end API tests, and the CLI contract tests are worth keeping as they stand.
The weak parts are the per-PR selection tests, the three hand-copied word tables, and two conftest files that couple modules to fixtures by name.
Verdicts use keep, reshape, merge, delete; the inventory holds the per-file reasons and TB1 to TB22 the defects.

## What works and should be kept

- Reference-value tests against NumPy per operation word and element type: `act-oracle/tests/test_elementwise.py:406`, `test_reduce.py:87-129`, `test_structural.py:239-403`, `test_compare_select_clamp.py:122-146`, `test_linalg.py:94-166`, `test_constant_iota.py:107-157`; these are language tests already and become the S8 corpus with stored expected values (T8).
- Front-end API tests with wide parametrization: `taidl/tests/test_instruction.py:48-66` (names), :213-268 (regions), :294-440 (overlap; seeds of T30), `test_expressions.py:24-250`, `test_data_model.py:34-59`; they map onto S4 to S6 and need only tags (T1).
- Grammar word and type sweeps that derive their cases from the grammar itself: `taidl/tests/test_grammar.py:279-297`, `act-backend/tests/test_refusals.py:146-157` and :185-197, `act-oracle/tests/test_refusals.py:67`; the sweep idea is right and becomes the expectations table (T11).
- Rejection tests that check the error the user sees and that nothing is left behind: `act-backend/tests/test_refusals.py:439`, `act-oracle/tests/test_refusals.py:467-477`, `act-backend/tests/test_generation.py:36`.
- CLI contract tests of the backend binary: `act-backend/tests/test_endings.py:112-404` cover exit status, output files, log directories, and concurrency at :263; they are complete and outside the language specification's scope.
- Kernel API contract tests of the oracle: `act-oracle/tests/test_kernel.py:46-1091` cover modes, loops, debug, declarations, and compile order; too long in one file, right in content.
- State and constraint order tests: `act-oracle/tests/test_state.py:81-130`, `test_constraints.py:67-149`; direct S5.3 and S5.4 tests.
- Goldens with provenance: `act-backend/tests/regenerate.py:35-52` rejects a golden the reference rejects and records oracle, jax, platform, and machine.
- The one cross-generator test: `act/tests/test_end_to_end.py:69-101` runs both generators on one ISA specification and checks an exact bf16 result; it is the seed of T13 and T14.
- Property-shaped tests: `act-backend/tests/test_names.py:95` compiles whatever names the generated sources spell; `test_generation.py:338` lays out names of every length.
- Test names read as sentences of behaviour throughout act-backend and act-oracle (`test_endings.py:117`, `test_kernel.py:652`), which the new suite keeps as its naming convention.

## What is weak

- Selection tests assert which instruction was chosen, not what the result is: `act-backend/tests/test_matching.py:648-661` parse `api.` calls out of assembly text; a correct but different selection fails them.
- Three copies of the operation-word list drift independently (TB17 and the inventory).
- Fixtures couple to module constants by name: `act-backend/tests/conftest.py:463-495` read `request.module.EXAMPLE_DIR`.
- Generated text is pinned: `act-backend/tests/test_generators.py:239`, :264, `test_mini_trn.py:72`, `test_constructs.py:500`.
- Formatter layout is tested twice: `test_generators.py:175-393` by hand and `conftest.py:468-472` on every generated tree.
- Seventeen tests show no assertion in their body; some assert inside helpers, some pass by not raising (TB1 to TB6).
- No generator documents its unsupported features in a form a test reads; `act-oracle/docs/operations.md:21` is close but hand-maintained.
- No differential test exists beyond the one 4x4 matmul in act.
- No test runs an instruction-level program with attributes, addresses, and state, so the run-time norm has no executable cases (T29).

## Verdict per file

| repo | file | verdict | goes to |
| --- | --- | --- | --- |
| taidl | `test_grammar.py` | reshape | T7 corpus cases; `TestDocs` to T11 doc-sync |
| taidl | `test_instruction.py` | keep | T1 tags; rejections also mirrored as corpus cases |
| taidl | `test_expressions.py` | keep | T1 tags |
| taidl | `test_data_model.py` | keep | T1 tags |
| taidl | `test_examples.py` | merge | T10 list |
| act-backend | `test_matching.py` | reshape | T13 differential; impl remainder with header |
| act-backend | `test_constructs.py` | merge | T7 and T13 |
| act-backend | `test_endings.py` | keep | constants at :620-641 to T13 |
| act-backend | `test_reader.py` | keep | sum and dot reading also in T13 |
| act-backend | `test_extraction.py` | keep | impl header |
| act-backend | `test_generators.py` | reshape | delete layout tests (T4); keep template tests and the mark test at :243 (the markers stay) |
| act-backend | `test_refusals.py` | reshape | T11 rows and T12 runner; the rest becomes `test_rejections.py` |
| act-backend | `test_generation.py` | keep | layout tests at :277-338 go with T4 |
| act-backend | `test_allocator.py` | keep | impl header |
| act-backend | `test_cost.py` | keep | - |
| act-backend | `test_names.py` | keep | extended in T9 |
| act-backend | `test_package.py` | keep | - |
| act-backend | `test_mini_trn.py`, `test_mini_trn_double.py` | merge | T14; one smoke test stays |
| act-backend | `conftest.py` | reshape | T2, T3 |
| act-oracle | `test_kernel.py` | reshape | T17 split; :1287-1728 to T7 and T29 |
| act-oracle | `test_refusals.py` | reshape | T11 rows and T12 runner; the rest becomes `test_rejections.py` |
| act-oracle | `test_templates.py` | merge | T8 |
| act-oracle | `test_elementwise.py`, `test_reduce.py`, `test_structural.py`, `test_compare_select_clamp.py`, `test_linalg.py`, `test_constant_iota.py` | keep | T8 moves the cases, the files stay as runners |
| act-oracle | `test_constraints.py`, `test_state.py` | keep | T1 tags |
| act-oracle | `test_examples.py` | keep | reads T10 list |
| act-oracle | `test_mini_trn.py`, `test_mini_trn_double.py` | merge | T14 |
| act-oracle | `conftest.py` | reshape | T2, T3, T11 |
| act | `test_end_to_end.py` | keep | grows into T13 and T14 |
| act | `check_versions.py` | keep | gains the corpus version check and the released-pin rule (T16) |

## Counts

| repo | keep | reshape | merge | delete |
| --- | --- | --- | --- | --- |
| taidl | 85 | 42 | 4 | 0 |
| act-backend | 112 | 83 | 16 | 12 |
| act-oracle | 110 | 31 | 7 | 0 |
| act | 1 | 0 | 0 | 0 |

- Delete counts the layout tests of `test_generators.py` and the three egraph regex and text pins of `test_refusals.py` and `test_mini_trn*.py`, as counted on the planning day; every other removal happens by merge, with the fact preserved as a corpus case or an expectations row.
