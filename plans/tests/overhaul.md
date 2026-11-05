# Overhaul: target structure from the current suites

The target suite is derived from what exists: every current file and fixture is kept as is, reshaped, merged, or deleted, and the facts of a merged or deleted test survive as a corpus case, an expectations row, or a differential case.
The steps run from today's trees in the order of the migration steps; this file shows where each file ends up.
T20 records the justification of every delete-and-rewrite.

## Target layout

| location | holds | comes from |
| --- | --- | --- |
| `taidl/conformance/S*/` | corpus cases: spec, verdict, kernel, inputs (T5) | `taidl/tests/test_grammar.py` parametrize blocks, rejection cases of `test_instruction.py` and `test_expressions.py`, spec texts from both generators' `test_refusals.py` |
| `taidl/conformance/accelerators.md` | real ISA description list (T10) | `taidl/tests/test_examples.py` `EXAMPLES` |
| `taidl/tests/test_conformance.py` | front-end verdict runner (T6) | new |
| `taidl/tests/test_*.py` (kept) | API unit tests with S tags (T1) | unchanged files |
| `act-backend/tests/expectations.toml`, `act-oracle/tests/expectations.toml` | supported or rejected per case (T11) | `test_refusals.py` loops, `conftest.py` `LOWERED` |
| `act-backend/docs/coverage.md`, `act-oracle/docs/coverage.md` | rendered from expectations (T11) | `act-oracle/docs/operations.md` table |
| `act-backend/tests/test_conformance.py`, `act-oracle/tests/test_conformance.py` | expectation runners (T12) | new |
| `act-backend/tests/test_*.py` (kept, header `impl` or `cli`) | search, allocator, CLI, build contract | `test_endings.py`, `test_reader.py`, `test_extraction.py`, `test_allocator.py`, `test_cost.py`, `test_generation.py`, `test_names.py`, `test_package.py`, trimmed `test_matching.py` |
| `act-oracle/tests/test_kernel_*.py` | kernel API by theme (T17) | `test_kernel.py` |
| `act-oracle/tests/test_operations_*.py` | S8 runners over corpus cases (T8) | `test_elementwise.py`, `test_reduce.py`, `test_structural.py`, `test_compare_select_clamp.py`, `test_linalg.py`, `test_constant_iota.py`, `test_templates.py` |
| `act/tests/test_end_to_end.py` | real ISA descriptions through both generators (T14) | itself plus `test_mini_trn*.py` of both generators |
| `act/tests/test_differential.py` | oracle vs backend on corpus kernels (T13) | selection facts of `test_matching.py`, `test_constructs.py`, `test_reader.py:240-298`, `test_endings.py:620-641` |
| `act/tests/regenerate.py` | goldens (T15) | both `regenerate.py` |
| `act/testkit/` and `taidl.testing` | shared helpers and tables (T3) | duplicated conftest helpers |

## Per file and fixture

| current | action | preserved | item |
| --- | --- | --- | --- |
| `taidl/tests/test_grammar.py` | reshape | every parametrized case as a corpus directory; lexer-level tests stay | T7 |
| `taidl/tests/test_grammar.py::TestDocs` | reshape | doc-sync intent; table shape stated | T11 |
| `taidl/tests/test_instruction.py` | keep as is, tag | all; TB1 and TB2 gain assertions | T1 |
| `taidl/tests/test_expressions.py` | keep as is, tag | all | T1 |
| `taidl/tests/test_data_model.py` | keep as is, tag | all | T1 |
| `taidl/tests/test_examples.py` | merge | loads and parses every listed spec | T10 |
| `act-backend/tests/test_matching.py` | reshape | spec-visible facts as differential cases; search facts as impl tests | T13, T18 |
| `act-backend/tests/test_constructs.py` | merge | each construct as a corpus case with kernel | T7, T13 |
| `act-backend/tests/test_endings.py` | keep as is | all; constants also as corpus cases | - |
| `act-backend/tests/test_reader.py` | keep as is, tag | all | T1 |
| `act-backend/tests/test_extraction.py` | keep as is, header `impl` | all | T20 |
| `act-backend/tests/test_generators.py` | reshape | template-fill and rewrite tests; layout tests deleted | T4 |
| `act-backend/tests/test_refusals.py` | reshape | every rejected ISA description as a corpus case and a row; `:439` as the runner's rule | T11, T12 |
| `act-backend/tests/test_generation.py` | keep as is | all but layout tests at :277-338 | T4 |
| `act-backend/tests/test_allocator.py` | keep as is, header `impl` | all | T20 |
| `act-backend/tests/test_cost.py` | keep as is | all | - |
| `act-backend/tests/test_names.py` | keep, extend | all | T9 |
| `act-backend/tests/test_package.py` | keep as is | all; TB4 gains an assertion | - |
| `act-backend/tests/test_mini_trn.py`, `test_mini_trn_double.py` | merge | reference and golden checks; one smoke test stays | T14 |
| `act-backend/tests/conftest.py` `work_dir`, `assembly`, `output` | reshape | behaviour; explicit example parameter | T2 |
| `act-backend/tests/conftest.py` `endings`, `kernel`, `accelerator`, `long_names` | keep as is | ISA description builders used by CLI tests | - |
| `act-backend/tests/conftest.py` `incremental` marker | delete | nothing; the mini_trn chain moves to T14 fixtures | T14 |
| `act-backend/tests/conftest.py` helpers duplicated in act-oracle | merge | moved to the kit | T3 |
| `act-backend/tests/regenerate.py` | merge | the record format | T15 |
| `act-oracle/tests/test_kernel.py` | reshape | all tests, split by theme; :1287-1728 also as corpus cases | T17, T7 |
| `act-oracle/tests/test_refusals.py` | reshape | every rejected ISA description as a corpus case and a row | T11, T12 |
| `act-oracle/tests/test_templates.py` | merge | each instruction case into S8 runners | T8 |
| `act-oracle/tests/test_elementwise.py` and the five reference files | keep, cases move | all; cases become corpus data | T8 |
| `act-oracle/tests/test_constraints.py`, `test_state.py` | keep as is, tag | all | T1 |
| `act-oracle/tests/test_examples.py` | keep as is | all; reads the T10 list | T10 |
| `act-oracle/tests/test_mini_trn.py`, `test_mini_trn_double.py` | merge | reference and golden checks | T14 |
| `act-oracle/tests/conftest.py` `LOWERED`, `DTYPES`, `ULPS` | reshape | the data, as expectations rows and case tolerances | T11, T8 |
| `act-oracle/tests/conftest.py` helpers duplicated in act-backend | merge | moved to the kit | T3 |
| `act-oracle/tests/regenerate.py` | merge | - | T15 |
| `act/tests/test_end_to_end.py` | keep, extend | all | T14 |
| `act/tests/check_versions.py` | keep, extend | all | T16 |

## Ordered steps from today

- Step 1 (T2): decouple the two conftest files; no test deleted; both generator suites still pass as they are.
- Step 2 (T1): tag kept tests with S ids; fix TB1, TB2, TB4 by adding the missing assertion.
- Step 3 (T5, T6): corpus loader and front-end runner in taidl with ten seed cases; existing taidl tests untouched.
- Step 4 (T7): one PR per outline file moves `test_grammar.py` blocks into cases and deletes the block moved.
- Step 5 (T11): expectations tables seeded from `test_refusals.py` loops and `LOWERED`; coverage.md rendered; `LOWERED` deleted.
- Step 6 (T16, T12): generators read the corpus through the pin; rejection loops deleted as the runner covers them.
- Step 7 (T3, T10, T17): shared kit, accelerator list, kernel file split; mechanical PRs with unchanged test counts.
- Step 8 (T8): S8 cases with references and tolerances; the six reference files become runners.
- Step 9 (T14, T15): end-to-end and goldens in act; `test_mini_trn*.py` and both `regenerate.py` deleted after one green release.
- Step 10 (T13): differential tests; selection-fact tests deleted one for one as their case lands.
- Step 11 (T9): property-based tests.
- Step 12 (T4, T18): deletion pass; headers on every surviving impl file.
