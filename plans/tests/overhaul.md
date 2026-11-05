# Overhaul: target structure from the current suites

The target suite is derived from what exists: every current file and fixture is kept as is, reshaped, merged, or deleted, and the facts of a merged or deleted test survive as a corpus case, an expectations row, or a differential case.
The steps run from today's trees in the order of the migration steps; this file shows where each file ends up.
T20 sets the impl-header rule, and `decisions/justifications.md` records every delete-and-rewrite.

## Target layout

| location | holds | comes from |
| --- | --- | --- |
| `taidl/conformance/<group>/<slug>/` | corpus cases: `spec.py` in the authoring surface, `case.toml`, `kernel.hlo`, `inputs.npz`, `program.toml` (T5) | `taidl/tests/test_grammar.py` parametrize blocks, rejection cases of `test_instruction.py` and `test_expressions.py`, ISA specification texts from both generators' `test_refusals.py` |
| `taidl/conformance/accelerators.md` | real ISA specification list (T10) | `taidl/tests/test_examples.py` `EXAMPLES` |
| `taidl/conformance/norms.md`, `coverage.md`, `CHANGES.md` | norm catalogue, coverage report, case-id delta (T26, T5) | new |
| `taidl/tests/test_conformance.py` | `check` verdict runner (T6) | new |
| converter script and equivalence tests (2.x line) | one `Model` equality test per converted file (T37) | new, deleted with the string surface |
| `taidl.testing` program runner, `act-oracle` adapter | `program.toml` cases on the oracle (T29) | seeds of `act-oracle/tests/test_kernel.py:590-598`, :1229-1240 |
| `taidl/tests/test_*.py` (kept) | API unit tests with S tags (T1) | unchanged files |
| `act-backend/tests/expectations.toml`, `act-oracle/tests/expectations.toml` | supported or unsupported per case (T11) | `test_refusals.py` loops, `conftest.py` `LOWERED` |
| `act-backend/docs/coverage.md`, `act-oracle/docs/coverage.md` | rendered from expectations (T11) | `act-oracle/docs/operations.md` table |
| `act-backend/tests/test_conformance.py`, `act-oracle/tests/test_conformance.py` | expectation runners (T12) | new |
| `act-backend/tests/test_*.py` (kept, header `impl` or `cli`) | search, allocator, CLI, build contract | `test_endings.py`, `test_reader.py`, `test_extraction.py`, `test_allocator.py`, `test_cost.py`, `test_generation.py`, `test_names.py`, `test_package.py`, trimmed `test_matching.py` |
| `test_rejections.py` in both generators | rejection tests that no row covers | what remains of `test_refusals.py` after T12 |
| `act-oracle/tests/test_kernel_*.py` | kernel API by theme (T17) | `test_kernel.py` |
| `act-oracle/tests/test_operations_*.py` | S8 runners over corpus cases (T8) | `test_elementwise.py`, `test_reduce.py`, `test_structural.py`, `test_compare_select_clamp.py`, `test_linalg.py`, `test_constant_iota.py`, `test_templates.py` |
| `act/tests/test_end_to_end.py` | real ISA specifications through both generators (T14) | itself plus `test_mini_trn*.py` of both generators |
| `act/tests/test_differential.py` | oracle vs backend on corpus kernels (T13) | selection facts of `test_matching.py`, `test_constructs.py`, `test_reader.py:240-298`, `test_endings.py:620-641` |
| `act/tests/regenerate.py` | goldens (T15) | both `regenerate.py` |
| `act/tests/` mapped-deletion script | fails a PR that drops a test id without a successor (T28) | new |
| `act/testkit/` and `taidl.testing` | shared helpers and tables (T3) | duplicated conftest helpers |

## Per file and fixture

| current | action | preserved | item |
| --- | --- | --- | --- |
| `taidl/tests/test_grammar.py` | reshape | every parametrized case as a corpus directory in the authoring surface; lexer-level tests are deleted with `taidl/antlr4/` | T7 |
| `taidl/tests/test_grammar.py::TestDocs` | reshape | doc-sync intent; table shape stated | T11 |
| `taidl/tests/test_instruction.py` | keep as is, tag | all; TB1 and TB2 gain assertions | T1 |
| `taidl/tests/test_expressions.py` | keep as is, tag | all | T1 |
| `taidl/tests/test_data_model.py` | keep as is, tag | all | T1 |
| `taidl/tests/test_examples.py` | merge | loads and parses every listed specification | T10 |
| `act-backend/tests/test_matching.py` | reshape | result-visible facts as differential cases; search facts as impl tests | T13, T18 |
| `act-backend/tests/test_constructs.py` | merge | each construct as a corpus case with kernel | T7, T13 |
| `act-backend/tests/test_endings.py` | keep as is | all; constants also as corpus cases | - |
| `act-backend/tests/test_reader.py` | keep as is, tag | all | T1 |
| `act-backend/tests/test_extraction.py` | keep as is, header `impl` | all | T20 |
| `act-backend/tests/test_generators.py` | reshape | template-fill and rewrite tests, and the mark test at :243 (the markers stay); layout tests deleted | T4 |
| `act-backend/tests/test_refusals.py` | reshape | every rejected ISA specification as a corpus case and a row; `:439` as the runner's rule; the arity check at :164-176 stays impl until the slow tier runs per PR | T11, T12 |
| `act-backend/tests/test_generation.py` | keep as is | all but layout tests at :277-338 | T4 |
| `act-backend/tests/test_allocator.py` | keep as is, header `impl` | all | T20 |
| `act-backend/tests/test_cost.py` | keep as is | all | - |
| `act-backend/tests/test_names.py` | keep, extend | all | T9 |
| `act-backend/tests/test_package.py` | keep as is | all; TB4 gains an assertion | - |
| `act-backend/tests/test_mini_trn.py`, `test_mini_trn_double.py` | merge | reference and golden checks; one smoke test stays | T14 |
| `act-backend/tests/conftest.py` `work_dir`, `assembly`, `output` | reshape | behaviour; explicit example parameter | T2 |
| `act-backend/tests/conftest.py` `endings`, `kernel`, `accelerator`, `long_names` | keep as is | ISA specification builders used by CLI tests | - |
| `act-backend/tests/conftest.py` `incremental` marker | delete | nothing; the mini_trn chain moves to T14 fixtures | T14 |
| `act-backend/tests/conftest.py` helpers duplicated in act-oracle | merge | moved to the kit | T3 |
| `act-backend/tests/regenerate.py` | merge | the record format | T15 |
| `act-oracle/tests/test_kernel.py` | reshape | all tests, split by theme; :1287-1728 also as corpus cases; :590-598, :1229-1240 as `program.toml` cases | T17, T7, T29 |
| `act-oracle/tests/test_refusals.py` | reshape | every rejected ISA specification as a corpus case and a row | T11, T12 |
| `act-oracle/tests/test_templates.py` | merge | each instruction case into S8 runners | T8 |
| `act-oracle/tests/test_elementwise.py` and the five reference files | keep, cases move | all; cases become corpus data | T8 |
| `act-oracle/tests/test_constraints.py`, `test_state.py` | keep as is, tag | all | T1 |
| `act-oracle/tests/test_examples.py` | keep as is | all; reads the T10 list | T10 |
| `act-oracle/tests/test_mini_trn.py`, `test_mini_trn_double.py` | merge | reference and golden checks | T14 |
| `act-oracle/tests/conftest.py` `LOWERED`, `DTYPES`, `ULPS` | reshape | the data, as expectations rows and case tolerances; `LOWERED` is replaced by a corpus runner over S8 words | T11, T8 |
| `act-oracle/tests/conftest.py` helpers duplicated in act-backend | merge | moved to the kit | T3 |
| `act-oracle/tests/regenerate.py` | merge | - | T15 |
| `act/tests/test_end_to_end.py` | keep, extend | all | T14 |
| `act/tests/check_versions.py` | keep, extend | all | T16 |

## Ordered steps from today

- Step 1 (T28, T2): mapped-deletion check; decouple the two conftest files; no test deleted.
- Step 2 (T1): tag kept tests with S ids; fix TB1, TB2, TB4 by adding the missing assertion.
- Step 3 (T5, T6, T26, T29, T38): corpus loader, `check` runner, surface cases, norm catalogue, and program runner with seed cases; existing tests untouched.
- Step 3b (T37): converter run over the ISA specification files, one equivalence test per file; no original deleted.
- Step 4 (T7): one PR per outline file moves `test_grammar.py` blocks into cases and deletes the block moved.
- Step 5 (T8): S8 cases with stored values and tolerances; the six reference files become runners.
- Step 6 (T11, T16, T12): expectations tables, generators reading the corpus through the pin, and runners; `LOWERED` and the rejection loops deleted as the runners cover them.
- Step 7 (T27, T3, T10, T17): CI tiers, shared kit, accelerator list, kernel file split; mechanical PRs with unchanged test counts.
- Step 8 (T14, T15): end-to-end and goldens in act; `test_mini_trn*.py` and both `regenerate.py` deleted after one green release.
- Step 9 (T13): differential tests; selection-fact tests deleted one for one as their case lands.
- Step 10 (T9, T24, T25, T23): property tests, fuzzer, mutation pilot, shape module check.
- Step 11 (T4, T18): deletion pass; headers on every surviving impl file.
- Step 12 (T37): the equivalence tests, the grammar tests, and the string surface are deleted together.
