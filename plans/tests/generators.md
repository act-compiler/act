# Per-generator expectations, differential testing, and CI tiers

Each generator states, per corpus case, one coverage status of S1.2: `supported`, `partial`, `excluded`, or `rejected` with a coded error.
That table is the single source for the generator's `docs/coverage.md`, and the differential tests run the oracle and the backend on the cases both support.
Items T11 to T15, T23, T27, and T31 cover the expectations table, its runner, differential tests, the end-to-end test, the golden policy, the shape module check, and the CI tiers.

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| T11 | Expectations table per generator, coverage.md generated from it | T5, D1 | M | Q1 |
| T12 | Expectation runner: the row's status, with the documented code | T11, T6, D6 | M | Q1 |
| T13 | Differential tests oracle vs backend on the supported intersection | T3, T12, T8, T29, B29 | L | Q2 |
| T14 | One shared end-to-end suite in act | T3, T10, B29 | M | Q1 |
| T15 | Golden policy: one regenerate, one record, one tolerance | T14, B29 | S | Q2 |
| T23 | Check the shape module copied into each generated oracle | U4, T12 | S | Q2 |
| T27 | CI tiers, budgets, and deterministic mode | T12, B29 | M | Q1 |
| T31 | Backend overlap rows: allocation per relation, copy for a tied shared input | T13, T30, B29 | M | Q2 |

## T11. Expectations table per generator, coverage.md generated from it

- What: `act-backend/tests/expectations.toml` and `act-oracle/tests/expectations.toml`, one row per corpus case id, with one status of S1.2.
- What: `supported` has no further field; `partial` adds the stated limit and the section id; `excluded` and `rejected` add the `code` and the section id of the clause they document.
- What: the oracle excludes nothing (owner decision 42), so its rows are `supported`, `partial`, or `rejected`.
- What: rows for tensor operators and element types replace `act-backend/tests/test_refusals.py:146-205` and `act-oracle/tests/conftest.py:75-155` (`LOWERED`).
- What: T11 owns the one `coverage.md` renderer: clause text from the diagnostic registry (D1), verdicts from the expectations rows; the existing `act-oracle/docs/operations.md:21-40` table is regenerated from it.
- Benefits: coverage.md cannot drift from the tests; a new corpus case with no row fails collection, so a feature that is not `supported` is documented on the day it is tested.
- Caveats: the backend has no coverage doc today (`act-backend/README.md:48` states the contract in one line); B writes the first clause texts.
- Fail points: a row marked supported for a case that compiles but selects nothing; T13 catches it because the differential run finds no assembly.
- Order and dependencies: after T5 publishes case ids and D1 the registry; before T4 deletes the regex tests.
- Verification: `docs/coverage.md` is byte-identical to the renderer's output in CI; every corpus case id has a row in both files.
- Breaking: no.
- Files: `act-backend/tests/test_refusals.py:146-205`, `act-oracle/tests/conftest.py:75-155`, `act-oracle/docs/operations.md`; every word and type fact preserved as a row.

## T12. Expectation runner: the row's status, with the documented code

- What: `tests/test_conformance.py` in each generator loads the pinned taidl's corpus, runs `generate_backend` or `generate_oracle` on each accepted case, and asserts the row's status.
- What: `supported` and `partial`: generation succeeds, and a `partial` case lies inside the limit its row states.
- What: `excluded`: generation succeeds, and the printed and stored exclusion carries the documented `code` and clause (S1.2).
- What: `rejected`: generation raises the documented `code`.
- What: a backend never assumes an unsupported input away: every input outside its support is `excluded` or `rejected` with a coded error, and the runner fails a case that compiles to a result instead.
- What: a `rejected` case leaves no output tree, as `act-backend/tests/test_refusals.py:439` and `act-oracle/tests/test_refusals.py:467` check today.
- What: the tests that stay in `test_refusals.py` move to `test_rejections.py` in the PR that deletes the loops.
- Benefits: the generator rejection tests shrink to a data file; the code is tested against the rows coverage.md shows the user.
- Caveats: a backend generation builds a cargo crate per case, so the runner has two tiers.
- Caveats: the fast tier runs the pre-build checks (`spec_checks` at `test_refusals.py:150`) per case; the slow tier runs per pull request (T27), grouping accepted cases of one section into one accelerator and building once.
- Caveats: the arity check at `act-backend/tests/test_refusals.py:164-176` stays an `impl` test until the slow tier runs per pull request.
- Fail points: a corpus bump in the pin adds cases with no rows; the runner fails collection and names the missing ids.
- Order and dependencies: after T11, T6, and D6 (unsupported registries); one PR per generator.
- Verification: `pytest -k conformance` passes in each generator with the pinned taidl; every row id exists in the corpus; every row status is one of the four of S1.2.
- Breaking: yes: generator rejections gain the code and the clause id, released with the matching generator majors (act-backend 2.0, taidl-to 3.0).
- Files: both `test_refusals.py`; the no-output-left rule at backend :439 and oracle :467 preserved in the runner.

## T13. Differential tests oracle vs backend on the supported intersection

- What: for each corpus case whose row is `supported` or `partial` in both generators (the supported intersection) and that carries a `kernel.hlo` and inputs, compile the kernel with the backend, simulate the assembly on the oracle, and compare to the case's stored result within the case tolerance.
- What: one invariant holds for every case: simulating the assembly of a supported kernel never faults.
- What: the first cases come from `act-backend/tests/test_matching.py` selection facts (:898 dot contracting pairs, :1247 slice ranges, :1177-1189 sum from a start), `test_constructs.py:437-575`, and `test_reader.py:240-298`.
- What: fault cases are oracle-only programs (T29); the backend is a producer that never emits a faulting issue, so T13 holds no fault case.
- What: the overlap rows are T31; the cases of T30 that both generators support also run here and must agree on values.
- What: T13 to T15 run the backend in the deterministic mode of B29.
- Benefits: replaces the assertion of which instruction was selected with a check of the result, which is what the language specification promises.
- Caveats: needs both generators in one environment, as `act/tests/test_end_to_end.py:11` does; lives in act, runs nightly and on release, and the per-pull-request slow tier of T27 is its fast companion.
- Fail points: a case where oracle and backend agree and both are wrong; the stored values from T8 guard it.
- Fail points: a kernel the backend cannot compile is a coded rejection, not a mismatch; the runner compares no-assembly with the T11 rows.
- Order and dependencies: after T3, T12, T8, T29, and B29; Q2; A owns the act tree.
- Verification: every case with a kernel is either differentially tested or named in an expectations row as `excluded` or `rejected` by one generator.
- Breaking: no.
- Files: `act-backend/tests/test_matching.py`, `test_constructs.py`, `test_reader.py:240-298`, `test_endings.py:620-641`; each selection fact preserved as a kernel plus expected result.

## T14. One shared end-to-end suite in act

- What: `act/tests/test_end_to_end.py` grows to run every ISA specification in T10's list: generate both, compile every kernel, simulate, compare to the reference and the golden, replacing `act-backend/tests/test_mini_trn*.py` and `act-oracle/tests/test_mini_trn*.py`.
- What: a kernel that an expectations row marks `excluded` or `rejected` is skipped and reported; T10's row names the S8 words and S5 features it needs.
- Benefits: one place runs the real ISA specifications; the generator repos stop carrying example data twice.
- Caveats: each generator repo keeps one smoke test that generates `mini_trn` and checks the tree is filled (`act-backend/tests/test_mini_trn.py:26-30`), so a generator PR still builds its own example.
- Caveats: the smoke test is the only per-pull-request signal that a generator builds its example.
- Fail points: a GPU run differs from the CPU golden; GPU checks are manual and use the reference tolerance (T15).
- Order and dependencies: after T3, T10, and B29; before T4 deletes the per-repo mini_trn files.
- Verification: the act suite reports one result per ISA specification and kernel of T10's list; the per-repo mini_trn test files are gone.
- Breaking: no.
- Files: `act/tests/test_end_to_end.py` grows; `act-backend/tests/test_mini_trn*.py`, `act-oracle/tests/test_mini_trn*.py` merge in; reference and golden checks preserved; the `incremental` marker at `act-backend/tests/conftest.py:25-39` goes.

## T15. Golden policy: one regenerate, one record, one tolerance

- What: one `regenerate.py` in act replaces `act-backend/tests/regenerate.py` and `act-oracle/tests/regenerate.py`; the record (`goldens.json` at `act-backend/tests/regenerate.py:42-52`) is the format; the tolerance lives beside the golden.
- What: goldens are exact on the CPU, generated in deterministic mode; a golden update is a PR that names the semantic change and shows the record diff.
- What: GPU is no bitwise golden target; hosted runners have no GPU, so GPU checks are manual and use the reference tolerance (`act-oracle/tests/test_mini_trn.py:22`).
- What: `precision_config` HIGHEST on every `dot` holds only while the oracle lowers to XLA; the durable rule is the bound of S1.3.15.
- Benefits: a golden mismatch names the oracle, jax, platform, and machine it came from, as the record already does.
- Fail points: a golden regenerated from a wrong result; the script rejects a golden the reference rejects (:35-37), and the policy keeps that rule.
- Order and dependencies: after T14 and B29.
- Verification: one regenerate script across the four repos; every golden has a record.
- Breaking: no.
- Files: both `regenerate.py`; the record format at `act-backend/tests/regenerate.py:42-52` preserved.

## T23. Check the shape module copied into each generated oracle

- What: a test generates an oracle and asserts its copied shape module equals the front end's module at the version stamped in it.
- What: the module imports nothing, so the test also imports it alone.
- Benefits: the single source of shape rules cannot drift from the oracle's copy.
- Caveats: lands after U4 moves the rules into the front end.
- Fail points: a generator that edits the copy; the equality check fails.
- Order and dependencies: after U4 and T12; Q2.
- Verification: the check runs in act-oracle CI against the pinned taidl.
- Breaking: no.
- Files: new test in `act-oracle/tests/`; no existing file changed.

## T27. CI tiers, budgets, and deterministic mode

| Tier | Trigger | Budget | Overrun action |
| --- | --- | --- | --- |
| taidl | pull request | 5 min | split the job |
| generator fast | pull request | 10 min | move cases to slow |
| generator slow | pull request, changed paths | 20 min | quarantine, then nightly |
| act differential | nightly | 90 min | ticket with owner |
| release | tag | 3 h | block the tag |

- What: the slow tier in act-backend runs one build and one oracle-simulated kernel per S8 word on each pull request, beside the nightly differential tier.
- What: budgets are proposals, fixed after a week of measurement; oracle-only tiers are re-budgeted when the oracle moves to fsim-lib.
- What: tests run the search in the deterministic mode of B29; the default wall-clock bound stays for users.
- What: most corpus kernels compile within 1 s on the standard CI machine; the exceptions are named in one list.
- What: a flaky test is quarantined in one PR with an owner and a removal date, never rerun until green.
- What: the oracle time per kernel is recorded for each T10 ISA specification and printed in the coverage docs, so the budgets rest on measured oracle cost.
- What: the gate for B29 is byte identity of assembly and candidate logs, via a same-job double run per architecture; corpus-side assembly facts are check lines.
- Benefits: a deleted per-pull-request test has a per-pull-request replacement; the tiers have triggers and limits.
- Caveats: amd64 versus arm64 equality is unproven, so goldens are per architecture until measured.
- Fail points: a tier over budget for a week; the overrun action fires.
- Order and dependencies: after T12 and B29.
- Verification: each tier appears in CI config with its trigger and budget; the coverage docs list the oracle time per kernel of each T10 ISA specification.
- Breaking: no.
- Files: CI config of the four repos; no test file removed.

## T31. Backend overlap rows (S5.5)

- What: per relation (tied, apart, exact-or-apart, may overlap), a kernel in which an input dies at the instruction that writes the output, and the test asserts the allocation overlaps the two only as the relation permits.
- What: the allocation is read from the assembly's addresses; the oracle simulates the same assembly and agrees with the stored result (Q6.2).
- What: a tied input shared with another reader gets a copy before the instruction, and the oracle result is unchanged.
- What: a tied input pinned to HBM or held as a shared view is a coded rejection, not a wrong answer.
- What: a default pair is allocated equal-or-apart; if the solver limit forces apart for default pairs, the rows assert apart and name the limit as a backend limit.
- What: the unmarked kernel in which an author forgot `apart` on a streaming unit is a T30 fault on the oracle and never a corrupt allocation on the backend.
- What: the backend emits no issue that faults on any T30 cell it compiles; T13 simulates every such assembly.
- Benefits: the allocator's touch exception is tested per relation, the place a dying input may share storage with its output.
- Caveats: the default-pair allocation and the tied equal-offset constraint are not landed in act-backend today, so the rows land with B29 and the allocator change.
- Fail points: a row that passes because the allocator separated the tensors for an unrelated reason; each row names the relation and checks the addresses.
- Order and dependencies: after T13, T30, and B29; Q2.
- Verification: each of the four relations has an allocation row, and the copy and HBM-pinned rows exist.
- Breaking: no.
- Files: seeds from `act-backend/tests/test_refusals.py:383-396`, `test_extraction.py:441-563`, `test_mini_trn.py:68-78`, and `test_mini_trn_double.py:70-80`; the facts are preserved as rows.
