# Per-generator expectations and differential testing

Each generator states, per corpus case, whether it supports the program or rejects it with a documented error.
That table is the single source for the generator's `docs/coverage.md`, and the differential tests run the oracle and the backend on the cases both support.
Items T11 to T15 cover the expectations table, its runner, differential tests, the shared end-to-end test, and the golden policy.

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| T11 | Expectations table per generator, coverage.md generated from it | T5, B, O | M | Q1 |
| T12 | Expectation runner: supported or rejected with the documented error | T11, T6 | M | Q1 |
| T13 | Differential tests oracle vs backend on the supported intersection | T3, T12, T8 | L | Q2 |
| T14 | One shared end-to-end suite in act | T3, T10 | M | Q1 |
| T15 | Golden policy: one regenerate, one record, one tolerance | T14 | S | Q2 |

## T11. Expectations table per generator, coverage.md generated from it

- What: `act-backend/tests/expectations.toml` and `act-oracle/tests/expectations.toml`, one row per corpus case id: `supported`, or `rejected` with the error regex and the coverage.md clause it documents.
- What: rows for operation words and element types replace `act-backend/tests/test_refusals.py:146-205` and `act-oracle/tests/conftest.py:75-155` (`LOWERED`); the existing `act-oracle/docs/operations.md:21-40` table is the first coverage.md and is regenerated from the rows.
- Benefits: coverage.md cannot drift from the tests because a script renders it from the same file; a new corpus case with no row fails collection, so an unsupported feature is documented on the day it is tested.
- Caveats: the backend has no coverage doc today (`act-backend/README.md:48` states the rejection contract in one line); B writes the first clause texts.
- Fail points: a row marked supported for a case that compiles but selects nothing; T13 catches it because the differential run finds no assembly.
- Order and dependencies: after T5 publishes case ids; B and O own the clause texts; before T4 deletes the regex tests.
- Verification: `docs/coverage.md` is byte-identical to the renderer's output in CI; every corpus case id has a row in both files.
- Breaking: no.
- Files: `act-backend/tests/test_refusals.py:146-205`, `act-oracle/tests/conftest.py:75-155`, `act-oracle/docs/operations.md`; every word and type fact preserved as a row.

## T12. Expectation runner: supported or rejected with the documented error

- What: `tests/test_conformance.py` in each generator loads the pinned taidl's corpus, runs `generate_backend` or `generate_oracle` on each accepted case, and asserts the row: generation succeeds, or raises the documented error matching the regex.
- What: a refused case must leave no output tree, as `act-backend/tests/test_refusals.py:439` and `act-oracle/tests/test_refusals.py:467` check today.
- Benefits: the generator rejection tests shrink to a data file; the error text is tested against the words coverage.md shows the user.
- Caveats: a backend generation builds a cargo crate per case, so the runner has two tiers: the fast tier runs the pre-build checks (`spec_checks` at `test_refusals.py:150`) per case, the slow tier groups accepted cases of one spec section into one accelerator and builds once.
- Fail points: a corpus bump in the pin adds cases with no rows; the runner fails collection and names the missing ids.
- Order and dependencies: after T11 and T6; one PR per generator.
- Verification: `pytest -k conformance` passes in each generator with the pinned taidl; every row id exists in the corpus.
- Breaking: yes, when T22 question 2 is confirmed: generator error text gains the coverage.md clause id, released with the matching generator majors.
- Files: both `test_refusals.py`; the no-output-left rule at backend :439 and oracle :467 preserved in the runner.

## T13. Differential tests oracle vs backend on the supported intersection

- What: for each corpus case supported by both generators and carrying a `kernel.hlo` and inputs, compile the kernel with the backend, simulate the assembly on the oracle, and compare to the case reference within the case tolerance.
- What: the first cases come from `act-backend/tests/test_matching.py` selection facts (:898 dot contracting pairs, :1247 slice ranges, :1177-1189 sum from a start), `test_constructs.py:437-575`, and `test_reader.py:240-298`.
- Benefits: replaces the hand-written assertion of which instruction was selected with a check of the result, which is what the spec promises.
- Caveats: needs both generators installed in one environment, as `act/tests/test_end_to_end.py:11` already does; lives in act, not in a generator repo, and runs nightly and on release (T22 question 7).
- Fail points: a case where the oracle and the backend agree and both are wrong; T8 references guard it since the oracle is held to NumPy there.
- Fail points: a kernel the backend cannot compile is a rejection, not a mismatch; the runner distinguishes no-assembly (compared with T11 rows) from a wrong result.
- What: an `error` case passes when the oracle raises the defined error on the backend's assembly, or when the backend itself rejects the kernel with the same S id; a silent result on either side is a failure.
- What: an in-place case passes when both generators show the output in the marked input's storage, which the kernel reads back after the instruction.
- Order and dependencies: after T3 (shared kit), T12, T8; Q2; A owns the act tree.
- Verification: every case with a kernel is either differentially tested or named in an expectations row as rejected by one generator.
- Breaking: no.
- Files: `act-backend/tests/test_matching.py`, `test_constructs.py`, `test_reader.py:240-298`, `test_endings.py:620-641`; each selection fact preserved as a kernel plus expected result.

## T14. One shared end-to-end suite in act

- What: `act/tests/test_end_to_end.py` grows to run every ISA description in T10's list: generate both, compile every kernel, simulate, compare to the reference and the golden, replacing `act-backend/tests/test_mini_trn*.py` and `act-oracle/tests/test_mini_trn*.py`.
- Benefits: one place runs the real ISA descriptions; the two generator repos stop carrying examples data twice.
- Caveats: the generator repos keep one smoke test each that generates `mini_trn` and checks the tree is filled (`act-backend/tests/test_mini_trn.py:26-30`), so a generator PR still builds its own example (T22 question 4).
- Fail points: a GPU run differs from the CPU golden; the skip at `act-oracle/tests/test_mini_trn.py:22` becomes a platform-aware tolerance in T15.
- Order and dependencies: after T3 and T10; Q1 with three or four engineers; before T4 deletes the per-repo mini_trn files.
- Verification: the act suite reports one result per ISA description and kernel of T10's list; the per-repo mini_trn test files are gone.
- Breaking: no.
- Files: `act/tests/test_end_to_end.py` grows; `act-backend/tests/test_mini_trn*.py`, `act-oracle/tests/test_mini_trn*.py` merge in; reference and golden checks preserved; the `incremental` marker at `act-backend/tests/conftest.py:25-39` goes.

## T15. Golden policy: one regenerate, one record, one tolerance

- What: one `regenerate.py` in act replaces `act-backend/tests/regenerate.py` and `act-oracle/tests/regenerate.py`; the record it writes (`goldens.json` at `act-backend/tests/regenerate.py:42-52`) is the format; the reference tolerance lives beside the golden, not in a conftest.
- Benefits: a golden mismatch names the oracle, jax, platform, and machine it came from, as the record already does.
- Caveats: goldens are exact on the CPU alone; the policy states when a golden is required and when a reference tolerance suffices (T22 question 6).
- Fail points: a golden regenerated from a wrong result; the script already rejects a golden the reference rejects at :35-37, and the policy keeps that rule.
- Order and dependencies: after T14.
- Verification: one regenerate script in the four repos; every golden has a record.
- Breaking: no.
- Files: both `regenerate.py`; the record format at `act-backend/tests/regenerate.py:42-52` preserved.
