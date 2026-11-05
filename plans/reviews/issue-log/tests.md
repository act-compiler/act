# Issue log: tests

Sources are `reviews/tests/qa-lead.md` (QA1 to QA22) and `reviews/tests/industry.md` (IN1 to IN16).
The tests track has no review-response file; each row was checked against `tests/` as it stands and against panel 6.
Severity is the review's own label.
Panel questions are cited as Qn.k and owner decisions as ODn.

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| QA1: run outcomes attach only to `kernel.hlo`, so instruction-level faults cannot be expressed | blocker | fixed | Q6.1; tests/corpus/format.md (`program.toml`), T29 |
| QA2: backend in-place check is ill-posed on value-semantic HLO; seeds use removed marks | blocker | fixed in part; the identical-region-text aliasing of R5 is superseded by OD22 and Q6.2 | Q6.2; tests/corpus/overlap.md (T30), tests/generators.md (T13) |
| QA3: an `error` case fails T13 on a silent result, against R4 and R10 | major | fixed | Q6.3; tests/generators.md (never-faults invariant), T29 |
| QA4: no catalogue from each defined fault to its cases | major | fixed | Q6.4; tests/corpus/norms.md (T26) |
| QA5: spec coverage reported, not enforced | major | fixed | Q6.7; tests/corpus/norms.md (zero-row CI failure) |
| QA6: migration deletes `LOWERED` and word loops before their runner exists | major | fixed | Q6.8; tests/migration.md (steps 8 and 9, deletion rules) |
| QA7: per-PR backend coverage moves to nightly with no per-PR replacement | major | fixed | Q6.8; tests/generators.md (T12 slow tier per PR) |
| QA8: T4 deletes a range that holds the one e-graph mark test | major | fixed | Q6.8; tests/inventory/cleanup.md (T4 keeps `test_generators.py:243`) |
| QA9: T7 loses parse-tree assertions | major | fixed | Q6.5; tests/corpus/format.md (`parsed` table) |
| QA10: no fuzzing of the front end | major | fixed | Q6.10; tests/corpus/extensions.md (T24) |
| QA11: no mutation testing | major | fixed | Q6.10, OD28; tests/corpus/extensions.md (T25) |
| QA12: no CI time budget per tier | major | fixed | Q6.12; tests/generators.md (T27) |
| QA13: nondeterminism rule is only the Hypothesis seed | major | fixed | tests/generators.md (T27, deterministic mode B29, quarantine rule) |
| QA14: T8 forbids the StableHLO interpreter that R29 rules as reference | major | superseded by OD28 and Q6.6: no interpreter, stored NumPy or closed-form data | tests/corpus/seeding.md (T8), tests/corpus/format.md |
| QA15: no golden update policy | minor | fixed | Q6.12; tests/generators.md (T15) |
| QA16: `test_linalg.py` tests the three words v3 removes | minor | fixed | tests/corpus/seeding.md (rejected cases) |
| QA17: an implementation-defined case with no expectation tests nothing | minor | fixed | Q6.4; tests/corpus/norms.md (allowed results listed) |
| QA18: rows for cases the front end rejects, no `partial` rendering | minor | fixed | Q6.7; tests/corpus/norms.md (`xfail` rows only for accepted cases) |
| QA19: inventory counts do not follow the verdicts | minor | fixed | tests/review.md (counts, 503 tests); tests/inventory/ (one verdict word set, `reshape`) |
| QA20: TB5 is wrong, the test asserts through `check_oracle` | minor | fixed | tests/bugs.md (TB5 retired) |
| QA21: inventory omits lint scripts, `cost/*.c`, `pytest.sh` runners, `plain_dir` | minor | fixed | tests/inventory/ (script and fixture rows); tests/inventory/cleanup.md (T2 names `plain_dir`) |
| QA22: debate rounds are decision history; T19 to T21 restate rulings | minor | fixed: `tests/decisions/round1.md` and `round2.md` are renamed `layout.md` and `cases.md`; T19 to T21 stay as scheduled items of the wave plan (Q8.4) | tests/decisions/README.md |
| IN1: one `rejected` verdict for every phase | major | fixed | Q6.5; tests/corpus/format.md (`phase`) |
| IN2: error matched by prose regex | major | fixed | Q6.5, Q3.9; tests/corpus/format.md (`code`, `clause`, `object`) |
| IN3: coverage per section, not per clause | major | fixed | Q6.7; tests/corpus/norms.md (T26 `coverage.md`) |
| IN4: reference library contradicts R29 | major | superseded by OD28 and Q6.6 | tests/corpus/seeding.md (T8) |
| IN5: no row kind for a known defect | major | fixed | Q6.7; tests/corpus/format.md (`xfail`) |
| IN6: T5 blocks on L's clause ids | major | fixed | Q6.5; tests/corpus/format.md (`clause = "pending"`) |
| IN7: fault case checks only the raise | major | fixed | Q6.3; tests/corpus/format.md (`expect = "fault"`), T29 (unchanged state) |
| IN8: one tolerance per type | minor | fixed | Q6.6; tests/corpus/format.md (`exact` or `almost` with `ulps`) |
| IN9: NaN and signed-zero comparison unstated | minor | fixed | Q6.6; tests/corpus/format.md (`nan = "any"`) |
| IN10: directory names encode section numbers | minor | fixed | Q6.11; tests/corpus/format.md (slug names) |
| IN11: no `origin` field in `case.toml` | minor | fixed | tests/corpus/format.md (T5 `origin`); tests/migration.md (T18 greps it) |
| IN12: no one-line `description` in `case.toml` | minor | fixed | tests/corpus/format.md (T5 `description`) |
| IN13: R17 gate needs assembly check lines | minor | fixed | tests/generators.md (assembly facts as check lines) |
| IN14: real ISA descriptions run with no conditioning | minor | fixed | Q6.12; tests/generators.md (T14 skip and report) |
| IN15: Python case scripts only a Python front end can run | minor | fixed | Q6.11; tests/corpus/format.md (text files; `spec.py` waits on L) |
| IN16: no report of case ids added or removed by a pin bump | minor | fixed | Q6.11; tests/corpus/format.md (`CHANGES.md`, `check_versions.py`) |
