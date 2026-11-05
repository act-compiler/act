# Norm catalogue and coverage report

T26 maps every run-time norm of S1.3 to its cases and reports what the corpus leaves uncovered.

## T26. Norm catalogue and coverage report

- What: `taidl/conformance/norms.md` is the S1.3 table (`overview.md` S1.3.1 to S1.3.17) plus a `cases` column and one expectation per generator.
- What: cases exist for every S1.3 row from S1.3.1 to S1.3.17.
- What: S1.3.6 (input region overlapping an output region outside its declared relation) is covered by the T30 matrix; its fault clause is S1.3.5.
- What: S1.3.12 and S1.3.17 (integer divide and remainder by 0, `min / -1`, out-of-range or NaN `convert`) hold fixed-value cases and no fault case, since tensor data never faults; only S1.3.2 (a computed divisor of 0 in an expression) is a fault.
- What: S1.3.15 (float accumulation of `dot`, `reduce_add`, `reduce_multiply`) gets cases that state the result-type bound of S8.4 (`gamma_n * sum|terms| + u * |exact|`; `gamma_n * |exact|` for `reduce_multiply`), plus a `bf16` reduction past the vacuous length written with a wider result type.
- What: a `pending` row names the deciding panel and has no case; a pending row fails the v3 release gate, since every violation needs a verdict.
- What: a case with several allowed results lists them in `case.toml`; a case with no expectation tests nothing and is not accepted.
- What: the seeds at `act-oracle/tests/test_kernel.py:712-720` and :749-757 move to oracle `cli` tests.
- What: `taidl/conformance/coverage.md` is rendered per S id, S9 clause, and norm row.
- What: CI fails on a zero row outside an allowlist (S1.1, S1.2, S10, pending rows).
- What: an `xfail` row exists only for a case the front end accepts.
- Benefits: a clause with no case is visible; a per-section count would hide it.
- Caveats: the norm rows are L's; a renumbered S1.3 id changes the table, so ids are renumbered only with a note in `CHANGES.md`.
- Fail points: a pending row hides a missing verdict; the release gate fails on it.
- Order and dependencies: after T5, T6, T30, and D14 (run-time violation table).
- Verification: every S1.3 row has cases or a pending mark; the renderer output is byte-identical in CI.
- Breaking: no.
- Files: new `taidl/conformance/norms.md` and `coverage.md`; no existing file changed.
