# Plan index

Sizes assume three to four engineers: S is under a week, M one to three weeks, L more than three weeks.
The release is one coordinated major release (TAIDL v3); every item that changes a message or a rejection is breaking and carries a migration note.
Other tracks are named by prefix: L language spec, B backend, O oracle, T tests, A act umbrella.
Shared code items are U1 to U15; case studies (C), prior art (P) and formal semantics (F) carry no dependency of a D item.
"O20 or O11" names the unroll outcome: O20 lands when O19 meets the Q5.1 bound, otherwise O11 lands, and exactly one of the two does (oracle/plan/overhaul.md).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| D1 | [Registry, classes and message format](D01-registry-and-format.md) | L1, L25 | M | Q1 |
| D2 | [Call sites and exception classes in the front end](D02-call-site-and-exceptions.md) | D1, U3 | M | Q1 |
| D3 | [Recode and reword the front-end messages](D03-recode-front-end.md) | D1, D2, L26 | M | Q1 |
| D14 | [Fault table and run-time norm](D14-run-time-norm.md) | L5, L6 | S | Q1 |
| D6 | [Unsupported-feature registries](D06-unsupported-registries.md) | D1, L7, B15, O14, O17 | M | Q1 |
| D8 | [Oracle run-time and call diagnostics](D08-oracle-runtime.md) | D1, D14, L6, O6, O20 or O11 | M | Q1 |
| D9 | [Silent fallbacks become errors](D09-silent-fallbacks.md) | D1, B24, B26, L5 | S | Q1 |
| D10 | [Registry coverage test and wording lint](D10-tests-and-lint.md) | D1, T5, T6 | M | Q1 |
| D4 | [Move run-time-independent checks into the front end](D04-move-checks-to-front-end.md) | D14, D3, U4, L3, L5, L10, L14, L16, B24 | L | Q2 |
| D5 | [Delete duplicated checks in generators](D05-delete-generator-copies.md) | D4, U10, B14, O6 | M | Q2 |
| D7 | [Backend runtime diagnostics (Rust and C++)](D07-backend-runtime.md) | D1, B4, B5, B25, B26 | L | Q2 |
| D11 | [Generate S9 and the unsupported tables](D11-generated-docs.md) | D3, D6, L7, L8, O14, B15 | S | Q2 |
| D12 | [act umbrella: scripts and entry points](D12-act-scripts.md) | D1, A6, A15 | S | Q2 |
| D13 | [Machine-readable output and code explanations](D13-json-and-explain.md) | D1, D7, D8 | S | after v3 |
| D15 | [Removed-names rejection](D15-removed-names.md) | D1, U2, L15 | S | Q2 |
| D16 | [Failure policy (decision 40)](D16-failure-policy.md) | D1, D7, D8, D18, B2 | M | Q2 |
| D17 | [Rejections at the author's line](D17-author-line-rejections.md) | D1, D2, U12 | S | Q2 |
| D18 | [Exclusion report](D18-exclusion-report.md) | D1, D2, D6, B46 | S | Q2 |

## Staffing sketch

- Engineer 1 (front end): D1, D2, D3, then D4.
- Engineer 2 (oracle and tests): D14, D8, D10, then D5 for the oracle.
- Engineer 3 (backend): D6, D9, then D7 and D5 for the backend.
- Engineer 4, if present: D11, D12, D13, D15, D17, D18 and the migration notes.

## Critical path

- D1 then D2 then D3 then D4 then D5.
- D4 waits for U4 and the language items L3, L5, L10, L14, L16; slip there moves D4 to the end of Q2.
- D17 follows the tracer U12; D18 follows B46 and D6.
