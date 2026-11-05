# Plan index

Sizes assume three to four engineers: S is under a week, M one to three weeks, L more than three weeks.
The release is one coordinated major release (TAIDL v3); every item that changes a message or a rejection is breaking and carries a migration note.
Other tracks are named by prefix: L language spec, B backend, O oracle, T tests, A act umbrella.
Shared code items are U1 to U10; case studies (C), prior art (P) and formal semantics (F) carry no dependency of a D item.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| D1 | Registry, classes and message format | L1, L25 | M | Q1 |
| D2 | Call sites and exception classes in the front end | D1, U3 | M | Q1 |
| D3 | Recode and reword the front-end messages | D1, D2, L26 | M | Q1 |
| D14 | Fault table and run-time norm | L5, L6 | S | Q1 |
| D6 | Unsupported-feature registries | D1, L7, B15, O14, O17 | M | Q1 |
| D8 | Oracle run-time and call diagnostics | D1, D14, L6, O6, O11 | M | Q1 |
| D9 | Silent fallbacks become errors | D1, B24, B26, L5 | S | Q1 |
| D10 | Registry coverage test and wording lint | D1, T5, T6 | M | Q1 |
| D4 | Move run-time-independent checks into the front end | D14, D3, U4, L3, L5, L10, L14, L16, B24 | L | Q2 |
| D5 | Delete duplicated checks in generators | D4, U10, B14, O6 | M | Q2 |
| D7 | Backend runtime diagnostics (Rust and C++) | D1, B4, B5, B25, B26 | L | Q2 |
| D11 | Generate S9 and the unsupported tables | D3, D6, L7, L8, O14, B15 | S | Q2 |
| D12 | act umbrella: scripts and entry points | D1, A6, A15 | S | Q2 |
| D13 | Machine-readable output and code explanations | D1, D7, D8 | S | Q2 |
| D15 | Removed-names rejection | D1, U2, L15 | S | Q2 |
| D16 | Failure policy (decision 40) | D1, D7, D8, B2 | M | Q2 |

## Staffing sketch

- Engineer 1 (front end): D1, D2, D3, then D4.
- Engineer 2 (oracle and tests): D14, D8, D10, then D5 for the oracle.
- Engineer 3 (backend): D6, D9, then D7 and D5 for the backend.
- Engineer 4, if present: D11, D12, D13, D15 and the migration notes.

## Critical path

- D1 then D2 then D3 then D4 then D5.
- D4 waits for U4 and the language items L3, L5, L10, L14, L16; slip there moves D4 to the end of Q2.
