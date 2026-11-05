# Quarter 2, supporting tracks

Plan items scheduled for quarter 2 in the supporting tracks, by track.

## casestudies

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| C5 | Body operations: evidence that admits L20 | C12 | S | -- |
| C7 | Numeric annotations: evidence for L21 | C10 | S | -- |
| C8 | Data-dependent addressing: study that admits L23 | C11 | M | -- |
| C12 | Re-run the case studies on v3, add NVDLA | C1, C10, C11, C8, L17 | S | R15 |

## priorart

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| P8 | Normative execute function and reference-interpreter status | P2 | M | -- |
| P9 | Generated instruction documentation pages | P1, P2 | S | -- |
| P10 | Reusable check extractors | B13, P3 | M | -- |
| P11 | Alignment as a buffer property; no new state kind in v3 | B16, C11, P8 | S | R14 |

## shared

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| U1 | Close the two emitter divergences | L6, O3 | S | -- |
| U4 | Shape and per-word element-type rules in the front end | L3, U3 | L | -- |
| U6 | Integer folding in the front end | L14, L6, U2 | S | -- |
| U7 | Conformance corpus with expected verdicts | L8, T5, T6, U3 | M | -- |
| U8 | Coordinated v3 release train | A16, A31, U10, U12, U13, U14, U2, U3, U4, U5, U6, U7 | M | -- |
| U11 | Adding a generator: template generator and expectations | L40, L41, T11, T12, U10, U2, U3 | M | -- |
| U12 | Authoring surface by tracing; absorbs L42 | L43, U3 | L | -- |
| U13 | Converter and equivalence tests; absorbs L44 | U10, U12 | M | -- |
| U14 | Delete the string surface and the parser dependency; absorbs L45 | B13, O2, T37, U10, U12, U13 | S | -- |

## diagnostics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| D4 | Move run-time-independent checks into the front end | B24, D14, D3, L10, L14, L16, L3, L5, U4 | L | -- |
| D5 | Delete duplicated checks in generators | B14, D4, O6, U10 | M | -- |
| D7 | Backend runtime diagnostics (Rust and C++) | B25, B26, B4, B5, D1 | L | -- |
| D11 | Generate S9 and the unsupported tables | B15, D3, D6, L7, L8, O14 | S | -- |
| D12 | act umbrella: scripts and entry points | A15, A6, D1 | S | -- |
| D15 | Removed-names rejection | D1, L15, U2 | S | -- |
| D16 | Failure policy (decision 40) | B2, D1, D18, D7, D8 | M | -- |
| D17 | Rejections at the author's line | D1, D2, U12 | S | -- |
| D18 | Exclusion report | B46, D1, D2, D6 | S | -- |

## semantics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| F4 | State variable semantics across both generators | F3 | L | -- |
| F5 | Numeric classes, bounds, reduction identities | F6 | M | -- |
