# Quarter 2, supporting tracks

Plan items scheduled for quarter 2 in the supporting tracks, by track.

## casestudies

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| C5 | Body-taking operations and convolution | -- | L | -- |
| C6 | Block-scaled and added element types | -- | M | -- |
| C7 | Numeric annotations | -- | M | -- |
| C8 | Data-dependent addressing | -- | L | -- |
| C12 | Re-run the case studies on v3 | C1, C11, C2, C3, C4, C5, C6, C7, C8, C9 | S | R15 |

## priorart

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| P6 | Cost-aware extraction | B10, P4 | L | -- |
| P8 | Normative execute function and reference-interpreter status | P2 | M | -- |
| P9 | Generated instruction documentation pages | P1, P2 | S | -- |
| P10 | Reusable check extractors | B13, P3 | M | -- |
| P11 | Typed hardware state kinds and data-model layout properties | B16, P8 | L | R14 |

## shared

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| S1 | Close the two emitter divergences | L6, O3 | S | -- |
| S2 | Document and pin the front-end object contract | B13, L15 | M | -- |
| S3 | One final validation of an ISA description | B21, B22, L5, S2 | M | -- |
| S4 | Shape and per-word element-type rules in the front end | B23, L12, L3, O7, S3 | L | -- |
| S5 | Element-type table and literal check | L16, O8, S2 | M | -- |
| S6 | Integer folding in the front end | L14, L6, S2 | S | -- |
| S7 | Conformance corpus with expected verdicts | L8, S3 | M | -- |
| S8 | Coordinated v3 release train | S2, S3, S4, S5, S6, S7 | M | R62 |
| S9 | One copy of docker and CI scripts | -- | S | R44 |

## diagnostics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| D4 | Move run-time-independent checks into the front end | B22, B24, D14, D3, L10, L12, L14, L16, L3, L5 | L | -- |
| D5 | Delete duplicated checks in generators | B14, B23, D4, O6, O7 | M | -- |
| D7 | Backend runtime diagnostics (Rust and C++) | B25, B26, B4, B5, D1 | L | -- |
| D8 | Oracle run-time and call diagnostics | D1, D14, L6, O11, O6 | M | -- |
| D10 | Registry coverage test and wording lint | D1, L28, T5, T6 | M | -- |
| D11 | Generate S9 and the unsupported tables | B15, D3, D6, L7, L8, O14 | S | -- |
| D12 | act umbrella: scripts and entry points | A15, A6, D1 | S | -- |
| D13 | Machine-readable output and code explanations | D1, D7, D8 | S | -- |

## semantics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| F4 | Control register semantics across both generators | F3 | L | -- |
| F5 | Numeric classes, tolerances, reduction identities | F9 | M | -- |
| F7 | Conformance suite derived from the rules | F1, F2, F3, F5 | M | -- |
