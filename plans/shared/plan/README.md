# Shared-code plan index

Sizes assume three to four engineers: S under a week, M one to three weeks, L over three weeks.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| S1 | Close the two emitter divergences | L6, O3 | S | Q1 |
| S2 | Document and pin the front-end object contract | L15, B13 | M | Q1 |
| S3 | One final validation of an ISA description | S2, L5, B21, B22 | M | Q1 |
| S4 | Shape and per-word element-type rules in the front end | S3, L3, L12, O7, B23 | L | Q2 |
| S5 | Element-type table and literal check | S2, L16, O8 | M | Q1 |
| S6 | Integer folding in the front end | S2, L6, L14 | S | Q2 |
| S7 | Conformance corpus with expected verdicts | S3, L8, T | M | Q1 |
| S8 | Coordinated v3 release train | S2, S3, S4, S5, S6, S7, A | M | Q2 |
| S9 | One copy of docker and CI scripts | A | S | Q2 |
