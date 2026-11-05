# Prior-art plan items

Sized for three to four engineers, two quarters, one coordinated major release (TAIDL v3).
B, O, L, T, F, D, U, A and C IDs are other tracks' items.
P3, P4, P10, and P12 are backend refactors whose value depends on the B items they name.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| P1 | Numbered per-operation constraints in S8 | L | M | Q1 |
| P2 | Three-word taxonomy (rejected, fault, defined value) for every S1.3 item | P1, L | S | Q1 |
| P3 | Named typed rewrite rules with priorities | B6, B8 | M | Q1 |
| P4 | Phased saturation schedule | P3, B6 | M | Q1 |
| P5 | Skipped-pattern and stop-reason reports | P3, D | S | Q1 |
| P6 | Cost-aware extraction | P4, B10 | L | after v3 |
| P7 | Executable examples per spec row | P1, P2, T | M | Q1 |
| P8 | Normative execute function and reference-interpreter status | P2, O, F | M | Q2 |
| P9 | Generated instruction documentation pages | P1, P2, A | S | Q2 |
| P10 | Reusable check extractors | P3, B13 | M | Q2 |
| P11 | Alignment as a buffer property; no new state kind in v3 | P8, B16, L, C11 | S | Q2 |
| P12 | Named and oracle-tested expansion rules | P3, P7 | S | Q1 |
