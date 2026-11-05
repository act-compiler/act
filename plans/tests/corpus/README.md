# Conformance corpus

The corpus is a set of small TAIDL programs, valid and invalid, each tagged with the spec section it exercises and the front-end verdict the spec gives it.
It lives in the taidl repo beside the spec, and the generators consume it through the taidl version they pin (`act-backend/pyproject.toml:16`, `act-oracle/pyproject.toml:16`).
Items T5 to T10 build the corpus; T11 to T15 say how each generator runs it.

## Files

| file | holds |
| --- | --- |
| [format.md](format.md) | T5 case format and T6 front-end verdict runner |
| [seeding.md](seeding.md) | T7 seeding from existing tests and T8 operations corpus |
| [extensions.md](extensions.md) | T9 property-based tests and T10 real accelerator list |
| [sections.md](sections.md) | how each outline section S1 to S10 is covered by the corpus |

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| T5 | Corpus format, location, and naming | L (S1, S9) | S | Q1 |
| T6 | Front-end verdict runner in taidl | T5 | M | Q1 |
| T7 | Seed the corpus from existing front-end tests (S2 to S7) | T1, T5, T6 | L | Q1 |
| T8 | Operations corpus with reference values (S8) | T5, L (S8.4) | L | Q1 |
| T9 | Property-based tests for shapes, element types, and names | T6, T8, T22 | M | Q2 |
| T10 | Real ISA descriptions as the end-to-end list | T5, A | S | Q1 |
