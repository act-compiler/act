# Conformance corpus

The corpus is a set of small TAIDL programs, valid and invalid, each tagged with the section of the language specification it exercises and the verdict the specification gives it.
It lives in the taidl wheel beside the language specification, and the generators consume it through the taidl version they pin (`act-backend/pyproject.toml:16`, `act-oracle/pyproject.toml:16`).
Items T5 to T10, T24 to T26, T29, T30, and T32 to T36 build the corpus; T11 to T15 say how each generator runs it.
The shared-code item U7 (corpus with expected verdicts) lands through T5 to T7, T11, and T12.

## Files

| file | holds |
| --- | --- |
| [format.md](format.md) | T5 case format, T6 front-end verdict runner, T29 program runner |
| [seeding.md](seeding.md) | T7 seeding from existing tests and T8 operations corpus |
| [extensions.md](extensions.md) | T9 property tests, T10 real accelerator list, T24 fuzzer, T25 mutation pilot |
| [overlap.md](overlap.md) | T30 overlap matrix, rejection, and cross-check cases |
| [layers.md](layers.md) | T32 width layer, T33 encoding layer and assembler round trip, T34 derived attributes, T35 generator metadata |
| [failure.md](failure.md) | T36 failure policy (decision 40) |
| [norms.md](norms.md) | T26 norm catalogue and coverage report |
| [sections.md](sections.md) | how each outline section S1 to S10 is covered by the corpus |

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| T5 | Corpus format, location, and naming | L1 | S | Q1 |
| T6 | Front-end verdict runner in taidl | T5, D3 | M | Q1 |
| T7 | Seed the corpus from existing front-end tests (S2 to S7) | T1, T5, T6 | L | Q1 |
| T8 | Operations corpus with reference values (S8) | T5, L3 | L | Q1 |
| T9 | Property-based tests for shapes, element types, and names | T6, T8 | M | Q2 |
| T10 | Real ISA specifications as the end-to-end list | T5 | S | Q1 |
| T24 | Nightly grammar fuzzer | T6, T9 | M | Q2 |
| T25 | Mutation pilot over front-end and oracle fault checks | T6, T12 | S | Q2 |
| T26 | Norm catalogue and coverage report | T5, T6, D14 | M | Q1 |
| T29 | Program runner over an oracle interface | T5, T6, D8 | M | Q1 |
| T30 | Overlap matrix, rejection, and cross-check cases | T5, T6, T29, D8 | M | Q1 |
| T32 | Width layer cases (range faults, `u64` rejection) | T5, T6, T29, L37, D16 | M | Q1 |
| T33 | Encoding layer cases and assembler round trip | T32, T6, T29, L38, L39 | L | Q2 |
| T34 | Derived attribute cases (closed-form links, rejected links) | T33 | M | Q2 |
| T35 | Generator metadata cases (neutrality, namespaces) | T5, T6, T12, U2, L40 | M | Q1 |
| T36 | Failure policy cases (decision 40) | T12, T13, T27, T29, D16 | M | Q2 |
