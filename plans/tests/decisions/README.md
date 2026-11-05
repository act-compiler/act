# Decisions

The decisions in force for the suite's shape: a conformance corpus in taidl, generator expectations, differential tests, and impl tests that say so.
Panel 6 (test suite) and owner decisions 1 to 33 are the sources; each decision names the item it binds.

## Files

| file | holds |
| --- | --- |
| [round1.md](round1.md) | decisions on corpus location, existing tests, goldens, and timing |
| [round2.md](round2.md) | decisions on selection tests, diagnostics, case format, kit, and examples |
| [rulings.md](rulings.md) | decisions T19 to T21 as plan items |
| [questions.md](questions.md) | T22, the answers in force |
| [justifications.md](justifications.md) | why each delete-and-rewrite is a delete, not a merge |

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| T19 | Decision: corpus lives in taidl, consumed through the pin | T5 | S | Q1 |
| T20 | Decision: impl tests survive with a header, PR-defense tests do not | - | S | Q1 |
| T21 | Decision: expectations table is the source, coverage.md is rendered | T11 | S | Q1 |
| T22 | Answers in force for the test-suite questions | - | S | Q1 |
