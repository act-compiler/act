# Decisions

The target design was argued between a pragmatist position (minimal change, keep what works) and a conformance-purist position (spec-derived suite, delete PR-defense tests), with the track lead adjudicating.
No sub-agent tool was available in the planning session, so both positions were written by the lead from the inventory and are recorded for the owner to overrule.
Round one settled the shape of the suite; round two took the points still contested and ruled on each.

## Files

| file | holds |
| --- | --- |
| [round1.md](round1.md) | first round: the two positions per topic and the ruling |
| [round2.md](round2.md) | second round on the contested points, with the ruling |
| [rulings.md](rulings.md) | decisions T19 to T21 as plan items |
| [questions.md](questions.md) | T22, open questions for the owner with defaults |
| [justifications.md](justifications.md) | why each delete-and-rewrite is a delete, not a merge |

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| T19 | Decision: corpus lives in taidl, consumed through the pin | T5 | S | Q1 |
| T20 | Decision: impl tests survive with a header, PR-defense tests do not | T18 | S | Q2 |
| T21 | Decision: expectations table is the source, coverage.md is rendered | T11 | S | Q1 |
| T22 | Open questions for the owner | - | S | Q1 |
