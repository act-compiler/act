# Shared code and generator interface

The interface between the TAIDL front end and its two generators, the logic they duplicate, and the plan to share it.

- [interface.md](interface.md): every front-end member each generator reads, with `file:line`, and where generators re-derive facts.
- [duplication.md](duplication.md): duplicated logic across repos, its copies, how they differ, and the best copy.
- [review.md](review.md): what works across the repos and should stay.
- [bugs.md](bugs.md): divergences between copies that produce wrong results, X1 to X10.
- [overhaul.md](overhaul.md): target split per module and the ordered steps.
- [decisions.md](decisions.md): the consolidate-or-decouple debate, positions and rulings SD1 to SD9.
- [plan/README.md](plan/README.md): plan items S1 to S9 with dependencies, size and quarter.
- [plan/S1.md](plan/S1.md): close the two emitter divergences.
- [plan/S2.md](plan/S2.md): document and pin the front-end object contract.
- [plan/S3.md](plan/S3.md): one final validation of an ISA description.
- [plan/S4.md](plan/S4.md): shape and per-word element-type rules in the front end.
- [plan/S5.md](plan/S5.md): element-type table and literal check.
- [plan/S6.md](plan/S6.md): integer folding in the front end.
- [plan/S7.md](plan/S7.md): conformance corpus with expected verdicts.
- [plan/S8.md](plan/S8.md): coordinated v3 release train.
- [plan/S9.md](plan/S9.md): one copy of docker and CI scripts.
