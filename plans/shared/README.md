# Shared code and generator interface

The interface between the TAIDL front end and its two generators, the logic they duplicate, and the plan to share it.

- [interface.md](interface.md): every front-end member each generator reads, with `file:line`, and where generators re-derive facts.
- [duplication.md](duplication.md): duplicated logic across repos, its copies, how they differ, and the best copy.
- [review.md](review.md): what works across the repos and should stay.
- [bugs.md](bugs.md): divergences between copies that produce wrong results, X1 to X10.
- [overhaul.md](overhaul.md): target split per module and the ordered steps.
- [decisions.md](decisions.md): the consolidate-or-decouple debate, positions and rulings SD1 to SD9.
- [plan/README.md](plan/README.md): plan items U1 to U11 with dependencies, size and quarter.
- [plan/U1.md](plan/U1.md): close the two emitter divergences.
- [plan/U2.md](plan/U2.md): typed front-end contract with a version.
- [plan/U3.md](plan/U3.md): one final validation that publishes derived facts.
- [plan/U4.md](plan/U4.md): shape and per-word element-type rules in the front end.
- [plan/U5.md](plan/U5.md): element-type table and literal check.
- [plan/U6.md](plan/U6.md): integer folding in the front end.
- [plan/U7.md](plan/U7.md): conformance corpus with expected verdicts.
- [plan/U8.md](plan/U8.md): coordinated v3 release train.
- [plan/U9.md](plan/U9.md): one copy of docker and CI scripts.
- [plan/U10.md](plan/U10.md): generators call `validate()` first.
- [plan/U11.md](plan/U11.md): adding a generator, template generator and expectations.
