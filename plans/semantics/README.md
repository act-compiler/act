# TAIDL formal core

A compact formal backbone for the TAIDL spec: abstract syntax, static and dynamic semantics, the holes found between the spec and its two generators, and the plan to close them.
Terms follow `/home/user/work/terminology`: tensor buffer (one data model entry), control register (state variable), access dimensions, region, memory state.

| file | contents |
| --- | --- |
| [syntax.md](syntax.md) | Abstract syntax of an ISA description, each production tied to its grammar or front-end line |
| [typing.md](typing.md) | Well-formedness and typing judgments for declarations, expressions, regions, and blocks |
| [typing-ops.md](typing-ops.md) | Shape and element type rule per operation family, with no implicit broadcast or promotion |
| [dynamics.md](dynamics.md) | Issue of an instruction as a function on memory state and control registers, numeric classes, determinism |
| [holes.md](holes.md) | Nineteen semantic holes with evidence, both generators' readings, and a ruling each |
| [rulings.md](rulings.md) | Explicit rulings on in-place marks, reduction initial values, and out-of-bounds regions |
| [review.md](review.md) | What the current design gets right |
| [plan/README.md](plan/README.md) | Plan items with dependencies, size, and quarter |
| [plan/F1.md](plan/F1.md) | Static shape and element type checker in the front end |
| [plan/F2.md](plan/F2.md) | Region aliasing rules and in-place mark meaning |
| [plan/F3.md](plan/F3.md) | Total integer arithmetic: overflow and zero divisors reject the issue |
| [plan/F4.md](plan/F4.md) | Control register semantics across both generators |
| [plan/F5.md](plan/F5.md) | Numeric classes, tolerances, reduction identities |
| [plan/F6.md](plan/F6.md) | Byte order, sub-byte packing, size of d0 |
| [plan/F7.md](plan/F7.md) | Conformance suite derived from the rules |
| [plan/F8.md](plan/F8.md) | Front-end rejections the spec already implies |
| [plan/F9.md](plan/F9.md) | Executable reference interpreter of the exact classes |
