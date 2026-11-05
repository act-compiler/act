# act-oracle plan

Planning for TAIDL-TO, the act-oracle generator, over the next two quarters.
Item IDs carry the prefix `O`; other tracks are named by their prefix: L language spec, B backend, T tests, A act umbrella.
Sizes assume 3-4 engineers: S is under a week, M one to three weeks, L more than three weeks.
Each item states whether it breaks compatibility, for one coordinated major release with a migration note.

| File | Contents |
| --- | --- |
| [review.md](review.md) | The oracle as it stands: what to keep, what is weak, a verdict per component. |
| [bugs.md](bugs.md) | Bugs and defects D1-D24, one row each with severity, `file:line`, fix sketch and item. |
| [overhaul.md](overhaul.md) | Target structure per module and the ordered steps from today. |
| [decisions.md](decisions.md) | Positions argued and the decisions taken, each with its evidence. |
| [architecture.md](architecture.md) | Items O19, O20, O1, O2, O3, O18: unroll spike and adoption, runtime layout, record, emitter, splitter. |
| [lowering.md](lowering.md) | Items O4, O9, O21, O22: word table, `f16` solve, excess precision, numerics norms. |
| [checks.md](checks.md) | Items O5, O6, O7, O24: dead code, error classes, front-end checks, call-time errors. |
| [fidelity.md](fidelity.md) | Items O8, O10, O11, O12, O13, O23: literals, in-place, wrapped starts, jax configuration, reference, in-module errors. |
| [coverage.md](coverage.md) | Items O14, O17: fold `docs/operations.md`, test the coverage table. |
| [naming.md](naming.md) | Items O15, O16 and the naming table. |
