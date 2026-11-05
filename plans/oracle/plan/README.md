# act-oracle plan

Planning for TAIDL-TO, the act-oracle generator, over the next two quarters.
Item IDs carry the prefix `O`; other tracks are named by their prefix: L language spec, B backend, T tests, A act umbrella.
Sizes assume two engineers: S is under a week, M one to three weeks, L more than three weeks.
Each item states whether it breaks compatibility, for one coordinated major release with a migration note.

| File | Contents |
| --- | --- |
| [review.md](review.md) | The oracle as it stands: what to keep, what is weak, a verdict per component. |
| [bugs.md](bugs.md) | Bugs and defects, one row each with severity, `file:line` and fix sketch. |
| [overhaul.md](overhaul.md) | Target structure per module and the ordered steps from today. |
| [decisions.md](decisions.md) | The architecture debate: positions, second round, rulings, replacement justification. |
| [architecture.md](architecture.md) | Items O1, O2, O3, O18: module layout, line record, templates, line splitter. |
| [lowering.md](lowering.md) | Items O4, O9: one word table, `triangular_solve` at `f16`. |
| [checks.md](checks.md) | Items O5, O6, O7: dead code, error classes, shape rules to the front end. |
| [fidelity.md](fidelity.md) | Items O8, O10, O11, O12, O13: literal rounding, in-place marks, wrapped starts, jax configuration, reference interpreter. |
| [coverage.md](coverage.md) | Items O14, O17: fold `docs/operations.md`, test the coverage table. |
| [naming.md](naming.md) | Items O15, O16 and the naming table: identifiers, terms, log lines. |
