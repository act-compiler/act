# Plans

Plans for the next two quarters of TAIDL, act-backend, act-oracle, and act, from one engineering planning meeting.
Each folder has one README.md index.
Plan items carry a track prefix and are referenced by ID only.

## Goals

- One unambiguous, generic TAIDL specification with no undefined behavior.
- Generators that follow the specification, each with a table of what it supports.
- One vocabulary across all repos, grounded in the ACT and TAIDL papers, then StableHLO and MLIR.
- Simpler front end, backend, and oracle, built from the current code rather than rewritten.
- A test suite derived from the specification instead of from individual pull requests.

## Folders

| Folder | Contents | Item prefix |
| ------ | -------- | ----------- |
| [roadmap/](roadmap/README.md) | Cross-repo order, quarters, load, and dependency checks | -- |
| [minutes/](minutes/README.md) | Owner decisions and tech-lead rulings | R |
| [language/](language/README.md) | Draft specification, extension proposals, language plan | L |
| [backend/](backend/README.md) | act-backend coverage table, review, bugs, overhaul, plan | B |
| [oracle/](oracle/README.md) | act-oracle coverage table, review, bugs, overhaul, plan | O |
| [tests/](tests/README.md) | Test inventory, conformance corpus, migration | T |
| [act/](act/README.md) | Umbrella repo: docs, tutorials, versions, docker | A |
| [semantics/](semantics/README.md) | Formal core: typing, dynamics, semantic holes | F |
| [shared/](shared/README.md) | Front-end to generator interface and duplicated logic | S |
| [diagnostics/](diagnostics/README.md) | Error-message catalog and one diagnostic scheme | D |
| [casestudies/](casestudies/README.md) | Ten accelerators sketched in TAIDL and the gap matrix | C |
| [priorart/](priorart/README.md) | What to borrow from Sail, ILA, Exo, TableGen, ISLE, egg, StableHLO, NKI | P |
| [terminology/](terminology/README.md) | Terms from the papers, MLIR, StableHLO, PyTorch, and NKI | -- |
| [glossary/](glossary/README.md) | Reconciled glossary | -- |
| [reviews/](reviews/README.md) | Independent reviews of the backend, oracle, and spec plans | -- |

## Plan item shape

- What, Benefits, Caveats, Fail points, Order and dependencies, Verification.
- Size is S, M, or L; quarter is Q1 or Q2, sized for 3-4 engineers.
- An item marked breaking lands in the one coordinated v3 release.
