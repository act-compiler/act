# Owner decisions and tech-lead proposals: response

Owner decisions bind; the tech-lead proposals wait on the debate panels.
A proposal whose change also follows from a review finding on evidence is applied under that finding; otherwise nothing changes until the panel rules.

## Owner decisions

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| Decision 11: 3-4 engineers over two quarters | accepted | Binding | overhaul.md lanes; plan/README.md |
| Decision 16: no undefined behavior | accepted | Binding; each coverage row states rejected, not selected, or supported | coverage-table.md; B23 |
| Decision 17: in-place output reuses the marked input's storage | accepted | Binding; `constraint.cc:214-230` only permits overlap | coverage-table.md; B30; D14 |
| Decision 19: buffers, state, `hbm` | accepted | Binding | coverage-table.md; naming.md; glossary_objections.md; B27, B28 |
| Decision 20: region | accepted | Binding; already the plan's word | none |
| Decision 21: `@s.` stays | accepted | Binding; already the plan's spelling | none |

## Backend proposals R17-R27

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| R17 deterministic gate first | pending panel | Same change made on the architect's step-1 evidence | B29, B3 |
| R18 typed payloads, B7, B9 before B6 | pending panel | Same change made on the architect's step-4 and step-8 evidence | isel.md |
| R19 conditions as data with one applier | pending panel | Architect and critic disagree on evidence; form left open | B6 |
| R20 one cost-guided extractor | pending panel | Same change made on the architect's step-6 evidence | B10 |
| R21 in-place as a solver equality | pending panel | Same change made under owner decision 17 | B30 |
| R22 reject overlapping rewrite rules | pending panel | Objection: overlapping IR-to-ISA rules are alternatives the extractor ranks by cost, so rejecting them rejects a general plus a specialized instruction; a deterministic cost tie-break meets the stated reason | none |
| R23 keep egg; borrow rule sets, explanations, stop reasons | pending panel | Plan keeps egg; rule phases come from the critic's evidence (B32); explanations not added | B32 |
| R24 no `clang-format` at generation | pending panel | Same outcome from the critic's B12 evidence | B12 |
| R25 region count checks to the front end | pending panel | B21 unchanged | none |
| R26 rename `TensorInfo`, keep `Buffer` | pending panel | `Buffer` kept under owner decision 19; the `TensorInfo` rename was already planned | naming.md |
| R27 state in v3 Q2 after the typed model | pending panel | B16 is already Q2; no new dependency added | none |

## Other proposals touching the backend

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| R5, R58 remove `set_inplace` and placement marks | pending panel | Coverage rows state today's code only | none |
| R9, R11, R44, R57, R62 (`ROOT` placement, no-semantics instructions, clause ids, constant literals, shape module) | pending panel | No plan change depends on them; B20's float range check matches the existing integer check (`spec_checks.py:97-105`) | none |
