# Issue log: act-glossary

Source is `reviews/act-glossary/review.md` (AG1 to AG21).
The act, glossary, and terminology tracks have no review-response file; each row was checked against `act/plan/`, `glossary/`, `terminology/`, `minutes/rulings.md`, and panel 7.
Severity is the review's own label.
Panel questions are cited as Qn.k and owner decisions as ODn.

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| AG1: three glossary files still carry terms that OD19 to OD21 override | blocker | fixed | Q7.2; glossary/terms.md; glossary/words-to-avoid.md (generated from the banned column); glossary/rulings.md |
| AG2: competing rulings in terminology, spec glossary, and naming todos | major | fixed | Q7.2; terminology/README.md (notes only, never set a term); language/spec-draft/glossary.md |
| AG3: `add_buffer` and `add_state` names and the singular of state unfixed | major | fixed | Q7.6; glossary/terms.md |
| AG4: glossary renames `hbm` to `global_memory` against OD19 | major | fixed | Q7.7; glossary/terms.md (hbm row) |
| AG5: five `update` renames rest on the removed control register term | major | fixed | Q7.10; glossary/terms.md (state update) |
| AG6: no "fault"; "undefined behavior" and "implementation-defined" remain | major | fixed | Q7.8; glossary/terms.md (fault, rejection rows) |
| AG7: in-place mark and placement mark rows against R5 and R58 | major | fixed | Q7.9, OD30; glossary/terms.md; glossary/rulings.md |
| AG8: "a named integer" too narrow; R27 says "State registers" | minor | fixed | glossary/terms.md (state variable row); minutes/rulings.md (R27) |
| AG9: short form "kernel" for two terms | minor | fixed | glossary/terms.md (kernel function row) |
| AG10: identifier lists cover only front-end files | major | fixed | Q7.10; glossary/terms.md (identifier column); act/plan/terminology-docs.md (A8) |
| AG11: stale line cites in the identifier column | minor | fixed | Q7.10 (symbols, not line numbers) |
| AG12: A8 states no target names; no TAIDL-TO or ACT-Backend entry | major | fixed | Q7.11; act/plan/terminology-docs.md (A8, size M); glossary/terms.md |
| AG13: A9, A10, A12 target a pre-v3 tutorial, against OD6 | blocker | fixed | Q7.1; act/plan/tutorials.md (A9, A11, A24) |
| AG14: no act item for the v3 release | major | fixed | Q7.3; act/plan/versions-release.md (A23) |
| AG15: install from a pinned tag is never tested | major | fixed | Q7.4; act/plan/tests-docker.md (A26) |
| AG16: no quickstart | major | fixed | Q7.5; act/plan/terminology-docs.md (A25); act/plan/tests-docker.md (A20) |
| AG17: A16 and A17 duplicate the release train; v3 numbers undefined | minor | fixed in part: the v3 numbers are stated and A23 added; A16 stays in wave 1 per Q8.5 and OD33 | act/plan/versions-release.md; Q8.5 |
| AG18: A19 covers state variables and in-place marks | minor | fixed | act/plan/tests-docker.md (A19 retitled) |
| AG19: pointers lead to the second tree `/home/user/work/` | minor | fixed | semantics/README.md; diagnostics/scheme.md; diagnostics/catalog/taidl-act.md |
| AG20: over-engineered items A6, A13, A14, A18, A22 | minor | fixed: A14 is part of A24, A22 part of A11, and A6, A13, and A18 land after v3 | act/plan/tutorials.md; act/plan/terminology-docs.md; act/plan/versions-release.md; act/plan/tests-docker.md |
| AG21: R53 names terms that were not added | minor | fixed | minutes/rulings.md (R53) |
