# Issue log: rulings

Source is `reviews/rulings/audit.md`: nineteen questionable or wrong rulings (R-numbers) and thirteen unsettled questions (RA1 to RA13).
Rulings have no review-response file; the audit fed the panel agendas, so each row was checked against the panel decisions, `meeting/resolutions.md`, and the track plans.
`minutes/rulings.md` holds draft answers that the process never makes binding, and its text is not edited; the decision that replaced a ruling is the landing place.
Severity maps the audit's verdict: wrong is a blocker, questionable a major, and an unsettled question takes its own weight.
The backend, oracle, and language responses also answered some of these rulings (R5, R20 to R22, R26, R28, R29, R58, R62); their rows are logged here and not in the track files.
Panel questions are cited as Qn.k and owner decisions as ODn.

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| R2: "reconciled glossary" points at renames OD19 forbids | major | fixed | Q7.2; glossary/rulings.md (renames struck), glossary/README.md |
| R4: "the backend may assume it never happens" is undefined-behavior wording | major | fixed | Q1.12 (a producer never emits a faulting issue); Q4.9 |
| R5: aliasing by identical region text, `set_inplace` removed | major | superseded by OD22, OD30, and Q9.2: the mark stays and each pair has one relation | semantics/rulings.md; B30, B39; O10; D33 |
| R7: constraints and starts read the post-update state | major | fixed | Q1.1 (one view, the pre-state); S5.3, S5.4, S5.6 |
| R8: accumulation clause conflicts with the tolerance clause | major | fixed | Q1.5 (two Higham bounds, no accumulation clause); S1.3.15 |
| R10: "the backend may assume it holds" | major | fixed | Q1.12; Q4.9 |
| R12: every buffer starts zero-filled, including `d0` | major | fixed | Q1.4 and the joint decision on unwritten reads; B36; S5.6 |
| R14: typed state kinds and async groups in v3 | major | superseded by Q2.5 and Q2.6: no new state kind in v3, concurrency out of v3 | casestudies/gaps.md; priorart/plan/P11-state-kinds.md |
| R15: Ascend study argued against by its own reason | major | fixed | Q2.7 (no Ascend study, NVDLA queued) |
| R20: an instruction-count extractor can disagree with a user cost model | major | fixed | Q4.5; B10, B37 |
| R21: equality needs B11 copies as one gated item | major | fixed | Q4.6; B30, B11 |
| R22: rejecting overlapping rewrite rules is wrong | blocker | fixed | Q4.1 (overlap allowed, ties by declaration index); B35 |
| R26: reason contradicts OD19 | blocker | fixed | OD19; backend/plan/naming.md (`Buffer` kept) |
| R28: unroll bound undefined, opposes F4 | major | fixed | Q5.1 with the bound set in meeting/resolutions.md; O19; semantics F4 |
| R29: StableHLO interpreter rests on one reviewer, not on PyPI | major | superseded by OD28 and Q6.6: no interpreter, NumPy references and module-text goldens | O13; tests/corpus/seeding.md |
| R54: new ASCII exemption is an owner rule change | major | fixed | Q7.1 (archived trees stay on tags, off master, no rule change); act/plan/tutorials.md |
| R56: exact arithmetic contradicts H4 and the backend's checked i64 | blocker | fixed | Q1.2 (every intermediate a checked i64); Q4.9, Q5.3 |
| R58: removal of the placement marks | major | superseded by OD30: the marks stay as shorthand | Q9.2; D10, D33; tests/corpus/overlap.md |
| R62: one M-sized shape item has no evidence for the size | major | fixed | Q3.1 (U4 sized L, move PR then checks PR); shared/plan/U4.md |
| RA1: R34 missing from the numbering | minor | fixed | minutes/rulings.md (R34 retired) |
| RA2: equal regions spelled differently, partial overlap at issue | major | fixed | Q9.3 (exact-or-apart default, one fault); S1.3.5 |
| RA3: overlapping output regions | major | fixed | Q9.3, Q1.3; S1.3.4 |
| RA4: owner of the tolerance | minor | fixed | Q1.5, Q6.6 (the spec states the bound, `taidl.testing` implements it) |
| RA5: float `dot` precision on GPU after R32 | major | fixed | Q6.12 (`precision_config` HIGHEST only while the oracle lowers to XLA; durable rule is the S1.3.15 bound); tests T15 |
| RA6: undefined values inside semantics (`compare` default, divide by zero) | major | fixed | Q1.8, Q1.9; S8.4 |
| RA7: byte order and sub-byte packing in `d0` | minor | fixed | Q5.4 (little-endian); S8.4, S1.3.14 |
| RA8: state width and range | major | fixed | Q1.2, Q4.7; S4.4 (initial value rejected past 64 bits) |
| RA9: data-dependent addressing | minor | fixed | Q2.5 (one study joined with L23); casestudies/plan/c08_data_dependent_addressing.md |
| RA10: size of `d0` | minor | fixed | Q2.9 (`d0` declared with one addressed dimension, sized by `hbm`); S4.3 |
| RA11: migration for v3 breaking changes | major | fixed | Q3.12, Q7.12 (one coded rejection naming the replacement, no converter) |
| RA12: no ruling sums Q1 and Q2 items against OD11 | major | fixed | Q8.9 (dates move, scope does not); roadmap/load.md; OD11 |
| RA13: precedence between semantics docs and the rulings | minor | superseded by meeting/process.md: only owner decisions bind, panel decisions settle each conflict | semantics/rulings.md (rewritten to the decisions) |
