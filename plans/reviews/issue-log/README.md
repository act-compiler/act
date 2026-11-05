# Issue log

One log of every finding raised in the review cycle, grouped as the reviews are, with its severity, its disposition, and where it landed.
Each group file is a table `Finding | Severity | Disposition | Where it landed`.
Sources are the review files in `reviews/`, the review-response files of the backend, oracle, and language tracks, and the panel decisions in `meeting/`.
Where a track kept no response file, each finding was checked against the track's files as they stand.

## Files

- [backend.md](backend.md): verification, architect, and critic reviews of the backend plan
- [oracle.md](oracle.md): verification, architect, and critic reviews of the oracle plan
- [spec.md](spec.md): ambiguity, implementer, and minimalist reviews of the language specification
- [tests.md](tests.md): QA lead and industry reviews of the test plan
- [semantics.md](semantics.md): checker and defined-behavior reviews of the formal core
- [shared-diagnostics.md](shared-diagnostics.md): review of the shared-code and diagnostics plans
- [casestudies-priorart.md](casestudies-priorart.md): review of the case studies and prior art
- [act-glossary.md](act-glossary.md): review of the act plan, glossary, and terminology
- [rulings.md](rulings.md): audit of the tech-lead rulings

## Vocabulary

- Fixed: the change is in the plan, the specification, or the code the finding names; "fixed in part" names the part not done and the item that carries it.
- Rejected: the track judged the finding wrong, with the reason in the row.
- Superseded: a later owner decision (ODn) or panel decision (Qn.k) overrode the finding's proposal, or settled the question the other way.
- Open: no change is in place and no decision covers it, or a remainder of a part-fixed finding has no owner item.
- Severity is blocker (the plan or text cannot be followed, or contradicts an owner decision), major (an item, rule, or text must change), or minor (wording, citation, count).
- Severity is the review's own label where it gave one: tests, semantics, shared-diagnostics, casestudies-priorart, and act-glossary.
- The audit labels a ruling wrong or questionable; wrong maps to blocker and questionable to major.
- The ambiguity review orders its findings most severe first; the proposed norm N0 and the first four findings are blockers, the next twenty majors, the rest minors.
- The backend, oracle, implementer, and minimalist reviews give no label; severity there is assigned by the log.

## Counts per disposition

| Group | Findings | Fixed | Fixed in part | Rejected | Superseded | Open |
| --- | --- | --- | --- | --- | --- | --- |
| backend | 67 | 56 | 10 | 0 | 1 | 0 |
| oracle | 65 | 57 | 4 | 1 | 3 | 0 |
| spec | 139 | 114 | 19 | 2 | 4 | 0 |
| tests | 38 | 34 | 1 | 0 | 2 | 1 |
| semantics | 50 | 43 | 0 | 0 | 7 | 0 |
| shared-diagnostics | 20 | 17 | 1 | 0 | 2 | 0 |
| casestudies-priorart | 20 | 19 | 0 | 0 | 1 | 0 |
| act-glossary | 21 | 20 | 1 | 0 | 0 | 0 |
| rulings | 32 | 27 | 0 | 0 | 5 | 0 |
| total | 452 | 387 | 36 | 3 | 25 | 1 |

## Counts per severity

| Severity | Findings |
| --- | --- |
| blocker | 20 |
| major | 220 |
| minor | 212 |

## Conventions

- A review row that agrees with the plan and asks for no change is a confirmation, not a finding; the track's response file counted these as accepted.
- Confirmations left out of the counts: 17 in backend, 6 in oracle, 13 in spec; each group file lists them in a closing section.
- A duplicate review row (for example a "missing" entry that repeats a table row) keeps its own line and names the row it repeats.
- Rows of the track response files that answer owner decisions or tech-lead rulings, not review findings, are not in the group files; the rulings among them are in rulings.md.
- Panel 11 (authoring surface and check function) is decided in `meeting/panel-11/decision.md`; no finding in the log waits on it.
- Items renamed or cut while closing findings: the language bugs are LB1 to LB29, F7 and F9 of the semantics plan are cut, and A14 and A22 of the act plan are merged; the roadmap carries F7 and F9 as dropped rows and A14 and A22 as merged rows.

## Findings left open

- QA22: `tests/decisions/round1.md` and `round2.md` hold decisions in force but keep round names, and deleting or renaming them is a file move; T19 to T21 stay as scheduled items of the wave plan (Q8.4)
