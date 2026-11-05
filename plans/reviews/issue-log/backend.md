# Issue log: backend

Sources are `reviews/backend/verification.md`, `architect.md`, and `critic.md`.
The dispositions come from `backend/review-response/` (verification, architect, critic) and were checked against the backend plan items and panel 4.
Rows of `backend/review-response/binding.md` are owner decisions and rulings applied, not review findings; the rulings among them appear in rulings.md.
Findings are keyed VER (verification), ARC (architect), and CRI (critic).
Panel questions are cited as Qn.k and owner decisions as ODn.

## Findings

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| VER-1: size and constraint errors for `@s.` do not name the state | minor | fixed | backend/coverage-table.md (`@s.` row); B27 |
| VER-2: C5 claim "before anything is written" is false; old binary is deleted first | major | fixed | backend/coverage-table.md (intro) |
| VER-3: C12 unit dimension past 64 bits is rejected too | minor | fixed | backend/coverage-table.md (buffer row) |
| VER-4: C35 unread-value check covers inputs | minor | fixed | backend/coverage-table.md (semantics lines row) |
| VER-5: C37 six element types accepted on a buffer `var_type` | major | fixed | backend/coverage-table.md (new buffer row) |
| VER-6: C44 float literals are not range-checked | major | fixed | backend/coverage-table.md (split row); B20 |
| VER-7: D9 "refuse" is in docstrings, not messages | minor | fixed | backend/plan/bugs.md D9 |
| VER-8: D11 mutation is at `slow.rs:24` | minor | fixed | backend/plan/bugs.md D11; B9 |
| VER-9: `parameter` has no coverage row | minor | fixed | backend/coverage-table.md (new row); backend/plan/coverage.md |
| VER-10: `concatenate` takes exactly two operands | major | fixed | backend/coverage-table.md (new row); B17 |
| VER-11: C40 needs a `partial` row for `concatenate` | major | fixed | backend/coverage-table.md (operation rows) |
| VER-12: `@a.` in a size is rejected, no row | minor | fixed | backend/coverage-table.md (new row) |
| VER-13: a constraint whose integers pass 64 bits is rejected, no row | minor | fixed | backend/coverage-table.md (new row) |
| VER-14: buffer `var_type` of the six narrow types is accepted, no row | minor | fixed | backend/coverage-table.md (new row) |
| VER-15: float `constant` literals not range-checked, no row | minor | fixed | backend/coverage-table.md; B20 |
| VER-16: rank-0 shape has no row, support unverified | major | fixed | backend/coverage-table.md (new row); B26 test |
| VER-17: S1.3 implementation-defined cases have no row | major | fixed | backend/coverage-table.md (new section); B15 |
| VER-18: one-element `broadcast` guard assumes the previous line | major | fixed | backend/plan/bugs.md D13; backend/coverage-table.md; B6 |
| ARC-1: B3 goldens neither prove an unchanged selection nor stay stable | major | fixed | B29 (new); B3; backend/plan/decisions.md; Q4.2 |
| ARC-2: B1 and B28 first, before lanes split | minor | fixed | backend/plan/overhaul.md step 2 |
| ARC-3: B13 should build from a typed front-end IR, not ANTLR | major | fixed in part: B13 builds from `instruction.py` objects until the front-end contract lands | B13; Q3.4 |
| ARC-4: typed payloads first, then B7, then B6 | major | fixed | backend/plan/isel.md order; B31; Q4.3 |
| ARC-5: `dot_?` caveat inverted, payloads never match | major | fixed | B31, B6, B8; backend/plan/decisions.md |
| ARC-6: B2 misses `metadata.json`; `hold_tensors` is allocator logic | major | fixed | B2, B4 |
| ARC-7: B10 freezes a cost-blind DFS and an exponential enumeration | major | fixed | B10; Q4.5 |
| ARC-8: B12 over-engineered; decide data versus code now | minor | fixed in part: B12 after B6; the data form is a two-rule spike per Q4.4 | B12, B6; Q4.4 |
| ARC-9: B7, B9, B26 move to Q1 | major | fixed | backend/plan/isel.md; backend/plan/defects.md; backend/plan/overhaul.md |
| ARC-10: B9 right after B7 | minor | fixed | B9 |
| ARC-11: B21, B22, B24, B27 agree; B23 rules from the B7 table | minor | fixed in part: B21 and B22 become one `validate()` call | B21, B22; Q3.2 |
| ARC-12: B23 re-homed to the language track | major | fixed | B23 cut; Q3.1, Q4.8 |
| ARC-13: B11 upgraded to a solver equality | major | fixed | B30 (new); D14; Q4.6, OD17 |
| ARC-14: B5 over-engineered | minor | fixed | B5 (redefined) |
| ARC-15: missing item, typed payloads | major | fixed | B31 (new) |
| ARC-16: missing item, rule scheduling | major | fixed | B32 (new); Q4.10 |
| ARC-17: missing item, staffing for 3-4 engineers | major | fixed | backend/plan/overhaul.md lanes; backend/plan/README.md; OD11 |
| ARC-18: rewrite from scratch rejected except dispatch and extractors | minor | fixed | backend/plan/overhaul.md; backend/plan/decisions.md |
| ARC-19: B10 candidate order keeps cost-blind selection | minor | fixed | B10 |
| ARC-20: B4 could delete `hold_tensors` | major | fixed | B4 |
| ARC-21: B2 `--log` printer must stay byte-compatible | minor | fixed | B2 |
| ARC-22: revised order table | minor | fixed in part: adopted with B30 in lane B Q1; B12 depends on B6 | backend/plan/overhaul.md |
| ARC-23: `search` returns `Result`, with a deterministic mode | major | fixed | B29; Q4.2, OD27 |
| CRI-1: B6 keeps conditions that index RHS positions | major | fixed | B6 |
| CRI-2: string payload and discriminant-only `matches` | major | fixed | B31 |
| CRI-3: two goldens are a thin net for B7 | major | fixed | B3 |
| CRI-4: B1 in the same commit as B10 and B5 moves | minor | fixed in part: B10 and B5 stay separate commits (one concern per commit) | B1 |
| CRI-5: B2 generate the struct for both sides from B13 | minor | fixed in part: the Rust struct is the schema and cbindgen derives the C header | B2, B13; Q4.11, OD28 |
| CRI-6: B3 oracle-differential gate for behavioral steps | major | fixed | B3; Q4.2 |
| CRI-7: B5 defer the split | minor | fixed | B5 (redefined) |
| CRI-8: B6 bind variables by name | major | fixed | B6 |
| CRI-9: B6 attributes as child e-nodes so egg matches them | major | fixed in part: typed fields (B31) replace the string parse; `find_matches` stays where a condition reads an unbound inner e-node | B31, B6 |
| CRI-10: B7 needs a typed payload | major | fixed | B7 depends on B31 |
| CRI-11: B8 fires and near-miss tests | minor | fixed | B8 |
| CRI-12: B9 add the constant node in `Analysis::modify` | minor | fixed in part: constant-ness in class data; the node is emitted at candidate building | B9 |
| CRI-13: B10 keep both strategies | major | fixed in part: files merge, one k-best extractor replaces two strategies | B10, B37; Q4.5 |
| CRI-14: B10 dedupe candidates across passes | minor | fixed | B34 (new) |
| CRI-15: B12 needs no formatter run | minor | superseded by OD29 (rustfmt runs on the generated crate) and Q4.12 | B12 |
| CRI-16: B13 emit the B2 struct | minor | fixed in part: B13 kept; the struct is not emitted from it | B13; Q4.11 |
| CRI-17: drop the Q2 data-driven applier experiment | minor | fixed | B6; backend/plan/decisions.md; Q4.4 |
| CRI-18: missing item, typed payload | major | fixed | B31 (new) |
| CRI-19: missing item, target constants in rewrite 0 | major | fixed | B33 (new); Q4.10 |
| CRI-20: missing item, rule sets and phasing | major | fixed | B32 (new); Q4.10 |
| CRI-21: missing item, search budget | minor | fixed | B34 (new) |
| CRI-22: missing item, differential testing as the gate | major | fixed | B3 |
| CRI-23: fail point, B6 `dot_?` relies on `FromOp` | minor | fixed | B31 caveat |
| CRI-24: fail point, B2 `--log` files read by tests | minor | fixed | B2 |
| CRI-25: dependency, B12 after B6 | minor | fixed | B12 |
| CRI-26: dependency, B9 before B10 | minor | fixed | B9, B10 |

## Confirmations

- The track recorded these rows as accepted with no change, because the review agrees with the plan; they are not findings.
- Verification: C33, D2, D7 (the rows already state it), S4.5 and S9 need no backend row, S8.5 needs no row.
- Architect: B14 removes the import cycle, B15 to B20 stay in Q2, keep the C++ allocator.
- Critic: B4, B11, B14, B15, B16, B17 to B20, B21 to B26, B27 and B28 keep; keep the allocator port rejected.
