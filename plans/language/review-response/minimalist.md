# Response to the minimalist review

Rows answer `reviews/spec/minimalist.md`: V1-V43 the verdicts table in order, G1-G4 the speculative-generality bullets, O1-O8 the order and sizes bullets, and C1 the cuts list.
A panel decision such as Q1.5 or Q9.2 is cited where it settles a row.

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| V1 buffers and regions keep | accepted | Used by every instruction | none |
| V2 `d0` special case merge | accepted | `accelerator.py:31` then `instruction.py:123` special-cases it | S4.3; L30, LB20 |
| V3 two attribute lists keep | accepted | B's allocator needs address-free sizes | none |
| V4 state keep, do not extend | accepted | No extension planned in this track | none |
| V5 constraints keep | accepted | Every fault case can be one | none |
| V6 `add_constraints` merge | accepted | Two spellings of one list (`accelerator.py:66`, `instruction.py:321-323`); owner decision 10 batches the break | S5.3, S9; L30, D34 |
| V7 `set_inplace` derived from region text | rejected | Owner decision 30 keeps the mark as shorthand; it expands to `same(In k, Out)` plus a recorded tie (Q9.2) | D33 |
| V8 cut placement marks | rejected | Owner decision 30 keeps them as shorthand; each expands to its relation's constraint (Q9.2) | D10, D33 |
| V9 semantics block keep | accepted | Checked both ways | none |
| V10 `ROOT` defer | accepted | Kept; no run-time effect stated | S7.1 |
| V11 cut presumed reshape | accepted | Q2.9 cuts it; the language plan applies the cut | S7.2 |
| V12 keep 75 plain words | accepted | One row each in `OPERATIONS` | none |
| V13 cut three rare words | accepted | Q2.8 removes them | L33, D33 |
| V14 `reduce_<reducer>` keep | accepted | Keeps bodies out of the grammar | none |
| V15 19 element types keep | accepted | O supports all | none |
| V16 S1.3 shrink to two outcomes | accepted | Owner decision 16 | S1.3 |
| V17 S2 and S3 copies merge | accepted | The copy had drifted (`IDLV2Expr.g4:25` missing) | S2, S3; D9, L8 |
| V18 cut reserved names | accepted | Q2.10 keeps one pointer sentence and no table | S4.1; D17 |
| V19 parsed trees to the README | partly | Moved out of S6 into the object model S4.6, since generators need it as a contract (implementer Q5-Q7) | S4.6 |
| V20 S8.2, S8.3, S8.5, S9 keep | accepted | One table each | none |
| V21 S10 bans for four words only | rejected | Owner decisions 19-20 retire "tensor buffer", "data slice", and "global memory", which only a ban records | S10 |
| V22 L1, L2 keep | accepted | Every item cites them | none |
| V23 merge L3, L12, O6, O7 | accepted | Q3.1 folds L12, O6, O7 into U4; L3 stays the spec text | L3, L12, D26 |
| V24 L4 to S | accepted | O already starts at the identity | L4 |
| V25 L5 to S | accepted | O already rejects out-of-bounds regions at the call | L5 |
| V26 L6 keep | accepted | S6.2 states it | L6 |
| V27 L7 cut from this track | partly | O's `coverage.md` exists; B has no table, so L7 narrows to B | L7 |
| V28 L8 simplify | accepted | No grammar copy to diff | L8 |
| V29 L9 defer | accepted | `tests/test_examples.py` already runs the examples | L9 |
| V30 L10, L11, L13 keep | accepted | Each turns a rule into a check | none |
| V31 L14 cut the code | accepted | S6.2 now rejects only the token `0` | L14 |
| V32 L15 shrink | accepted | Wider renames go to the L30 table (Q7.10) | L15, L30 |
| V33 L16 defer | rejected | B and O disagree on literals today (`spec_checks.py:94-107`, `utils.py:398-426`) | L16 |
| V34 L17 keep, shrink | accepted | MX types cover Blackwell and Trainium2; the rest have one user | L17 |
| V35 L18 cut | partly | Deferred after v3, IDs stable; SME is outside owner decision 15 and D28 gives an idiom | L18, D28 |
| V36 L19 defer | accepted | Span plus strided `slice` describes it | L19, L24 |
| V37 L20 defer | accepted | `dot` plus `pad` and `slice` describes it (D27) | L20 |
| V38 L21 cut | partly | Deferred after v3: `clamp` then `convert` describes saturation | L21, L24 |
| V39 L22 cut | accepted | Names no accelerator | L22 |
| V40 L23 to late Q2 | accepted | Real need, no idiom | L23 |
| V41 L24 keep, add idioms | accepted | Idioms replace four deferred items | L24 |
| V42 L25, L26 keep | accepted | Error texts change once | none |
| V43 cut L27, L28 | partly | L27 owned by B, O, A; L28 after v3 | L27, L28 |
| G1 fewer than two accelerators | accepted | As V35, V38, V39 | L18, L21, L22 |
| G2 describable today | accepted | Idioms in L24 | L24 |
| G3 only L17 and L23 add capability | accepted | Plan keeps those two in v3 | extensions plan |
| G4 apply the review rule to Q2 | partly | Q2.3 sets the admission rule for every extension | extensions plan |
| O1 sizes assume two engineers | accepted | Owner decision 11 | plan headers |
| O2 28 items, three L | partly | L22 cut; L3 and L12 stay L until the merge (R62) | L22 |
| O3 tutorial fail points void | accepted | Tutorials use syntax `IDLV2.g4:6` rejects | L10 |
| O4 one breaking batch | partly | v3 batch per owner decision 10; the word cuts are in L33, and the marks stay (owner decision 30) | overhaul steps |
| O5 L5 and S1.3 one S item | accepted | Owner decision 16 | L5 |
| O6 remove `set_inplace` | rejected | As V7 | D33 |
| O7 78 words, not 70 | accepted | `IDLV2.g4:45-74` | L3, D1 |
| O8 order after the batch | accepted | L2, shape rules, L17, then L23 | overhaul steps |
| C1 cuts list | partly | Each cut is answered in its V row | see V rows |
