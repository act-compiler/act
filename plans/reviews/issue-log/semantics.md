# Issue log: semantics

Sources are `reviews/semantics/checker.md` (SC1 to SC31) and `reviews/semantics/defined-behavior.md` (DB1 to DB19).
The semantics track has no review-response file; each row was checked against `semantics/` as it stands and against panels 1, 2, 4, and 9, and the joint decisions of panels 1 and 5.
Severity is the review's own label.
Panel questions are cited as Qn.k and owner decisions as ODn.
Where a review proposed one side of a conflict and a decision took the other, the finding is superseded.

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| SC1: constraints read the pre-state, R7 says post-update | blocker | superseded by Q1.1: every expression of an issue reads the pre-state | semantics/dynamics.md (issue steps, `rho_0`); semantics/holes.md H3 |
| SC2: the track keeps `set_inplace`, R5 removes it | blocker | superseded by OD22, OD30, and Q9.2: the mark stays as shorthand for a relation | semantics/rulings.md (in-place marks); semantics/plan/F2.md |
| SC3: per-operator i64 check against R56's exact arithmetic | major | superseded by Q1.2: every intermediate is a checked i64 | semantics/dynamics.md (expression evaluation); semantics/plan/F3.md |
| SC4: dynamics allows any overlap, holes.md rejects partial overlap | major | fixed | Q9.1, Q9.3; semantics/rulings.md; semantics/dynamics.md |
| SC5: REQUIRE list incomplete and unnumbered | major | fixed: REQUIREs added for reshape count, dimension at least 1, and constraints; clause ids are the S1.3 ids | semantics/dynamics.md steps 1 to 3; Q6.4 |
| SC6: "reject" used for run-time errors | major | fixed | Q7.8; semantics/dynamics.md ("fault" throughout) |
| SC7: "tensor buffer" and "control register" against OD19 | major | fixed | semantics/README.md; semantics/plan/F4.md; semantics/holes.md (H12) |
| SC8: backend state in scope, typed state kinds | major | superseded by Q4.7 and Q2.5: read-only state folds in v3, updates after v3, no new state kind | semantics/plan/F4.md |
| SC9: numeric classes omit `convert`, indices, data movement | major | fixed: one table places each of the 75 words of S8.1 in a class, once per type class | Q1.8, Q1.9; semantics/dynamics.md (numeric classes) |
| SC10: two norms for division by zero | major | fixed | Q1.9, joint decision of panels 1 and 5 point A; semantics/dynamics.md (integer divide row) |
| SC11: `cholesky`, `triangular_solve`, `batch_norm_inference` removed in v3 | major | fixed: no rule and no class row names the three words; typing-ops.md and syntax.md state that v3 drops them | Q2.8; semantics/typing-ops.md; semantics/dynamics.md |
| SC12: WF-INSTR allows inputs with no semantics | minor | fixed | Q2.9; semantics/typing.md (WF-INSTR) |
| SC13: initial `Mem`, `Reg`, and `\|d0\|` undefined | minor | fixed | semantics/dynamics.md (machine state) |
| SC14: grammar and `names.py` citations do not resolve | major | fixed | semantics/syntax.md (`IDLV2Expr.g4:11-25`); semantics/typing.md (`names.py:7-14`) |
| SC15: spec-draft citations do not resolve | major | fixed: spec-draft citations are section ids | semantics/dynamics.md; semantics/holes.md; semantics/typing.md; semantics/typing-ops.md; semantics/review.md |
| SC16: backend and oracle citations are stale | major | fixed: `spec_checks.py` and `expression_writer.py` cites point at lines that exist | semantics/holes.md; semantics/dynamics.md; semantics/review.md |
| SC17: expression literal range is 0 to 2^63 - 1, not signed | minor | fixed | semantics/syntax.md; semantics/typing.md (`0 <= k < 2^63`) |
| SC18: operand grammar `o` too loose | minor | fixed | semantics/syntax.md (`o` by operator) |
| SC19: WF-IO holds only when regions are given | minor | fixed | semantics/typing.md (WF-IO, WF-INSTR) |
| SC20: `compare` `type` must lie in the allowed set | minor | fixed | semantics/typing-ops.md (`compare`) |
| SC21: `broadcast` and `dynamic_slice` side conditions unchecked against StableHLO | minor | fixed: `broadcast` takes distinct dimensions as StableHLO does, not strictly increasing; `dynamic_slice` and `dynamic_update_slice` state sizes, one index type, and clamping | semantics/typing-ops.md |
| SC22: float sum over size-1 dimension is not "unchanged" at -0.0 | minor | fixed | Q1.11; semantics/rulings.md (reduction initial values) |
| SC23: review.md overstates "total by construction" and "identically" | minor | fixed | semantics/review.md |
| SC24: R8 accumulation rule and NaN comparison missing | minor | fixed | Q1.5, Q1.7; semantics/dynamics.md (accumulating, special values) |
| SC25: plan ids F1 to F9 collide with the language bugs F1 to F23 | minor | fixed: the language bugs are LB1 to LB29 | language/plan/bugs.md; language/plan/README.md |
| SC26: source of `\|d0\|` differs between tracks | minor | fixed | Q2.9; semantics/rulings.md (out-of-bounds regions) |
| SC27: F1 duplicates L3, L12, O6, O7 | major | fixed | Q3.1; semantics/plan/F1.md (feeds U4) |
| SC28: F2, F3, F5, F6, F8 overlap language items | major | fixed: each F item names the language item that owns it | semantics/plan/README.md; semantics/plan/F1.md to F8.md |
| SC29: F7 and F9 duplicate the corpus and the interpreter | major | fixed: F7 and F9 are cut and carried by T7, T8, T26, and T29 (Q6.6, OD28) | semantics/plan/README.md; semantics/plan/F7.md; semantics/plan/F9.md |
| SC30: plan order, F5 after F6, tests with each item | minor | fixed: F5 depends on F6 | semantics/plan/README.md; semantics/plan/F5.md |
| SC31: items missing for H10, H11, R12, R14, R44, R55 to R58 | major | fixed: the rules are rewritten to the decisions in place | semantics/dynamics.md; semantics/holes.md; Q1.4, Q1.12 |
| DB1: constraints and starts read different states | blocker | superseded by Q1.1: one view, the pre-state, not the post-update state the review proposed | S5.3, S5.4, S5.6; semantics/dynamics.md |
| DB2: R56 exact arithmetic against intermediate faults | blocker | fixed | Q1.2; S6.2, S1.3.1; semantics/dynamics.md |
| DB3: R5 against the in-place flag and exact overlap | blocker | superseded by OD22, OD30, Q9.2, and Q9.3: the mark stays, each pair has one relation | S5.5, S1.3.6; semantics/rulings.md |
| DB4: zero-filled buffers hide reads of unwritten storage | major | fixed | Q1.4 and the joint decision on unwritten reads; S5.6; B36 |
| DB5: the backend is a producer that never emits a fault | major | fixed | Q1.12, Q4.9; B30, B36 |
| DB6: float accumulation bound | major | fixed | Q1.5; S1.3.15, S8.4; L31 |
| DB7: excess precision in bf16 chains | major | fixed | Q1.6; D24, O21 |
| DB8: denormals kept or flushed | major | fixed | Q1.6 (kept, verified by a pinned-jaxlib test per platform); semantics/dynamics.md |
| DB9: `convert` out of range | major | fixed | Q1.8; semantics/dynamics.md (convert row) |
| DB10: tensor data never faults, indices clamp | major | fixed | Q1.9, joint decision of panels 1 and 5 point A |
| DB11: two words for one outcome ("does not hold", fault) | minor | fixed | Q1.12; semantics/dynamics.md step 1 |
| DB12: integer edge cases left open | minor | fixed: `min % -1`, negative integer `power`, and integer `dot` are stated; the oracle verifies the `power` values against XLA CPU | semantics/dynamics.md (integer exact, integer divide); semantics/plan/F5.md |
| DB13: -0.0 in a size-1 float sum | minor | fixed | Q1.11; semantics/rulings.md |
| DB14: NaN payload and zero sign | minor | fixed | Q1.7; semantics/dynamics.md (special values) |
| DB15: float literal rounding to infinity | minor | fixed | Q1.10; S7.5 |
| DB16: special values of approximate words | minor | fixed | Q1.7; semantics/dynamics.md (special values) |
| DB17: R7's reason cites placement-mark code | minor | superseded by Q1.1: R7's post-state rule is not carried | none |
| DB18: S1.3 lists 15 undefined rows | minor | fixed | OD16; S1.3 fault list |
| DB19: H3 remedy rejects looped starts, R28 unrolls | minor | fixed | Q5.1; O11 conditional on the spike |
