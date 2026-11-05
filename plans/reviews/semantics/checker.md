# Checker review: formal semantics track

Scope: `plans/semantics/` (syntax, typing, typing-ops, dynamics, holes, rulings, review, plan F1-F9).
Evidence: `taidl/taidl/*.py`, `IDLV2.g4`, `IDLV2Expr.g4`, the act-backend and act-oracle checkouts, owner decisions, tech-lead rulings.
Verdicts per row: correct, wrong, unverifiable.

## Verdict

The track is a sound backbone: the front-end claims in `syntax.md` and `typing.md` are nearly all right, and all 78 words, the 19 element types, and the (FE) and NEW labels match the code.
It is not ready to adopt, because three of its central rulings were overtaken by the final decisions.
Constraints read the pre-update state (R7 says post-update), `set_inplace` stays and an unmarked exact overlap is rejected (R5 removes the mark and aliases identical texts), and every intermediate must fit i64 (R56 checks only the final value).
Dynamics also contradicts holes.md on input and output overlap, and its numeric classes leave `convert` and the data-movement words open, which decision 16 forbids.
Many line citations do not resolve: grammar, `names.py`, spec-draft, and backend lines.
The F plan overlaps the language, test, and oracle plans in six of nine items (F1, F2, F3, F5, F7, F8) and F9 duplicates R29.

## Tally (rows or claims checked)

| File | Checked | Correct | Wrong | Unverifiable |
| --- | --- | --- | --- | --- |
| syntax.md | 22 | 17 | 5 | 0 |
| typing.md | 30 | 25 | 5 | 0 |
| typing-ops.md | 34 | 30 | 1 | 3 |
| dynamics.md | 28 | 21 | 5 | 2 |
| holes.md | 19 | 15 | 4 | 0 |
| rulings.md | 3 | 1 | 2 | 0 |
| review.md | 11 | 9 | 1 | 1 |
| Total | 147 | 118 | 23 | 6 |

## Findings

CONFLICT marks a clash with a final owner decision (OD) or a tech-lead ruling (R).

| ID | Severity | Where | Finding | Proposed change |
| --- | --- | --- | --- | --- |
| SC1 | blocker | dynamics.md step 1-3; holes.md H3, H11; plan/F4 | CONFLICT R7. Step 1 evaluates constraints on `rho_0` (pre-update state); R7 makes starts, sizes, and constraints all read the state after updates. H3 and F4 offer "Q2 pre-state everywhere" as the alternative; no Q2 list exists in the track. | Order the issue as updates, then constraints, regions, and semantics all on `Reg_1`. Rewrite H3, H11, F4 to R7 and delete Q2. A fault in an update precedes any constraint test. |
| SC2 | blocker | holes.md H1; rulings.md "In-place marks"; typing.md WF-INPLACE, WF-INPLACE=; syntax.md `ip`; plan/F2 | CONFLICT R5, OD17. The track keeps `set_inplace`, rejects an exact overlap without the mark, and rejects a flag on an instruction with other than one output. R5 removes `set_inplace` in v3: input and output alias exactly when region texts are identical. | Delete `ip`, WF-INPLACE, WF-INPLACE=. Define alias as identical (buffer, starts, sizes) text of an input and an output, allowed, with value semantics unchanged. State what a partial overlap does (fault or read-pre-state), then repoint F2 to R5 and R21. |
| SC3 | major | dynamics.md "Expression evaluation"; holes.md H4; plan/F3 | CONFLICT R56. `[[e1 + e2]]` is "in Z, then REQUIRE in i64" for every operator, so a transient out-of-range value faults. R56: arithmetic is exact and only the final value outside i64 faults. | Evaluate in Z and apply one REQUIRE to the final value of each expression text. Reword H4 and F3. Backend `checked_*` (expression_writer.py:72-96) is per operator and must change. |
| SC4 | major | dynamics.md steps 4-6 and bullet "input that overlaps an output sees the pre-state"; holes.md H1 | Internal contradiction. Dynamics allows any input and output overlap by read-before-write and has no REQUIRE for it. H1, rulings.md, and F2 reject partial overlap. | After SC2, add one REQUIRE for partial input and output overlap (or state it is allowed) and use the same wording in H1, rulings.md, and F2. |
| SC5 | major | dynamics.md steps 3-6; plan/F7 | Dynamics lists REQUIRE only for state range, region bounds, and output disjointness. Missing: presumed-reshape element count equal at issue when dims are backtick expressions (H10), backtick dim >= 1 and in i64 (oracle `check_sizes`, utils.py:651-665), constraint failure as a named fault. F7 promises "one issue per REQUIRE". | Number every REQUIRE (R1..Rn) in dynamics.md so F7 and the clause ids of R44 attach to them. Add the missing ones. |
| SC6 | major | dynamics.md, holes.md, typing.md ("reject", "REJECT", "ill-formed") | R55 names a run-time error a "fault": no region and no state written, reported with the clause id (R44). The track uses REJECT, "reject the issue", and "ill-typed" for both static and run-time cases. | Use "fault" for issue-time errors and "reject" for front-end errors only. Add "a fault writes nothing" to `issue`. Give each REQUIRE a clause id. |
| SC7 | major | README.md, syntax.md, dynamics.md, plan/F4, F8 | OD19, OD20 and the glossary: a data model has "buffers" and "state". The track says "tensor buffer", "control register", and "data model" for one entry (`model(m,...)`, "data models" D). | Rename: buffer, state (variable), `D` as buffers, "control register semantics" to "state semantics". Keep "region" (OD20) and `@s.` (OD21). |
| SC8 | major | plan/F4; syntax.md S; typing.md `Sg` | CONFLICT R27, R14. F4 lets the backend keep rejecting state in v3. R27: implement in v3 (Q2). R14 puts typed state kinds (TMA descriptors, async groups) in v3. `Sg = {s -> i64}` has no typed kinds. | F4: backend state is in scope for Q2, not optional. Add a typing and dynamics rule for typed state kinds (new item). |
| SC9 | major | dynamics.md "Numeric classes" | OD16 gap: classes omit `convert` I to I (sub-byte), F to I (NaN, out of range), I to F, F to F overflow (f8e4m3fn has no infinity, so "IEEE-754" is wrong for it), float `remainder`, `power` at I, `dot` and `reduce_*` at I, and every data-movement word. `dynamic_slice`, `dynamic_update_slice`, and `gather` clamp or fault on out-of-range indices and the track is silent. Row 3 lists "sine, ..." and is not exhaustive. | One table with each of the 78 words in exactly one class (a test diffs it against `OPERATIONS`). Define convert and f8 overflow as fault or fixed value, and index handling as clamp (StableHLO) or fault. |
| SC10 | major | dynamics.md "integer divide" row; holes.md H17 | Two norms for division by zero: an expression `/ 0` faults (H5), but `divide` at I returns -1 and `remainder` returns x. OD16 asks for one simple norm; ruling R55 says "one word for every case". | Tech lead to rule: fault on integer `divide` and `remainder` by zero and on `min / -1`, or record XLA values as a deliberate second norm. |
| SC11 | major | typing-ops.md cholesky, triangular_solve, batch_norm_inference; dynamics.md approximate row; plan/F5 | CONFLICT R60. These three words are removed in v3. | Delete the rows and mentions. The list is 75 words after R60; recount in syntax.md `op` (78 today, matches IDLV2.g4:45-74 and operations.py). |
| SC12 | minor | typing.md WF-INSTR; syntax.md conventions | CONFLICT R11. WF-INSTR allows an instruction with inputs, no outputs, and no B (oracle checks only outputs, semantic_generator.py:14-19). R11: without semantics legal only with no regions at all; with regions rejected. Also a B with zero outputs is unwritable (instruction.py:311-318 forces `%Out0`). | WF-INSTR: B absent iff In = [] and Out = []; B present implies |Out| >= 1. |
| SC13 | minor | dynamics.md "Machine state" | R12: every buffer starts zero-filled, unwritten storage unchanged. Mem_0 and Reg_0 (= initial values) are not defined; `|d0|` is a parameter in holes.md but not in `sigma`. | Add `Mem_init = 0`, `Reg_init = v0`, and `|d0|` to the machine state. |
| SC14 | major | syntax.md e/f, cond, upd; typing.md WF-NAME | Wrong citations: `IDLV2Expr.g4:113-118`, `:121-124`, `:110` (the file has 66 lines; correct 14-19, 22-25, 11). `names.py:172-179` (file has 14 lines; correct 7-14). | Fix the four citations. |
| SC15 | major | holes.md, typing.md, dynamics.md, review.md citations to `instructions.md`, `expressions.md`, `operation_meaning.md`, `overview.md` | Unverifiable. In `language/spec-draft/` instructions.md has 64 lines, expressions.md 47, operation_meaning.md 29, but cites reach 123-167, 187-206, 89-97. overview.md cites are off by 1-3 (H1 45 is 48; H4 33 is 34; H5 35 is 37; H9 37 is 38; H14 43 is 44). Only semantics.md, conformance.md, and declarations.md cites resolve. | Cite section ids (S5.4, S1.3, S8.4) as R44 does, not draft line numbers. |
| SC16 | major | holes.md, dynamics.md, review.md citations to act-backend and act-oracle | Stale lines. `spec_checks.py` is 418 lines: one-output check is 22-30 (cited 214-221), state rejection 370-385 (cited 562-577), size tokens 274-282 (476-481), region count 394-418 (586-610), dead value and ROOT checks 179-200 (378-384). `instruction_codegen.py` has 224 lines (cited 667-737), oracle `expression_writer.py` 111 (cited 479-560; `//` is at 20-21), `operand_sizes.py` 20 (cited 616-630). typing-ops cites `spec_checks.py:286-299` for literal range; it is 94-107. The `utils.py`, `shapes.py`, `oracle_visitor.py`, `semantic_generator.py` cites are right. | Re-pin to the current checkouts or cite function names. |
| SC17 | minor | syntax.md `k`; typing.md literal rule | `-2^63 <= k` is wrong for expressions: `INT` is unsigned (IDLV2Expr.g4:59), so the literal range is 0 <= k < 2^63 (instruction.py:400-408). Only an attribute `NUMBER` can be -2^63 (IDLV2.g4:91, instruction.py:435). | Split the two ranges. |
| SC18 | minor | syntax.md `o ::= %v \| k \| lit` | Too loose. Every operand is a `%` name except `parameter(k)` and `constant(lit)` (operations.py:191-214). | `o ::= %v`, with the two exceptions stated. |
| SC19 | minor | typing.md WF-IO (FE) | Holds only after `set_inputs` and `set_outputs` ran (instruction.py:254-257). A semantics with `%Out0` and no `set_outputs` is accepted. | Mark WF-IO "FE, if regions are given"; add to F8 a rejection of a B without both region calls. |
| SC20 | minor | typing-ops.md `compare` | The cited table (oracle_visitor.py:114-118) is the allowed set, not the default; line 278 rejects a `type` outside it (for example UNSIGNED on a signed operand). The rule gives the default only. The defaults match StableHLO. | Add "T, when given, is in the allowed set of t"; cite 114-118 and 278. |
| SC21 | minor | typing-ops.md `broadcast`, `dynamic_slice` | Unverifiable. "b* strictly increasing" and "ti in I" for each index may be stricter than the pinned StableHLO (R6); the draft does not say which. | Check against the pinned version and mark each as a TAIDL difference or drop it. |
| SC22 | minor | rulings.md "Reduction initial values" | Wrong at -0.0: a float sum starts from +0.0, so an empty dimension set or a size-1 dimension gives `+0.0 + x`, not the operand "unchanged" or "the element itself". The identities themselves match `reduce_init` (utils.py:439-451) and R3. | Say "the element combined with the identity". |
| SC23 | minor | review.md bullets 4, 5 | "Total-by-construction" contradicts dynamics (division by zero, overflow). "Implemented identically in both generators" contradicts H4 (oracle exact, backend checked). "Division by the literal 0 rejected in every place" is false for `(0)` and `-0` (instruction.py:414). | Reword all three. |
| SC24 | minor | dynamics.md float approximate row and Determinism | R8 says accumulate in the result element type, within a per-word tolerance of the exact rounded result. The track omits the accumulation rule. "Tight enough for the comparison the consumer makes" is no norm. No tolerance is defined for NaN, infinity, or denormal results. | State R8 verbatim, define the tolerance on finite values, and say how NaN and infinity compare. |
| SC25 | minor | plan IDs; plan/README.md | ID collision: F1-F9 here and F1-F23 in `language/plan/bugs.md`. "S" is both the shared track and spec sections S1-S10. | Rename the track items (for example X1-X9) and say "shared track" in dependencies. |
| SC26 | minor | rulings.md H8; holes.md H8 | `|d0|` is an issue-time program parameter here; `language/plan/bugs.md` F22 and L5 add an optional declared size in the ISA description. | Pick one with the language track and cite its item. |

## Plan items F1-F9

| ID | Severity | Where | Finding | Proposed change |
| --- | --- | --- | --- | --- |
| SC27 | major | plan/F1 | Duplicates L3, L12, O6, O7, which R62 merged into one M item. F1 is size L and depends on all of L. | Replace F1 with "typing.md and typing-ops.md as the source of the R62 table"; no second checker. |
| SC28 | major | plan/F2, F3, F5, F6, F8 | Each overlaps a language item: F2 with L5, F3 with L6 and L14, F5 with L2 and L4, F6 with L5 (d0 size) and L2, F8 with L10, L11, L16. H12 closes in F3, F4, and F8. | The semantics track owns rule text and verdicts; the language track owns the front-end change and spec text. Cite the L item in each F item and keep one owner for H12. |
| SC29 | major | plan/F7, F9 | F7 duplicates the T5-T10 corpus, T6 runner, and T12 (tests/overhaul.md:11-17). F9 duplicates R29 and O13 (StableHLO reference interpreter; new dependency needs owner approval). F9 is Q1 but depends on F1 (L, Q1). | Drop F7 and F9. Hand rule ids and REQUIRE numbers to T as case names. |
| SC30 | minor | plan/README.md order | F5 should depend on F6 (byte order feeds `bitcast_convert`). F2 and F4 must wait for SC1 and SC2. Item tests should land with each item, not in a Q2 F7. | Order: SC1-SC3 rewrites, then F3, F8, F2, F6 (Q1), then F4, F5 (Q2). |
| SC31 | major | plan (missing items) | No item for H10 (reshape count), H11 (constraint precondition), non-d0 region bounds (only L5), R14 typed state, R12 zero-fill, R44 clause ids and R55 fault vocabulary, R57 constants (L16), R58 removal in syntax.md. | Add one item: "rewrite the track to decisions R5, R7, R11, R12, R14, R55-R58"; map H10 and H11 to named items. |

## Correct (spot list)

- 78 words and 19 types: IDLV2.g4:37-74, operations.py:41-164.
- WF-MODEL, WF-STATE, WF-ATTRS: accelerator.py:37-57, instruction.py:72-74.
- WF-ORDER and WF-STATE64 are correctly NEW: accelerator.py:33-44 has no order check, line 55 checks `int` only.
- WF-ROOT, WF-NAME%, WF-SSA, WF-IN, WF-ARITY, WF-LIT64: instruction.py:268-318, operations.py:171-239, instruction.py:419-437.
- WF-MARK and WF-INPLACE as stated: instruction.py:163-231.
- R3 (identities), R4 (out-of-bounds fault), R9 (ROOT on any line), R41 (`[]`), R57 (literal ranges) agree with the track.
- Floor division and remainder: expression_writer.py (oracle) 20-21, quotient.txt in act-backend templates.
- Type classes in typing-ops.md agree with `TYPES` (oracle_visitor.py:19-90).
