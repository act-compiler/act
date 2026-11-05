# Panel 9 decision: overlap and hazard declarations

| Question | Decision (one line) | Confidence |
| -------- | ------------------- | ---------- |
| Q9.1 | Four relations per same-buffer input-output pair (tied, apart, exact-or-apart, may overlap) plus written constraints over any region pair; no hazard-order declaration in v3 | medium-high |
| Q9.2 | Option D: each pair gets exactly one relation, declared by a marker or the default, expanded into the one constraint list; the tie and the pair relation are also recorded for the backend | medium-high |
| Q9.3 | Default exact-or-apart; every violated constraint of any source is the one fault at issue (S1.3.5); static rejection only where text alone decides | medium-high |
| Q9.4 | Front end emits one list; oracle checks it before any load and evaluates the semantics on the pre-state; backend allocates under each pair's relation, narrowing its touch exception; corpus runs a declaration-by-placement matrix | medium-high (oracle, corpus), medium (backend) |

## Final rule set

- Relations an ISA specification declares, per input k and output j of one buffer:
  - tied: the output reuses input k's storage (OD17);
  - apart: no shared element;
  - exact-or-apart: equal starts and equal sizes in every dimension, or apart;
  - may overlap: no overlap restriction.
- Any further rule (input-input distinctness, a positional rule such as RVV widening) is a written constraint.
- Output-output pairs stay apart (S1.3.4), now an implicit constraint of S1.3.5.
- Marker expansions, over evaluated start and size terms (OD34):
  - `set_inplace(k)`: static checks (same buffer, OD24; equal size text), then `same(In k, Out)`, plus the recorded tie;
  - `set_apart(k)`: `apart(In k, Out)`;
  - `set_no_partial_overlap(k)`: `same(In k, Out) or apart(In k, Out)`, the explicit spelling of the default;
  - `set_may_overlap(k)` (new): the empty expansion.
- `apart(R, S)`: some dimension with `R.start + R.size <= S.start` or `S.start + S.size <= R.start`.
- `same(R, S)`: in every dimension equal starts and equal sizes; a dimension whose texts are identical is skipped.
- Default for an undeclared same-buffer input-output pair: exact-or-apart; pairs in different buffers need nothing.
- Value under any permitted overlap: the instruction's semantics evaluated on the pre-issue state (Q1.1 C).
- Violation: any constraint, implicit, marker, or written, that fails at issue is a fault before any write (S1.3.5); S1.3.6 folds into it.
- Rejected at declaration: a cross-buffer tie, unequal size text on a tie, `set_apart` on identical text, and identical input and output text without `set_inplace`.
- Oracle: checks the list before any load, then loads all inputs before any store.
- Backend: honors every constraint, reads the pair relation and the tie, and never emits a faulting issue (Q1.12).
- Corpus: five declarations by three placements (apart, exact, partial), plus rejection, F5, OD34, input-input, cross-check, and backend rows (Q9.4).

## Q9.1: what an ISA specification declares

Decision: option C, the four relations above, with constraints over any pair of regions for everything finer; no hazard-order declaration (option B) in v3.

1. Only same-buffer input-output pairs need a relation: different buffers are different storage, and OD24 rejects a cross-buffer tie.
2. Output-output overlap has no value without a write order, so it stays a fault, which no debater disputes.
3. Input-input overlap is reads only, so the language generates no relation for it; an ISA rule on it is a written constraint such as `@a.s1 != @a.s2`.
4. Tied is OD17; it is the AMX `tsrcdest` read-modify-write and the accumulate forms.
5. Apart is the hardware that reserves or corrupts overlap, such as RVV slide and gather; declaring it turns a reserved encoding into a defined fault (OD16).
6. Exact-or-apart is the overlap OD22 itself makes legal for the unmarked `r0 = r1 + r2`, and the RVV equal-EEW rule.
7. May overlap is required by OD31's last clause: constraints only remove runs (instructions.md S5.3), so without it a partial overlap a real ISA permits has no spelling once the default forbids it.
8. RVV narrowing (destination in the lowest part of the source) is such a case: equal start, smaller output, so it is partial, and its author writes `set_may_overlap` plus a positional constraint.
9. This clears the Q2.3 bar on its own terms: the idiom cannot be stated, and OD31 binds over the panel's site count.
10. "Exact" means equal starts and equal sizes; equal starts with unequal sizes is partial (the F5 defect in `keep_off`).
11. Option B is not admitted: every cited order-bearing case (x86 `REP MOVSB`, Arm `CPYF`) is a CPU bulk copy outside OD15's scope, and no in-scope tensor ISA defines a non-snapshot value for a permitted overlap.
12. Such hardware declares the pair apart in v3, a defined fault; an order-bearing relation enters later only under the Q2.3 rule.
13. Value: a permitted overlap gets the semantics on the pre-issue state, which is the meaning of every TAIDL expression (Q1.1 C), not an assumed hazard model.
14. The hazard model OD31 rules out is "which overlaps are permitted without the author saying so"; the default permits only the exact overlap OD22 already permits, so no further one is assumed.

Evidence, verified: taidl/taidl/instruction.py:163-172 (`set_inplace` appends nothing), 211-224 (`keep_off` compares starts only, skips identical start text); RVV spec, Vector Operands: overlap only at equal EEW or the fixed lowest or highest part, "Any instruction encoding that violates the overlap constraints is reserved" (fetched); felixcloutier.com TDPBSSD: accumulation written back to `tsrcdest` (fetched).
Evidence, debater-cited: RVV slide and gather overlap reservation (Fable, Sonnet, fetched by them); AMX #UD on equal tile operands (Sonnet; the fetched TDPBSSD page names only exception class AMX-E4, so it is not decisive, and step 3 does not rest on it); `REP MOVSB` and `CPYF` order (Fable, search results).
Dissent: none in the final positions; Sonnet and Fable note that `set_may_overlap` has no in-scope tensor site yet.
Confidence: medium-high.

## Q9.2: how declarations are written

Decision: option D under OD30; every same-buffer input-output pair has exactly one relation, from a marker or the default, and the front end generates that relation's expansion into the one constraint list of S5.3.

1. OD30 makes markers shorthand that expand to constraints through the mechanism that gives every region its implicit constraints.
2. The default relation is generated per pair the same way the bounds constraints are generated per region; a marker declares the pair's relation, and the generator emits that relation's expansion instead.
3. No constraint is appended and then removed, so this is within OD30 as written; `keep_off` already replaces its own earlier text when a mark is given again (instruction.py:227-231).
4. Expansions are as in the rule set; `same` adds the size equality today's text lacks.
5. The identical-text skip stays sound: a dimension with identical start text cannot separate the regions (sizes are at least 1) and cannot break equality.
6. Under OD34 every term is the evaluated start or size expression; texts that differ but evaluate equal satisfy `same` at issue, and no expansion reads an attribute name.
7. The canonical form is the expanded constraint text, because OD30 promises the user a readable equivalent; `apart` and `same` are documentation names for it.
8. The front end also records, per input-output pair, its relation and, for `set_inplace`, the tie; the backend reads that record (Q9.4), since the allocator works on tensors, not on constraint text.
9. Which is which: a relation between input k and the output is a marker; anything else (input-input, positional, strided) is a written constraint.
10. Markers address the single output, as `keep_off` requires today; an instruction with several outputs gets the default on each pair, and a marker for it enters under Q2.3 when a site appears.
11. Panel 2 Q2.1 and Q2.2 (remove the markers) are superseded by OD22 and OD30; their constraint idioms become the documented expansions.
12. `set_may_overlap` is the working name; panel 7 may rename it without changing its meaning.

Evidence, verified: instruction.py:174-231; spec-draft instructions.md S5.3 (three sources, one fault) and S5.5 (pending); act-backend malloc_instructions_h_generator.py:71-73,509-510 and generator.py:65-68 (address-naming constraints go to the allocator).
Dissent: Opus preferred named predicates as the canonical form in round 1, then conceded; Sonnet asked the owner to confirm that a marker may stand in for the default, which step 3 answers.
Confidence: medium-high.

## Q9.3: violation and the undeclared case

Decision: option C whose violation is option B: the default is exact-or-apart, and every failed constraint of any source is the one fault at issue, before any write.

1. OD30 and S5.3 give one outcome for implicit, marker, and written constraints: the fault of S1.3.5; S1.3.3, S1.3.4, and S1.3.6 fold into it.
2. The default cannot be apart: OD22 says the unmarked `r0 = r1 + r2` allows `r2` to equal `r0`, and Opus withdrew apart on that text.
3. The default cannot be any overlap: it makes read-before-write the language-wide permission OD31 rules out, and Sonnet withdrew it.
4. Any overlap also ships wrong kernels: the allocator already lets a dying input share storage with its output at any offset (constraint.cc:206-210), so an author who forgets `set_apart` on a streaming unit gets a corrupt kernel the oracle passes, which OD27 forbids.
5. Under exact-or-apart the same omission faults at issue, so the error is loud.
6. Exact overlap is order-independent for elementwise semantics, the case OD22 names, so its pre-state value matches the hardware there.
7. For non-elementwise semantics over its own operand, exact overlap is order-dependent on hardware; that author declares apart, and a front-end lint warns on an unmarked same-buffer pair of such an instruction, with no change to semantics.
8. Option A (reject when overlap is possible) fails: starts are run-time expressions, so it would reject valid specifications.
9. Static rejection stays only where text decides: cross-buffer tie (OD24), tie size text, `set_apart` on identical text.
10. Identical input and output text without `set_inplace` is rejected, asking for the mark (panel 1 Q1.3): the overlap is certain, and without the tie the backend cannot copy a shared operand.
11. The norm, as one sentence: "Each same-buffer input-output pair must be equal or apart unless the ISA specification declares otherwise; an issue where any constraint fails is a fault before any write; a permitted overlap reads the state before the issue."
12. Panel 1 Q1.3's partial-overlap fault stands as the default, with "exact" now including size equality and with `set_may_overlap` as the declared exception.

Evidence, verified: act-backend constraint.cc:148-180 (live ranges), 206-235 (`[a,b]` and `[b,c]` only touch; only the tie exempts an edge); instruction_codegen.py:162-181; owner-decisions.md rows 22, 27, 30, 31.
Dissent: none in the final positions; Sonnet gives exact-or-apart versus apart 0.6, resting on the reading of OD22, which step 2 settles from its text.
Confidence: medium-high.

## Q9.4: generators and corpus

Decision: one constraint list and one pair-relation record per instruction, read by both generators; the corpus checks every relation on both.

1. Front end: emits bounds, output-output apart, each pair's relation expansion, and written constraints into `constraint_asts`, and the pair-relation record beside `inplace`.
2. Oracle: `generate_constraints` already runs first and inputs load before outputs store (instruction_codegen.py:162-181), so the new work is only the `set_inplace` equality, the default, and the size term.
3. Oracle value needs no per-relation code: the semantics on the pre-state is computed for every relation.
4. Backend allocator: the pair constraints name addresses, so they reach CP-SAT (generator.py:65-68).
5. Backend touch exception: for a tensor pair that only touches at instruction u (input dying at u, output of u) in one storage, the allocator applies the pair's relation from the record:
   - tied: equal offsets (B30);
   - exact-or-apart: equal offsets or the apart disjunction of constraint.cc:88-103;
   - apart: the apart disjunction;
   - may overlap: nothing.
6. Backend copy pass: unchanged and tie-only; `fix_inplace` copies a shared overwritten operand and errors on an HBM-pinned or shared-view operand (inplace_fix.rs:6-17,59-80).
7. If exact-or-apart blows up the solver on large kernels, the backend may use apart for default pairs, stricter and never faulting, recorded as a backend limit, not a language rule.
8. Corpus matrix: declarations (none, `set_apart`, `set_no_partial_overlap`, `set_inplace`, `set_may_overlap`) by evaluated placements (apart, exact, partial), each expecting the pre-state value or the one fault.
9. Corpus extra rows:
   - rejections: cross-buffer tie, tie size text mismatch, `set_apart` on identical text, identical text unmarked;
   - F5: equal starts, unequal sizes faults under the default and is legal under `set_may_overlap` (RVV narrowing model);
   - OD34: starts differing in text and equal in value, with a negative attribute, legal under the default;
   - a written positional constraint (RVV widening) and a written input-input constraint (AMX distinctness);
   - cross-check: each marker and its hand-written expansion give the same fault set over a swept attribute range;
   - lint: an unmarked same-buffer non-elementwise instruction warns.
10. Backend rows: per relation, an input dying at the instruction, asserting the allocation overlaps only as permitted and the oracle agrees on the assembled program (Q6.2); a tied shared input gets a copy; the HBM-pinned case errors.
11. These rows close the pending unmarked and partial overlap rows of the norm catalogue (panel 6).

Evidence, verified: instruction_codegen.py:162-181; generator.py:65-68; constraint.cc:88-103,206-235; inplace_fix.rs:6-17,59-80.
Dissent: none material; Fable rates the backend medium because B30 and the default disjunction in the solver are unlanded.
Confidence: medium-high on oracle and corpus, medium on the backend.

## Escalations

- Sonnet (OD31 reading X or Y): decided here, reading X, Q9.1 steps 13-14; it interprets a binding decision and changes no direction.
- Sonnet and Fable (OD22 reading): decided from its text, Q9.3 step 2.
- Sonnet (OD30 and a permissive marker): decided, Q9.2 steps 2-3.
- Opus and Sonnet (order-bearing relation): decided by OD15 scope and Q2.3, Q9.1 steps 11-12.
- Nothing goes to the owner.
