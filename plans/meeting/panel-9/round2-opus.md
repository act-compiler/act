# Panel 9 round 2: Opus (semantics and spec author)

Binding: owner decisions 16, 17, 22, 24, 30 (with its clarification), 31, 34.
Round-1 inputs: round1-opus.md, round1-sonnet.md, round1-fable.md; finding and evidence labels below are those files' own.

## The cross-cutting test: is the pre-state value a hazard model?

- All three round-1 files give an overlapped output the value computed from the inputs' pre-issue contents (the snapshot).
- That value is not a hazard assumption by itself: it is the meaning of every TAIDL expression, since an issue reads the pre-state and commits at the end (panel-1/decision.md Q1.1 C, line 5).
- A hazard model is a different claim: that a given overlap is permitted, because the hardware happens to read before it writes.
- So the test is not "which value" but "which overlaps get that value without the author saying so".
- Sonnet's default (any overlap legal, snapshot value) answers "all of them", which is read-before-write as a language-wide model, the case owner 31 names and forbids.
- My round-1 default (apart) answers "none", which assumes no hazard model but contradicts owner 22: "`r0 = r1 + r2` allows `r2` to equal or differ from `r0`" is stated of the unmarked instruction (owner-decisions.md, decision 22).
- Fable's default (exact-or-apart) answers "exactly the overlap owner 22 already declared legal", so its snapshot value on exact overlap is an owner-fixed relation, not an assumed hazard model.
- Every other permitted overlap gets the snapshot only because the author declared it, which is the meaning of a declared relation (owner 31).
- When the hardware differs, the ISA author has three moves, in order:
  - The hardware corrupts or reserves the overlap: declare `set_apart` (or a narrower constraint), which makes it a defined fault (owner 16), as a reserved RVV encoding is (Fable table, RVV row).
  - The hardware matches the snapshot on part of the overlaps: permit only that part with a constraint; for example an element-forward copy equals the snapshot when `out_start <= in_start`, so the author writes `out_start <= in_start` or apart.
  - The hardware defines an order-dependent value that differs from the snapshot (an element-sequential copy that smears a pattern): v3 cannot state it, and the author must forbid that overlap; this is the one residual gap against owner 31's literal text (see Escalation).

### Q9.1

Strongest opposing argument (Sonnet step 6): AMX makes any two of three tile operands equal a #UD (E3), an input-input relation, so the declarable set cannot be limited to input-output pairs as my round 1 step 2 limited it.
Second (Fable step 3): owner 31 needs an explicit any-overlap opt-in for an engine that permits partial overlap, such as a within-buffer DMA (trainium2.md:19).
Rebuttal or concession:
1. I concede Sonnet's point: the language must allow a constraint over any two regions, input-input included; it needs no marker, since a plain constraint states it (Sonnet step 6), and no implicit input-input relation is generated.
2. I keep that output-output overlap stays a fault (S1.3.4), which no file disputes.
3. Fable and I agree on an any-overlap relation; Sonnet needs none only because its default already is any overlap, which the cross-cutting test rejects.
4. With owner 22 forcing exact overlap legal by default, the closed set is four relations per same-buffer input-output pair: apart, exact-or-apart (the default), may overlap, tied.
5. No file cites a tensor ISA with a defined non-snapshot value for a permitted overlap (Fable table; Sonnet E1-E4), so no order-bearing relation (option B) is admitted under the two-site rule (panel-2/decision.md Q2.3, line 10).
6. The relation "exact" must be named exactly: equal starts and equal sizes in every dimension; RVV low-part narrowing (equal start, smaller output, Sonnet step 5) is not exact, so it is may-overlap narrowed by a user constraint, not the default.
Final position: option C, four relations (apart, exact-or-apart default, may overlap, tied) plus constraints over any region pair; changed from three relations with an apart default.
Confidence: medium-high.
What still divides the panel: Sonnet's any-overlap default; whether equal-start-unequal-size counts as exact (Sonnet's exact-or-contained versus my full equality).

### Q9.2

Strongest opposing argument (Sonnet step 1, Q9.3 step 2): under owner 30 and S5.3 markers only append constraints, and constraints only remove runs, so a default that a marker must lift is outside the one mechanism; hence the default must be the largest set.
Rebuttal:
1. No constraint is ever removed in my design or Fable's: the front end generates, per same-buffer input-output pair, the constraint of that pair's declared relation, and the undeclared relation's constraint is exact-or-apart.
2. This is how the implicit bounds constraint already works: it is generated from the region's declaration (instructions.md:27; owner 30 names it as the same mechanism), not appended and later removed.
3. A may-overlap marker therefore "expands to the constraints it stands for" (owner 30): the empty set, as the pair's declared relation; Fable's step 7 reaches the same expansion and calls it suppression, which is the same output.
4. Each marker's expansion, over evaluated terms (owner 34; Fable step 5, Sonnet step 10 agree):
   - `set_apart(k)`: the per-dimension disjunction `keep_off` emits today (instruction.py:216-219).
   - `set_no_partial_overlap(k)`: exact-or-apart, now equal to the default, kept as explicit shorthand under owner 30; its expansion must add size equality, fixing F5 (instruction.py:212-226).
   - `set_inplace(k)`: static same-buffer and equal-size checks (owner 24) plus per-dimension `out_start == in_start`, with the tie recorded for the backend (all three files agree; inplace_fix.rs:59-80).
   - may overlap: no constraint.
5. I concede that named predicates `apart` and `same` are not needed in v3: the backend reads the pair's relation from the front end's declaration record, as it reads the tie, not from constraint text (Sonnet step 6 argues the same for the tie).
   They return when strided regions (L19) need an element-set test, which is Fable's own stated condition.
Final position: option D; one generated constraint per same-buffer input-output pair from its declared relation, undeclared meaning exact-or-apart; markers are that declaration; changed only in the default and in dropping named predicates.
Confidence: medium-high.
What still divides the panel: whether owner 30 permits a declaration whose expansion is the empty set in place of the default (Sonnet would say a default must never need lifting).

### Q9.3

Strongest opposing argument (Sonnet steps 2-5): constraints only subtract, so any default stricter than "any overlap" fails a scenario a real architecture allows (RVV `vnsrl.wi v0, v0, 3`), and one norm "read every input before writing any output" is the simplest for owner 16.
Rebuttal:
1. The RVV case is not failed: its author declares may overlap narrowed to the low part, and the run has the snapshot value; nothing in the language forbids it (Q9.2 steps 1-3).
2. Sonnet's norm is short because it states read-before-write as the language's hazard model, which owner 31 names as the thing TAIDL does not assume.
3. Sonnet's own risk (step 9) is the decisive cost: an author who omits a declaration gets an oracle value the hardware may not give, and the backend may then allocate a partial overlap the hardware corrupts (Sonnet F6; my F1), a silently wrong kernel.
4. Under exact-or-apart the same omission yields a fault at issue, so the error is loud and the author adds a declaration.
5. My round-1 apart default fails owner 22, so I concede it to Fable's exact-or-apart.
6. Option A (reject when overlap is possible) fails because starts are run-time expressions (all three files agree); option B alone fails owner 22.
7. Static rejection stays only where text decides it: identical start text on a pair declared apart is false at every issue, so the front end rejects it.
8. One-sentence norm: "Each same-buffer input-output pair must be equal or apart unless the ISA specification declares otherwise; any constraint that fails at issue, implicit, marker, or written, is a fault before any write; a permitted overlap reads the pre-state."
Final position: option C whose violation is option B: default exact-or-apart, partial overlap a fault unless declared, every violation the one constraint fault S1.3.5; changed from an apart default.
Confidence: medium-high.
What still divides the panel: Sonnet's any-overlap default and its proposed reversal of panel-1 Q1.3; Fable and I now agree.

### Q9.4

Strongest opposing argument (Sonnet step 6): the backend must add no overlap rule of its own; correctness rests on the author's declarations, and a forgotten `set_apart` is the author's error.
Rebuttal:
1. I agree the backend adds no rule of its own; it honors the generated per-pair constraint like any constraint (panel-1 Q1.12, producer never faults).
2. The difference is the generated constraint itself: the allocator's touch exception lets a dying input share storage with the output at any offset (constraint.cc:206-210), which is read-before-write by default.
3. Under exact-or-apart the implicit constraint restricts that exception to equal or disjoint placement for undeclared pairs, one more disjunction in `overlap_constraints` (constraint.cc:88-103; Fable step 4).
4. Oracle: evaluates every generated constraint before any load and computes the snapshot from loads taken before stores (instruction_codegen.py:162-181); all three files agree.
5. Corpus: Fable's legal, fault, and rejected case per relation, plus Sonnet's marker-versus-hand-written cross-check, plus three cases from my round 1:
   - an F5 case (equal starts, unequal sizes) that faults under the default;
   - an owner-34 case whose starts differ in text and evaluate equal, with a negative attribute, legal under the default;
   - a may-overlap case narrowed to `out_start <= in_start`, modeling an element-forward copy.
6. Plus an AMX input-input distinctness case as a written constraint (Sonnet E3).
Final position: one generated constraint list per instruction read by both generators, the tie read additionally by the backend, the allocator honoring the implicit exact-or-apart; corpus as above; unchanged except for the default.
Confidence: medium (B30 and the implicit disjunction in the solver are design, not code).
What still divides the panel: Sonnet's expectation that undeclared partial overlap gives the snapshot value in the oracle.

## Escalation to the owner

- Not a counter to a decision; a scope question on owner 31.
- Read literally, "must not fail scenarios real architectures allow" covers hardware that defines an order-dependent value under overlap (an element-sequential copy), which only an order-bearing relation (option B) can state.
- No in-scope tensor ISA in any round-1 file defines such a value, so the panel offers none in v3 and the author forbids that overlap; the owner may confirm that deferral or ask for option B.
