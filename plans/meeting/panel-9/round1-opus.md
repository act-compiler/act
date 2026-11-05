# Panel 9 round 1: Opus (semantics and spec author)

Lens: each declaration has one meaning that both generators implement identically, with no undefined behavior.
Binding: owner decisions 16, 17, 22, 24, 30 (with its clarification), 31, 34.
Owner 31 postdates panel 1 Q1.3 steps 5-6, panel 2 Q2.1 step 7, and panel 5 Q5.6, which all rest on read-before-write as the default; those steps no longer hold as a default and are not re-debated here.

## Findings that frame every question

- F1. The backend allocator already assumes read-before-write for every instruction: live ranges `[a,b]` and `[b,c]` "only touch", so an input last used by an instruction gets no interference edge with that instruction's output and may share storage with it in any way, partial overlap included (act-backend/act_backend/generic/cpp/malloc/src/constraint.cc:148-180, 206-235).
- F2. The oracle loads every input before it stores any output and ignores the in-place mark (act-oracle/taidl_to/instruction_codegen.py:162-181), so its value under any overlap is the snapshot value.
- F3. `set_apart` and `set_no_partial_overlap` are pure constraint generators: they append address constraints and nothing else (taidl/taidl/instruction.py:174-230; spec-draft instructions.md:59).
- F4. Address-naming constraints reach the backend's CP-SAT allocator, so a restriction written as a constraint is honored at allocation (act-backend/act_backend/malloc_instructions_h_generator.py:510; spec_checks.py:351-366).
- F5. `set_no_partial_overlap` compares starts only, so equal starts with unequal sizes pass as "not partial" although the regions overlap partially (taidl/taidl/instruction.py:212-226); the relation is under-defined today.
- F6. Real ISAs permit an overlap exactly where they guarantee the snapshot result, and reserve the rest: RVV permits destination-source overlap only for equal EEW or at a fixed lowest or highest part, and "Any instruction encoding that violates the overlap constraints is reserved" (RVV spec, Vector Operands section, github.com/riscv/riscv-v-spec v-spec.adoc, fetched).
- F7. WebAssembly bulk memory copies overlapping ranges "as if the source region was first copied to a temporary buffer" (github.com/WebAssembly/bulk-memory-operations, proposals/bulk-memory-operations/Overview.md, via search result text).
- F8. XLA declares an output aliased to an input at compile time (`input_output_alias={ {}: 0 }`); if the runtime does not donate the buffer, XLA copies it and "the program runs as if the buffer had been donated" (openxla.org/xla/aliasing, fetched).

### Q9.1

Position: option C, a closed set of three per-pair relations between input k and output j of one buffer: apart (the default), may overlap (snapshot value), and tied (owner 17); every finer restriction is a constraint.

Argument:
1. Different buffers are different storage, and owner 24 rejects a tie across buffers, so only same-buffer pairs need a relation.
2. Input-input overlap is reads only and needs no relation; output-output overlap has no defined value without a write order, so it stays a fault.
3. A dataflow semantics without a hazard model can define exactly one value for an overlapping input-output pair: the value computed from the inputs' pre-issue contents (the snapshot).
4. Any other value depends on the hardware's element order, which TAIDL does not model, and owner 31 forbids assuming one.
5. Real ISAs permit overlap only where they deliver that snapshot value and reserve the rest (F6), and Wasm and memmove define overlap the same way (F7), so the snapshot loses no architecture-permitted scenario.
6. So the value side needs no choice: every permitted overlap means the snapshot, which the oracle already computes (F2).
7. What varies across real ISAs is where overlap is permitted, which is the declaration the author must be able to make.
8. "Apart" is the default (an implicit constraint, Q9.2) because it assumes no hazard model (owner 31) and is the meaning of C `restrict` and of a reserved RVV encoding, turned from undefined into a fault (owner 16).
9. "May overlap" covers owner 22's `r0 = r1 + r2` with `r2` equal to `r0`, RVV equal-EEW overlap, and memmove-style copies.
10. "Tied" is owner 17: the output reuses input k's storage, the read-modify-write form of AMX `tsrcdest`, PSUM accumulate, and Gemmini `compute_accumulated`.
11. "No partial overlap" and RVV's positional rule are "may overlap" narrowed by an address constraint, so they need no fourth relation.
12. A separate hazard-order declaration (option B) adds a second axis whose only legal value under step 3 is the snapshot, so it carries no information.

Evidence: step 1 owner-decisions.md:23; step 2 act-oracle decorator.py:124-138 for the kernel-level analogue; step 3-6 F2, F6, F7; step 8 C11 6.7.3.1 (from memory, cppreference returned 403); step 10 www.felixcloutier.com/x86/tdpbf16ps pseudocode `tsrcdest.row[m].fp32[n] := tsrcdest.row[m].fp32[n] + tmpf32` (fetched), casestudies/gemmini.md:106, casestudies/strengths.md:5; step 11 F3, F6.
Confidence: medium-high.
What would change my mind: a cited real instruction whose ISA document defines a non-snapshot result for a permitted overlap (for example a DMA defined to copy forward element by element), which would need a fourth, order-bearing relation.

### Q9.2

Position: option D under owner 30: one constraint mechanism with three sources (implicit, marker-generated, user-written); every same-buffer input-output pair gets an implicit `apart` constraint unless a marker names that pair, and a marker's expansion stands in its place.

Argument:
1. The spec already has implicit region constraints, stated today as separate faults: a region inside its access dimension (S1.3.3) and two outputs apart (S1.3.4); under owner 30 they are generated constraints violated like any other (S1.3.5).
2. The input-output case (S1.3.6, pending this panel) joins them: the generator emits, for each input k and output j of one buffer, the implicit constraint `apart(In k, Out j)`.
3. A constraint can only remove runs, never permit one another constraint forbids (instructions.md:38), so a marker that permits overlap cannot append beside the implicit `apart`; it must take its place for the pairs it names.
4. That is the one rule beyond appending: a pair named by an overlap marker gets the marker's expansion instead of the implicit `apart`; an unnamed pair keeps `apart`.
5. Expansions, each one meaning:
   - `set_apart(k)`: `apart(In k, Out 0)`, the implicit constraint stated explicitly, kept as shorthand under owner 30.
   - `set_no_partial_overlap(k)`: `same(In k, Out 0) or apart(In k, Out 0)`, covering owner 22's `r0 = r1 + r2` with `r2` equal to `r0`.
   - `set_inplace(k)`: `same(In k, Out 0)`, plus the static checks owner 24 and S5.5 already give (one buffer, equal sizes by text).
   - a may-overlap marker, if the panel admits one: the empty expansion, so the author adds a narrowing constraint such as RVV's positional rule.
6. Vocabulary required: two region predicates, `apart(R, S)` (no shared element: some dimension where one interval ends before the other starts) and `same(R, S)` (equal starts and equal sizes in every dimension).
7. `same` must compare sizes as well as starts, fixing F5, where today's expansion passes equal starts with unequal sizes.
   Under owner 34 both predicates and the implicit bounds compare evaluated starts and sizes, whatever expression gives them (for example `1024 - @a.addr_out` with a negative attribute), and no expansion reads an attribute's name.
   Identical start text is therefore only a shortcut that proves equality; texts that differ but evaluate equal satisfy `same` at issue.
8. Both predicates already exist as expanded text (taidl/taidl/instruction.py:212-226), and the allocator holds the same disjunction (constraint.cc:88-103), so naming them adds no new arithmetic to either generator.
9. Naming them, rather than expanding to per-dimension inequalities, lets the backend read the pair relation it needs for allocation and copies (Q9.4) without pattern-matching expanded text.
10. `set_inplace` additionally tells the backend which input the instruction destroys, which a predicate alone does not (panel-1 Q1.3 step 1; F8, where XLA declares the aliased pair); that fact rides on the marker, and its run-time content is the `same` constraint.
11. The value of every pair the constraints permit to overlap is the snapshot (Q9.1 step 3); permission is the author's declaration, so no default hazard model is assumed (owner 31).

Evidence: step 1 spec-draft overview.md:40-42; step 2 overview.md:43; step 3 instructions.md:38; step 5 taidl/taidl/instruction.py:174-230, instructions.md:49-52, owner-decisions.md:21,23; step 6-8 instruction.py:212-226, constraint.cc:88-103, F5; step 10 panel-1/decision.md Q1.3, openxla.org/xla/aliasing (fetched).
Confidence: medium-high on implicit `apart` plus replacement, medium on admitting a fourth (may-overlap) marker.
What would change my mind: no cited ISA needing a partial overlap other than through RVV-style positional rules, which would let `set_no_partial_overlap` plus a user constraint replace the fourth marker; or a ruling that markers only append, which forces the implicit default to be no constraint and so read-before-write, against owner 31.

### Q9.3

Position: options B and C as one: the stated default is the implicit `apart` constraint, and every violation (implicit, marker, user) is the one constraint fault S1.3.5; nothing is rejected at generation for mere possibility.

Argument:
1. Owner 30's clarification gives one violation outcome for all three constraint sources, and S1.3.5 already makes that outcome a fault, so S1.3.3, S1.3.4, and S1.3.6 fold into S1.3.5.
2. Overlap is a property of evaluated addresses, which are host integers at issue (panel-1 Q1.9), so the check is exact per issue.
3. An undeclared overlap violates the implicit `apart` constraint (Q9.2 step 2), so it faults like any constraint; this is option C (a stated default) whose violation is option B.
4. The default assumes no hazard model, so it meets owner 16 and owner 31 together, and its sentence is the same as for every other constraint.
5. It fails no scenario a real architecture allows, because a marker or user constraint can permit each one (Q9.1, Q9.2); an undeclared overlap is the author's statement that the ISA does not allow it, as a reserved RVV encoding does (F6).
6. Rejecting at generation whenever overlap is possible (A) would reject nearly every same-buffer pair with attribute starts, while most corpus instructions carry no marker.
7. Overlap is defined on elements, so `apart` and `same` compare element sets; for box regions that is the per-dimension interval test the allocator uses, and a strided region (L19) must be compared by elements, not bounding boxes.
8. Constraint evaluation order is the existing one: implicit and marker constraints read the state after updates, as their starts do (instructions.md:37), so one issue sees one set of addresses.
9. Static rejection stays only where text alone settles it: identical input and output text whose pair no overlap marker names makes the implicit `apart` false at every issue, so the front end rejects it, as panel-1 Q1.3 does; texts that differ but may evaluate equal (owner 34) are left to the issue-time check.
10. The norm, one sentence: "An issue faults when any constraint fails, whether implicit, marker-generated, or written; where the constraints let an input and an output share elements, the output is computed from the input's contents before the issue."

Evidence: step 1 overview.md:40-43 (S1.3.3 to S1.3.6), owner 30 clarification; step 2 panel-1/decision.md Q1.9; step 6 corpus counts (27 marker uses in taidl/tests/test_instruction.py, one each in two backend examples); step 7 constraint.cc:88-103; step 8 instructions.md:37; step 9 panel-1/decision.md Q1.3 "Identical input and output text without the mark".
Confidence: medium-high.
What would change my mind: evidence that a common shipped ISA permits overlap so widely that authors would have to mark most instructions, making read-before-write the cheaper default to state.

### Q9.4

Position: both generators read one constraint list per instruction, implicit and marker constraints included; the oracle checks it and computes the snapshot; the backend allocates under it; the corpus runs a marker-by-placement matrix on both.

Argument:
1. The front end emits the full list (implicit bounds, implicit `apart` per unnamed pair, marker expansions, user constraints) into `constraint_asts`, so neither generator re-derives a default.
2. Oracle: per issue it evaluates every constraint and faults on the first false one, then computes from loads taken before any store (F2 already gives the value); the cost is one host interval test per pair.
3. Backend: address-naming constraints already reach CP-SAT (F4), so the implicit `apart` constrains allocation like any written constraint.
4. Backend: the touch exception in the interference graph is then wrong for an unnamed pair whose input dies at the instruction, because it silently assumes read-before-write (F1); the implicit `apart` closes it, or the exception applies only to marker-named pairs.
5. Backend: `set_inplace` keeps `fix_inplace` copies and forces equal offsets (inplace_fix.rs:59-80; panel-4 Q4.6); a pair permitted to overlap needs no copy, since its value is the snapshot.
6. Backend as producer never emits a faulting issue (panel-1 Q1.12), so its allocation satisfies every constraint of every source, checked on the assembled program (panel-6 Q6.2).
7. Corpus: a matrix of five declarations (no marker, `set_apart`, `set_no_partial_overlap`, `set_inplace`, may-overlap if admitted) by three evaluated placements (apart, exact, partial), each case expecting a value or the one constraint fault.
8. Plus one positional case modeled on RVV widening (overlap only at the destination's highest part) as a user constraint, one F5 case (equal starts, unequal sizes), and one owner-34 case whose starts differ in text but evaluate equal, with a negative attribute.
9. Plus a backend case per declaration where an input dies at the instruction, asserting the allocator overlaps only where the constraints permit and the oracle agrees on the result.

Evidence: step 1 taidl/taidl/instruction.py:216-230 (`append_constraints`); step 2 instruction_codegen.py:162-181; step 3 malloc_instructions_h_generator.py:510; step 4 constraint.cc:148-180, 206-235; step 5 inplace_fix.rs:59-80; step 6 panel-1/decision.md Q1.12, panel-6/decision.md Q6.2; step 8 RVV spec Vector Operands section (fetched).
Confidence: medium.
What would change my mind: a backend measurement showing the implicit `apart` of step 4 costs real allocation capacity on corpus kernels, which would argue for authors marking may overlap broadly, not for changing the default.

## Escalation to the owner

- None; the positions follow owner 31 and note where it supersedes earlier panel steps.
