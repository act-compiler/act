# Panel 9, round 1: Sonnet (ISA author lens)

Binding inputs applied: owner 16, 17, 22, 24, 30 (markers stay as shorthand), the clarification of 30 (implicit, marker-generated, and user-written constraints are one mechanism with one violation outcome), and 31.
Round-1 order of work: code read, then RVV spec text fetched, then AMX and NKI searched.
Spec-draft S5.3 already states the one-mechanism rule and S5.5 leaves each expansion pending; this file proposes the expansions.

## Facts the answers rest on

- F1: `set_apart` and `set_no_partial_overlap` already append constraint text and track it per mark (taidl/taidl/instruction.py:174-188, `keep_off` at 190-228, `placed` bookkeeping).
- F2: `set_inplace` appends no constraint today: it checks the flag count and exclusivity, then stores `self.inplace` (taidl/taidl/instruction.py:163-172).
- F3: Spec-draft S5.5 says equal starts are checked at issue as S1.3.6, but nothing in the front end produces that check as a constraint, so the oracle cannot emit it through `generate_constraints`.
- F4: The backend splits constraints by whether they name an address attribute: the allocator holds those that do, the selector checks the rest (act-backend/act_backend/generator.py:65-68, `names_address` at malloc_instructions_h_generator.py:71-73).
- F5: The one fact no constraint carries is the tie: `list(instruction.inplace)` goes to `InstructionInfo` (act-backend/act_backend/instruction_info.py:112), the copy pass reads it (inplace_fix.rs:59-63), and the allocator lets tied live ranges overlap (cpp/malloc/src/constraint.cc:214-230).
- F6: The allocator lets any two tensors whose live ranges only touch share addresses at any offset (constraint.cc:206-210, `[a,b]` and `[b,c]`), so an input dying at an instruction can land partially over its output unless a constraint forbids it.
- F7: The oracle emits constraints, updates, sizes, inputs, semantics, outputs in that order (act-oracle/taidl_to/instruction_codegen.py:162-181), so every input is loaded before any output store.
- F8: Implicit region constraints are start >= 0, size >= 1, start + size <= access dimension, each a fault (spec-draft instructions.md:27, overview.md S1.3.3).
- F9: The casestudy AMX sketch fixes tile indices in the instruction name (`tdpbf16ps_2_0_1`, casestudies/amx.md:46-55), so operand distinctness holds by construction there and no constraint states it.

## External evidence

- E1: RISC-V V spec, "Vector Register Overlap" (github.com/riscv/riscv-v-spec, v-spec.adoc lines 990-1008 as fetched): overlap is legal when EEWs are equal, when the destination is narrower and overlaps the lowest part of the source, or when it is wider and overlaps the highest part of the destination; any other encoding "is reserved".
- E2: Same file, lines 4575-4579 and 4755-4757: `vslideup` and `vrgather` forbid any destination-source overlap, with the note "avoids WAR hazards on the input vectors"; line 1980: indexed segment loads forbid it for restartability.
- E3: Intel AMX `tdpbssd` family: any two of the three tmm operands equal raises #UD (GNU binutils commit "x86: Simplify check for distinct TMM register operands"; Intel SDM page for TDPBSSD lists exception class AMX-E4, via felixcloutier.com/x86/tdpbssd).
- E4: AWS Neuron NKI `nki.isa.nc_matmul` page (awsdocs-neuron, v2.26.1): with accumulate the result is added to the existing PSUM content, so the output tile is also an input; there is no separate accumulator operand.
- E5: PTX `mma.sync`: searched, not verified; sources show accumulate-in-place idiom with C and D the same registers, and one source says D registers must not overlap each other.
  Weight: low; no claim below depends on it alone.

## Q9.1: what must an ISA specification be able to declare about overlap?

Position: Option C, closed and small: three input-output relations (tied, apart, exact-or-contained) as constraint meanings, plus general constraints for any pair of regions; no hazard-order declaration.

1. Real ISAs state exactly four things: operands tied (AMX accumulator, E3 and F9; NKI accumulate, E4), operands forbidden to overlap (E2, E3), operands allowed to overlap only in a pattern (E1), and operands free to overlap (RVV same EEW, E1).
2. Free overlap needs no declaration: owner 22 makes exact overlap legal, and a read-before-write value is already what S5.2 and F7 give.
3. Tied is "same buffer, equal sizes, equal starts" (owner 17 and 24); it is the one relation that also needs the tie fact for the backend (F5).
4. Forbidden is a disjointness predicate; it is what `set_apart` appends (F1).
5. Pattern overlap is the RVV narrowing and widening rule (E1).
   Low-part narrowing is "starts equal or apart" with a smaller output, which `set_no_partial_overlap` already appends (F1).
   High-part widening is "ends equal", which no marker states; it is the hand-written constraint `out_start + out_size == in_start + in_size`, so the vocabulary must allow sums of starts and sizes (it does; F1 builds such text with `term`).
6. AMX distinctness is an input-input relation (E3: any two of three operands equal faults), so the declarable set cannot be limited to input-output pairs; plain constraints state it for parametric tile indices (`@a.s1 != @a.s2`).
7. A hazard-order declaration (option B) is rejected for v3: it would declare writes-before-reads, which needs a sequential-semantics model owner 31 says TAIDL does not have, and no cited tensor-ISA site needs it (admission rule: panel-2 Q2.3, two public sites).
   An ISA whose hardware output depends on order is declared illegal with apart, which turns it into a defined fault (owner 16).
8. The set is closed by construction: every relation is a predicate over starts and sizes, which is what a constraint is.

Evidence: steps 1, 5, 6 cite E1 to E4; steps 3, 4 cite F1, F2, F5; step 7 cites panel-2/decision.md Q2.3.
Confidence: medium-high.
Would change my mind: a public tensor-ISA document where inputs are read after some output elements are written and the result is defined (an order declaration then has two sites); or a relation, such as strided interleave, that no start-and-size predicate can state.

## Q9.2: how are the declarations written, and what does each marker expand to?

Position: Option D under owner 30: markers are shorthand that expand into the one constraint mechanism; the expansions below are the proposal.
Rule for which is which: a relation over start and size expressions is a constraint (written by hand or expanded from a marker); the tie is the only extra datum, and it rides on `set_inplace` beside its constraints.

1. Implicit, marker-generated, and user-written constraints are one list with one outcome, a fault (spec-draft S5.3, S1.3.5); the markers must append to that list and nothing else.
2. `set_inplace(k)` expands to this, and F2 and F3 show today it expands to nothing.
   - Declaration check, as a rejection: same buffer and equal sizes by text; different buffers are rejected (owner 24).
   - Constraints: for each dimension whose start text differs, `out_start == in_start`; if all start texts are identical, none.
   - Recorded datum: the tie (output 0 reuses input k), read by the copy pass and the allocator (F5).
3. `set_apart` expands as in F1: one disjunction over the differing-start dimensions of `out_end <= in_start` and `in_end <= out_start`.
4. `set_no_partial_overlap` expands as in F1: for each differing-start dimension, `starts equal or apart`.
5. The vocabulary these need is: `==`, `<=`, `+`, `or`, `and`, and references to a region's start and size expressions as spelled; all exist (F1).
6. The tie is not a predicate on attribute values: it names a dataflow edge, so it cannot be recovered from constraint text without inferring intent from `==`.
   Explicit recording avoids that inference and matches how placement marks already keep `placed` (F1).
7. A violated expansion is an ordinary constraint fault: the equal-start fault of S5.5 (S1.3.6) folds into S1.3.5, so the table in overview.md loses a row, and the oracle emits all of it through its existing constraint emitter (F7).
8. Timing: spec-draft S5.3 now has every constraint read the pre-issue state, so expanded constraints and the spelled starts agree and a hand-written equivalent behaves identically; the earlier two-timing trap is closed.
9. Gaps in marker reach, not in vocabulary: placement marks need exactly one output (F1, `keep_off`), and no marker covers input-input pairs (Q9.1 step 6); both are handled by hand-written constraints, and a marker is added only under the Q2.3 rule.
10. Owner 34 (a start is any expression over attributes, possibly negative; implicit constraints test the evaluated address) changes no expansion.
    Every expansion compares start and size expressions as spelled, and identical text is identical value under the pre-issue state, so skipping a dimension with identical text stays sound.
    A start such as `1024 - @a.addr_out` enters `out_start == in_start` or the apart sum unchanged; the implicit range constraints are separate constraints on the evaluated value, and an expansion never assumes a start is non-negative.
    The allocator split (F4) keys on the `@a.` prefix (taidl/taidl/antlr4/IDLV2Expr.g4:34), which is the attribute kind and not a name, so owner 34's "no tool infers meaning from a name" holds.
11. Text identity is not required for a tie, and I do not argue for removing the flag (owner 30).
    A tie spelled with different start text expands to the equality constraint, so the same fact is stated either way.

Evidence: F1 to F5, F7; spec-draft instructions.md:27-39 and S5.5; overview.md S1.3.3 to S1.3.6.
Confidence: medium-high on 1 to 5 and 7, medium on 6.
Would change my mind: a finding that the backend can use the equality constraint alone to place the copy and the allocator permission, making the recorded tie unnecessary.

## Q9.3: what happens on a violation, and on undeclared overlap?

Position: a violation is a fault, the same as any constraint; undeclared overlap, exact or partial, is legal and reads before it writes (option C, a stated default), which revises panel 1's partial-overlap fault.

1. Violation: implicit, marker, and user constraints share one outcome, a fault at issue before any write (owner 30 clarification; spec-draft S5.3; panel-1 Q1.12 atomicity).
   Owner 16 is met: the outcome is defined.
2. Constraints and markers only remove scenarios; none can add one.
   Therefore the default must be the largest legal set, or an ISA that allows an overlap cannot say so, and owner 31 ("must not fail scenarios real architectures allow") breaks.
3. RVV shows this is not hypothetical: `vnsrl.wi v0, v0, 3` overlaps the lowest part of its source and is legal (E1), and a default fault on partial overlap would reject it with no way to opt in.
4. Panel 1 faulted partial overlap by default (Q1.3 steps 7-9); its cited ISAs are the cases an author forbids, which is the job of `set_apart` and `set_no_partial_overlap`, and E1 is a case it must allow.
5. One norm for the whole language: "An issue reads every input before it writes any output; overlap changes no value; an overlap the hardware forbids is a constraint, and violating it is a fault."
   It is shorter than panel 1's two norms (exact legal, partial fault), which serves owner 16's "simple norm".
6. Panel 1's ergonomic objection (a systolic author would need a disjointness constraint on every instruction) is answered by owner 30: it is one flag per input on `set_apart`.
7. Option A (reject the specification at generation when overlap is possible) fails because starts are expressions over run-time state, so possibility is nearly always true and it would reject valid specifications.
8. Option B (a run-time fault on any undeclared overlap) is panel 1's fault default and fails step 2.
9. Risk: an author who omits a declaration gets an oracle value the hardware may not give.
   Owner 31 puts the declaration on the author, and a lint can warn on a marker-free instruction whose regions share a buffer, without changing semantics.
   Unlike panel 1's Q1.4 uninitialized read, the author has a way to say it.

Evidence: E1, E2; F6, F7; panel-1/decision.md Q1.3 steps 5-9 and Q1.12; owner decisions 16, 22, 31.
Confidence: medium (0.6); this reverses a panel-1 medium ruling, so the moderator should route it to the owner if the panels split.
Would change my mind: a public tensor-ISA document that defines partial overlap as a fault and none that permits it, and a count of authors' omissions showing silent mismatches; or a ruling that the oracle must also model hardware garbage.

## Q9.4: what does each generator do, and how does the corpus check it?

Position: no generator branches on a marker; each reads one constraint list, the backend additionally reads the tie, and the corpus has cases per relation including the rejections.

1. Front end: expand every marker at declaration time into the shared constraint list, together with the region's implicit constraints; reject malformed marks (cross-buffer, size mismatch, more than one tie) there.
2. Oracle: emit the list through the existing constraint emitter, evaluated on the pre-issue state before any store, so a violation faults the whole issue (F7, panel-1 Q1.12).
   Values need no per-relation code: all inputs load before any store, so the declared value is read-before-write under every relation.
3. Oracle duty added: nothing beyond the marker expansion now emitted for `set_inplace` (F2, F3); the equal-start check is no longer separate code.
4. Backend selector: check the non-address constraints; backend allocator: hold the address-naming constraints, which every marker constraint is (F4).
   Apart and no-partial constraints therefore reach the allocator unchanged.
5. Backend tie: the copy pass gives a shared tied operand its own copy and errors where it is pinned in HBM or a view (inplace_fix.rs:6-17, 59-80); the allocator lets tied live ranges overlap (constraint.cc:214-230).
6. Backend soundness trap: with touching live ranges the allocator may overlap an input and an output at any offset (F6).
   Under Q9.3 that is legal by the language, so correctness rests on the author's declarations; the backend must add no overlap rule of its own, and a systolic ISA that forgets `set_apart` is the author's error.
7. Backend never emits a faulting issue (panel-1 Q1.12 step E): the allocator solution must satisfy every instruction constraint, including the tie equality; if no placement exists, the result is an error, never a wrong kernel (owner 27).
8. Corpus, per relation, each with one passing and one faulting case:
   - tied: equal starts pass, unequal fault, cross-buffer rejected at declaration;
   - apart: disjoint pass, overlapping fault;
   - exact-or-contained: equal starts pass, apart pass, partial fault (RVV pattern from E1, low-part narrowing);
   - high-part widening written as a hand constraint (E1);
   - input-input distinct (AMX, E3) as a hand constraint;
   - undeclared overlap: exact and partial both give the read-before-write value in the oracle;
   - an implicit constraint (start below 0) and a user constraint fault by the same code path, checking one mechanism, one outcome.
9. Backend corpus: a tied instruction whose input has another reader gets a copy; the same with the input pinned in HBM errors; an apart instruction never gets overlapping addresses in the solution.
10. Cross-check: generate marker and hand-written constraint text for the same relation and require identical oracle behavior.

Evidence: F2 to F7; panel-4 Q4.6, panel-5 Q5.6, panel-6 Q6.2 decisions not re-read this round, so steps 8 to 10 are proposals against the agenda's stated need.
Confidence: medium-high on 1 to 5 and 7, medium on 6, 8 to 10.
Would change my mind: a panel 4, 5, or 6 ruling that fixes a corpus format or generator split these steps conflict with.

## Escalation to the owner

- Q9.3: panel 1 faults partial overlap by default; I argue it must be legal because constraints only subtract and RVV legalizes narrowing overlap (E1), so owner 31 and a default fault conflict; owner confirmation requested if the panels split.
