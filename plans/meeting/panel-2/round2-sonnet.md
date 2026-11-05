# Panel 2, round 2: Sonnet (new user writing an ISA description)

Evidence verified this round:
- Backend use of the mark: `act-backend/act_backend/generic/src/isel/extractor/inplace_fix.rs:59-63` reads `inplace_children()` per node and copies a shared operand before an overwriting instruction.
- That pass runs on the program graph before any address exists, so it needs a static fact per instruction variant.
- Term counts, text extracted from the two owner papers: ACT (OOPSLA 2026) has "ISA description" 53 times and "TAIDL definition" 0 times.
- TAIDL (MICRO 2025) has "TAIDL definition" 28, "ISA definition" 15, "ISA description" 1.
- LLVM CodeGenerator page: instructions are "three address" with rare exceptions; x86 `ADD %EAX, %EBX` is the two-address `%EAX = %EAX + %EBX`.

### Q2.1

Strongest opposing argument (Opus and Fable, Option C):
- Every real in-place case is an accumulator whose input and output regions are spelled identically (AMX tile, Hopper `wgmma` `rf`, NKI PSUM accumulate, mini_trn softmax).
- So text identity already says "same storage", the flag restates it, and any other overlap should fault.

Rebuttal and concession:
- Concession: for the identical-text subset the flag is redundant as information; Opus and Fable are right there.
- I withdraw my round-1 blanket fault on other overlap; owner 16 does not force it.
- Test `test_extraction.py:507` (`d0` input, `d2` output) cannot share storage under owner 17, so it migrates and is not evidence for the mark.
- Rebuttal 1: the owner's input shows text identity expresses only "must reuse".
- Identical text forces one address, so it can never say "may reuse".
- "May reuse" is the unmarked form with two attributes, as in `r0 = r1 + r2` where `r2` happens to equal `r0`: the result is defined by read-before-write (`language/spec-draft/instructions.md:54`, S5.2), and no fault is warranted.
- Option C's "any other overlap of an input and an output is a fault" faults that case, and a three-address instruction with `rd == rs2` is the common case on real ISAs (LLVM page: three-address is the norm).
- Rebuttal 2: the "must" form with two differently spelled starts (a tied operand, such as `@a.src` and `@a.dst` that the hardware requires equal) is not a text identity.
- A constraint `@a.src == @a.dst` states the fault, but gives the allocator no clobber fact.
- Rebuttal 3: the backend consumes the flag as that clobber fact before addresses exist (`inplace_fix.rs:59-63`); evaluated equality (Option D) arrives too late, and a constraint is a text pattern no pass should match.
- Rebuttal 4: with derivation, an identical-text pair is silently a clobber; a description author who forgets this gets a miscompile when another reader shares the operand, because the pass would not insert the copy.
- So the redundancy is one-directional: the flag is derivable only when the texts are identical.
- Read-after-write hazard: S5.2 defines every overlap by read-all-then-write-all, so overlap alone never produces a wrong result in the model.
- A pipeline that streams element by element and cannot honor that states a constraint; the spec names the constraint idiom (see Q2.2).

Final position: keep `set_inplace` as the "must reuse" (tied operand) declaration, with the front end checking both directions.

- A true flag k requires the same buffer, equal size texts, and equal starts at issue, else a fault (S5.5 as drafted).
- An output region textually identical to an input region without the flag is rejected at declaration, so the clobber fact is never silent.
- Unmarked overlap of any other kind is defined by S5.2, not a fault.
Confidence: medium.

What still divides the panel:
- Opus and Fable derive the flag from text; I keep one explicit declaration (tied pair, backend fact before addresses).
- Whether the default should fault on partial overlap (apart or identical) is open; it is a norm choice, and I do not require it.

### Q2.2

Strongest opposing argument (Opus, Option A):
- Under Q2.1 C an undeclared overlap already faults, so `set_apart` is the default and `set_no_partial_overlap` is "identical or apart", which Q2.1 C also gives.
- A helper is a third spelling of one rule.

Rebuttal:
- With the blanket fault withdrawn (Q2.1), unmarked overlap is defined, so a DMA-like instruction that cannot tolerate src-dst overlap (Fable's example) must now state it, and the default no longer does.
- That makes the constraint idiom necessary, not optional, and it is not a third spelling: it is the only spelling after the marks go.
- The marks still read the state at a different point (audit R7), so removal is still right.

Final position: remove both marks and document one constraint idiom for "apart" and one for "identical or apart".
Confidence: medium-high.

What still divides the panel: Opus needs a default overlap fault for Option A to hold; if the panel adopts such a default, A and C converge on removal and differ only in whether the idiom page exists.

### Q2.3

Strongest opposing argument (Fable, Option D as the procedure):
- A numeric threshold alone admits a construct that two toy sketches "need"; the real bar is a per-extension review, with the threshold and the idiom test as its inputs.
- Opus adds that a site counts only if its sketch ran through the front end.

Rebuttal and concession:
- Concession: I add Opus's condition that a counted sketch must run; `casestudies/method.md` says no sketch was run, and the Hopper sketch has rank bugs (CP15).
- Invented ISAs count zero (I, Opus, Fable agree); the half-weight option is dead.

Final position: syntax is admitted only with an idiom-failure argument and two public, in-scope accelerators whose sketches ran, counted per construct in a matrix column, for every extension including L17.
Confidence: medium-high.

What still divides the panel: who counts (the L-track lead with an outside reviewer per Opus, "the panel" per Fable) and whether "idiom fails" includes "idiom is only verbose" (all three say no; Fable would admit an idiom that loses information).

### Q2.4

Strongest opposing argument (Opus on `scaled_dot`; Fable on L19 and L23):
- Opus: `scaled_dot` is `convert`, `multiply` by a broadcast power-of-two scale, then `dot`; the scale multiply is exact, so the idiom is lossless and only the two MX types need syntax.
- Fable: L19 is universal (tensor-map strides, NKI `dma_copy`, AIE-ML buffer descriptors) and L23's `gather` idiom needs a bounded covering input that real indirect DMA does not have.

Rebuttal and concession:
- Concession (Opus, `scaled_dot`): under the rule I defend in Q2.3 a one-word shorthand for an exact idiom does not earn syntax; I drop `scaled_dot` from v3.
- L17 in v3 is `f4e2m1fn` and `f8e8m0fnu` only, with `tf32` dropped as a storage type (CP11).
- L19 and L23: Fable's failure claims are plausible but unrun; the covering input for indirect DMA is all of `d0`, whose size is an issue-time parameter (RA10), which is costly but expressible.
- I accept Fable's ordering: L19 then L23 are the first candidates, and the C2 sketches decide each by a failed idiom, not by opinion.

Final position: v3 admits the two MX types; every other L item waits for a failed-idiom demonstration, with L19 then L23 first in line; L22 is cut.
Confidence: medium (low on L19 and L23 ordering).

What still divides the panel:
- Fable commits L19 and L23 as next after v3; Opus keeps both as idioms until shown otherwise; I sit between, deferring to the Q2.3 test.
- Whether the second MX site is XDNA2 or Gaudi3 (CP11) needs the owner's confirmation; Fable and I flag it, Opus does not.

### Q2.5

All three agree: no new state kind in v3 (Options C and D); data-dependent addressing goes to one study with L23.
Final position: no new state kind; descriptors use existing state, attributes, and a region read.
Confidence: medium.

### Q2.6

All three agree on Option B: out of v3, the spec says program order is the only order, and a constraint used as a wait faults when false.

Final position: Option B.
Confidence: high.

### Q2.7

Strongest opposing argument (Fable, Opus):
- Fable: NVDLA has a public register-level manual and fills the descriptor-engine class, while Ascend's instruction level is not public and its site was unreachable.
- Opus and Fable: add the GPU tensor-core description now (Option E), since owner 15 names `mma`, `wgmma`, and TMA and R47 plans it anyway.

Rebuttal and concession:
- Concession (E): the Hopper sketch is the starting point and, once its missing attributes and rank bugs are fixed and it runs, it is the GPU description; I now accept it as v3 evidence, not only as a test.
- Concession (NVDLA over Ascend): `casestudies/method.md:7` records that hiascend.com was not reachable, and audit R15 finds R15's own reason argues against Ascend; if one study is added, NVDLA.
- Disagreement that remains: whether any new study is needed before the v3 decision; I say no, because correcting three verdicts and running the Hopper sketch already feeds the matrix.

Final position: correct the verdicts and recount; count only sketches that ran; run and fix the Hopper sketch as the GPU description; queue NVDLA (not Ascend) after v3.
Confidence: medium.

What still divides the panel: Fable wants NVDLA before the admission decisions; Opus and I admit it only if a decision turns on that class.

### Q2.8

All three agree on Option A: remove `cholesky`, `triangular_solve`, `batch_norm_inference`; C reduces to A because the `BOOL`, `TRANSPOSES`, and `NUMBER` forms are used by no other word (`taidl/operations.py:18-23`, `:145-156`).

Final position: Option A.
Confidence: high.

### Q2.9

Strongest opposing argument (Opus and Fable on `d0`):
- `d0` is declared with no dimensions and then special-cased to one at `taidl/instruction.py:123`.
- Declaring it as an ordinary buffer with one addressed dimension and an issue-time size removes the special case and one rule from every generator.

Rebuttal and concession:
- Concession: for a learner, "d0 is a predeclared buffer like any other, with one addressed dimension and a run-time size" is one concept; "d0 is special" is two.
- I withdraw "keep `d0` special and state it".

Other items:
- Presumed reshape removed; `constraints=` only; `ROOT` unchanged: all three agree.
- Instruction without semantics: R11 as written (legal only without regions); Opus states it as "semantics present if and only if an output region exists", which would reject an input-only instruction with no semantics.
- Fable and I agree on R11, because state-only configuration instructions (`LDTILECFG`, Gemmini `config_*`, `SMSTART`) are the norm.

Final position: remove presumed reshape; `d0` is a predeclared ordinary buffer; keep `constraints=` only; defer `ROOT`; adopt R11.
Confidence: medium-high.

What still divides the panel: whether an input-only instruction with no output region may omit semantics (Opus's rule forbids what R11 allows).

### Q2.10

Strongest opposing argument (Opus on reserved names, Fable on generation):
- Opus: reserved names belong to one generator, not the language, and the error message that reaches the user (naming the word and the generator) is the fix.
- Fable: if a table exists it must be generated from the generators' own lists, so it never drifts.

Rebuttal and concession:
- Concession (Opus): the error message must name the reserved word and the generator regardless; I add that requirement.
- Rebuttal: a new user choosing instruction names such as `copy` or `dot` benefits from seeing the list before the first generation run, and the error appears only after a full description is written.

Final position: grammar page links the `.g4` files and keeps token tables and binding notes (all agree); reserved names stay in S4 labeled as the oracle's list, and the error names the word and the generator.
Confidence: medium.

What still divides the panel: Opus cuts the table; Fable generates it; I label it.

### Q2.11

All three agree on Option C: the spec fixes the issue model normatively and only the names a generator must read (`constraint_asts`, `update_asts`, `instr_input_asts`, `instr_output_asts`, `instr_expression_asts`); the rest of the object model goes to the `taidl` README.

Final position: Option C.
Confidence: medium.

### Q2.12

Strongest opposing argument (Opus, with Fable):
- Owner 14 puts the papers first; the newer ACT paper uses "ISA description" and the file name should follow it.
- Opus rests on the ACT paper count; Fable rests on the TAIDL paper.

Rebuttal and concession:
- I checked both papers: ACT has "ISA description" 53 times and "TAIDL definition" 0; TAIDL has "TAIDL definition" 28 and "ISA description" once (as part of "scalar ISA description languages" in my round-1 reading).
- Fable's claim that the TAIDL paper uses "ISA description" is wrong for that paper; Opus's count for the ACT paper is right.
- My round-1 preference rested on the TAIDL paper alone; the ACT paper postdates it and is the one that describes the user's file as part of the full flow, so I concede.
- "TAIDL definition" survives only as the TAIDL paper's term; it is not the term for the file in the newer paper.
- Renames: attribute names that a generator or user reads (`parameters`, `comp_attr`, `var_type`) change in the v3 batch; `num_dim_str` is deleted; `array_dim_str` moves to the oracle (Opus); the grammar rule `instruction` stays (D13); the glossary drops `add_tensor_buffer` and `add_control_register` first (owner 19).

Final position: "ISA description" for the user's file, "spec" for the language document; renames as above.
Confidence: medium-high on the file name, medium on the rename set.

What still divides the panel: whether the grammar rule `instruction` is renamed in the batch (Fable would, if a parser regeneration is already scheduled; Opus notes L17 regenerates the parsers).

## Escalation to the owner

No strong counter to a binding decision; one point for the owner to confirm or correct.

- Decision 17 says the output reuses the marked input's storage.
- The evidence shows the backend reads the flag as a clobber fact before addresses exist (`inplace_fix.rs:59-63`), and that identical region text expresses only the "must reuse" case.
- The panel's choice depends on one fact that only the owner can state: whether a description may declare a tied pair whose input and output starts are spelled differently (for example two attributes the hardware requires equal).
- If yes, `set_inplace` is not redundant and stays (my position).
- If no, the flag is a checked redundancy of text identity, and the owner may still keep it as the explicit signal.
