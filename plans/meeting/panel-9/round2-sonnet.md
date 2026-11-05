# Panel 9, round 2: Sonnet (ISA author lens)

Binding inputs applied: owner 16, 17, 22, 24, 30 and its clarification, 31, 34.
Evidence labels E1 to E3 are the round-1 sonnet external facts (RVV overlap rules, AMX distinct tmm operands); F-labels below are new to this file.

## The decision 31 test, applied to all three round-1 files

- All three files compute the value of an overlapping input-output pair from the inputs' pre-issue contents.
- Opus and Fable justify it as "the only value definable without a hazard model" (round1-opus Q9.1 step 3; round1-fable Q9.1 step 4).
- That is a rule about when reads happen relative to writes, which is read-before-write by another name.
- Owner 31 says TAIDL assumes no such model, and the author declares how an instruction behaves under overlap.
- My own round-1 norm ("an issue reads every input before it writes any output", Q9.3 step 5) is the strongest form of the assumption: it applies to every overlap the author said nothing about.
- Two readings of 31 exist, and the files do not separate them.
  - Reading X: the pre-state value is what an instruction's semantics means (a function of its input values), so it is not a hazard model, and the author's declaration only says which overlaps the hardware permits.
  - Reading Y: any value for an overlapping pair is a claim about hardware order, so the language may not supply one; each overlap-permitting declaration carries the claim "the hardware computes the pre-state value", and an undeclared overlap has no defined value.
- Under Y, owner 16 forces a fault for the undeclared case, because a defined behavior needs a value.
- Under X, a larger default is coherent.
- Which reading holds is a question for the owner; the escalation section states it.
- What an author does when the hardware differs, under either reading:
  - The author declares the narrower relation (apart, or exact-or-apart plus a positional constraint), so the differing scenarios become defined faults (owner 16).
  - The kernel cannot use those scenarios, which is a loss, but never a wrong value.
  - If the hardware defines a non-pre-state value (a forward-copying DMA smear), no relation in any round-1 file can state it; only an author-declared order (option B) could.
  - That gap is the real exposure of 31's last clause, and no cited tensor-ISA site needs it today (round1-fable table: the shipped hazards are temporal, not within-instruction).

## Facts checked this round

- F-a: `keep_off` builds `equals` from start terms only, so `set_no_partial_overlap` with equal starts and unequal sizes passes although the regions overlap partially (taidl/taidl/instruction.py:211-224, `equals.append(... == ...)` with no size term); round1-opus F5 is correct.
- F-b: `set_inplace` appends no constraint (taidl/taidl/instruction.py:163-172); unchanged from my round 1.
- F-c: owner 22 states `r0 = r1 + r2` "allows `r2` to equal or differ from `r0`" with no marker named (owner-decisions.md:21).

### Q9.1

Strongest opposing argument (Fable, round1-fable Q9.1 steps 3 and 6): a fourth relation, "any overlap, pre-state value", is needed as an opt-in, because partial overlap is legal on some hardware (RVV narrowing) yet must not be the default.
My round-1 answer had three relations and made partial overlap legal by default, so it needed no opt-in; Fable's structure is the stronger one under reading Y and under owner 22.
Concession: my claim that constraints only subtract, so the default must be the largest set (round 1 Q9.3 step 2), assumed append-only markers; Opus (Q9.2 step 3, a named pair takes the marker's expansion in place of the implicit constraint) and Fable (Q9.2 step 7) both show a selection rule removes that need.
Rebuttal 1, against Opus Q9.1 step 2 ("input-input overlap is reads only and needs no relation"): AMX `tdpbssd` faults when any two of its three tmm operands are equal (E3), so an input-input distinctness relation exists in a shipped ISA, and the source-versus-accumulator pairs are apart relations on an instruction whose accumulator is also tied.
It is a hand-written constraint (`@a.s1 != @a.s2`), not a fifth marker, so the relation set stays closed; Opus's claim is wrong only as a statement about what the vocabulary must reach.
Rebuttal 2, against my own round-1 step 7: I rejected a hazard-order declaration (option B) because owner 31 "says TAIDL does not have" a sequential model.
That misreads 31: the language must not assume a model, but an author-declared one is exactly what "the ISA author declares how an instruction behaves" permits.
B stays out of v3 for the evidence reason only (panel-2 Q2.3 admission rule, two public sites; none found), and the vocabulary should leave room for it.
Final position: option C, four per-pair relations between input k and output j of one buffer (tied, apart, exact-or-apart, any-overlap), everything finer a constraint, input-input rules as constraints, no hazard-order declaration in v3.
Confidence: medium-high.
Still divides the panel: whether any-overlap is a marker with a v3 site (Fable: no case study sketches one; my RVV narrowing is the only evidence, and RVV is not a tensor ISA).

### Q9.2

Strongest opposing argument (Opus, round1-opus Q9.2 steps 4, 6, 7): name two region predicates, `apart(R, S)` and `same(R, S)` with `same` comparing starts and sizes; every marker expands to them, and a marker that permits overlap replaces the implicit constraint for its pair.
Concession 1: `same` must compare sizes; my round-1 expansion for `set_no_partial_overlap` ("starts equal or apart") inherits F-a, and my claim that the expansions "already exist" (round 1 Q9.2 step 4) was too generous.
Concession 2: named predicates are better than expanded text, because they give the backend a pair relation without pattern-matching and give a user a readable hand-written equivalent (the cross-check of round-1 Q9.4 step 10 then compares predicates, not strings).
Concession 3: a permissive marker is not an append; it selects the pair's relation, and the implicit relation for an unnamed pair is what the selection replaces.
That is compatible with owner 30 only if the implicit overlap constraint is treated as the expansion of an unwritten default relation, the same way the bounds constraints are generated; owner 30 does not say a marker may remove a generated constraint.
I flag this in the escalation as a point to confirm, not a counter.
Rebuttal against Opus and Fable on the tie: both keep `set_inplace` expanding to `same(In k, Out 0)`, and I agree, but the recorded tie must stay a separate datum; a `same` predicate alone cannot tell the backend which input dies, since `set_no_partial_overlap` can also produce equal placements.
Evidence: inplace_fix.rs:59-80 and constraint.cc:214-230 consume the tie (round-1 F5).
Rule for which is which: a relation over starts and sizes of a named pair is a marker expansion in predicates; anything else (input-input, positional, strided) is a hand-written constraint; the tie is the only datum beyond constraints.
Expansions under owner 34 compare evaluated start and size terms, never attribute names, and the implicit range constraints stay separate.
Final position: option D; markers `set_inplace` (-> `same`, plus the tie), `set_apart` (-> `apart`), `set_no_partial_overlap` (-> `same or apart`, the explicit spelling of the default), and an any-overlap marker (-> no constraint, replacing the default for its pair).
Confidence: medium-high on predicates and the size fix; medium on replace-versus-append being within owner 30.
Still divides the panel: whether the tie datum is separate (me, Fable) or recoverable from `same` (Opus step 9 leans on predicates, step 10 keeps the marker); and whether the any-overlap marker exists.

### Q9.3

The three defaults against owners 16 and 31:

| Default for undeclared overlap | Owner 16 (defined, simple norm) | Owner 31 first clause (no hazard model assumed) | Owner 31 last clause (no allowed scenario failed) | Owner 22 (`r2` may equal `r0`, no marker named) |
| ------------------------------ | ------------------------------- | ----------------------------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| Apart (Opus) | yes, a fault | yes, assumes nothing about values | yes only if the marker is available and authors use it | conflicts if 22 describes the unmarked statement |
| Exact-or-apart (Fable) | yes, a fault on partial | partly: supplies a value for exact overlap | same as apart, for partial | consistent |
| Any overlap legal (my round 1) | yes, no fault | no under Y: supplies a value for every overlap | yes | consistent |

Strongest opposing argument (Opus, round1-opus Q9.3 steps 3 to 5): the default is the implicit `apart`; an undeclared overlap is the author's statement that the ISA does not allow it, as a reserved RVV encoding does (E1); nothing real is lost because a marker permits each scenario, and the default assumes no hazard model.
Rebuttal, in two parts.
- Part 1: apart default makes every elementwise instruction with a same-buffer output carry a marker, and owner 22 describes `r0 = r1 + r2` as allowing `r2` equal to `r0` with no marker (F-c); I read 22 as a statement about the unmarked statement, so apart conflicts with it.
  If the owner meant only "the author can allow it", apart is permitted and is the lowest-assumption choice.
- Part 2: Opus's argument that apart "assumes no hazard model" is right, and it is the only default that survives reading Y with no help from the owner; I concede it is the safest ruling.
Rebuttal against Fable's partial-overlap fault (round1-fable Q9.3 step 5): I now agree with the fault, with a better reason than the one it gives.
- For an elementwise op, an exact overlap gives the same value under every write order, because each output element depends only on its own input element.
- A partial overlap gives an order-dependent value even for an elementwise op, because output element i is input element i+d.
- So the exact-or-apart default is the largest set whose value does not depend on an unstated order, for the elementwise cases owner 22 names.
- It does not hold for a non-elementwise op (an in-place matrix product over its own operand), where exact overlap is order-dependent; that author must declare apart, and the oracle value is the pre-state value, so a lint should flag any unmarked same-buffer instruction that is not elementwise.
Concession against my own round-1 Q9.3: any-overlap legal fails under reading Y, and its supporting argument (constraints only subtract) is answered by relation selection.
I withdraw the reversal of panel 1 and its escalation of partial overlap.
Option A (reject at generation) fails for all files, since starts are run-time expressions.
Violation outcome: the one constraint fault at issue, before any write (owner 30 clarification).
Final position: option C with exact-or-apart as the stated default per same-buffer input-output pair, violations are the one constraint fault; apart is the fallback if the owner reads 22 as author-permission only; any-overlap is available as the declaration.
Changed position: yes, from any-overlap legal to exact-or-apart.
Confidence: medium (0.6) between exact-or-apart and apart; high that any-overlap legal is wrong under reading Y.
Still divides the panel: Opus (apart) versus Fable and me (exact-or-apart), which turns on how to read owner 22.

### Q9.4

Strongest opposing argument (Opus, round1-opus Q9.4 step 4): the allocator's touching-live-range exception silently assumes read-before-write, so an unnamed pair whose input dies at the instruction can be placed over its output; the implicit constraint must reach the allocator.
My round 1 (Q9.4 step 6) said the backend adds no overlap rule of its own because the language makes overlap legal.
That was correct only under my old default; with a stated exact-or-apart default it is wrong, since a producer must never emit a faulting issue (panel-1 Q1.12) and the allocator can currently place a partial overlap (round-1 F6, constraint.cc:206-210).
Concession: the front end emits the implicit exact-or-apart constraint per unnamed pair into the same list as bounds and marker constraints; it is address-naming, so it reaches the allocator (round-1 F4).
Concession 2 (Fable, round1-fable Q9.4 steps 4 and 8): if exact-or-apart blows up the OR-Tools model, the backend may use apart only, which is stricter than the oracle and still never faults; I add that this fallback must be recorded as a backend limit, not a language rule.
Rebuttal against Fable Q9.4 step 5 ("any overlap: no constraint ... the copy pass is not needed"): agreed for the value, but the tie datum is still the only reason a copy is needed; keep it unchanged for the tied pair only.
Corpus: adopt Opus's matrix (declarations by evaluated placements: apart, exact, partial) as the base.
Add my cases: input-input distinct (AMX, E3) as a hand constraint; an F-a case (equal starts, unequal sizes) expecting a fault under exact-or-apart; the marker versus hand-written cross-check (also Fable step 7), compared as predicates; an owner-34 case with differing text and equal evaluated value, with a negative attribute.
Add a lint case: an unmarked same-buffer non-elementwise instruction warns, with no change to semantics.
Final position: the front end emits one constraint list (bounds, implicit relation per unnamed pair, marker expansions, user constraints); the oracle checks it before any store and computes the pre-state value; the backend allocates under it and keeps the tie datum for copy and offset equality; the corpus has a legal, a faulting, and a rejected case per relation.
Confidence: medium-high.
Still divides the panel: nothing material, except the cost of exact-or-apart in the solver (Fable's open measurement).

## Escalation to the owner

- Decision 31, reading. All three round-1 files, and I in round 1, supply the pre-state value for any overlap the language accepts. Read strictly ("assumes no hazard model such as read-before-write"), that is a language-wide read-before-write rule. Evidence: round1-fable Q9.1 step 4, round1-opus Q9.1 steps 3 and 4, round1-sonnet Q9.3 step 5. Please say whether the pre-state value is the meaning of an instruction (reading X) or must be an author claim attached to each overlap-permitting declaration (reading Y). Under Y an undeclared overlap faults by owner 16, and the default is apart or exact-or-apart.
- Decision 31, last clause. No relation in any round-1 file can express hardware that defines a non-pre-state result for an overlap, such as a forward-copying DMA. If the owner wants those architectures expressible, an author-declared order (option B) is needed; no case study supplies a site for it yet. Please say whether v3 must be able to express it.
- Decision 22 against a fault default. If 22 means the unmarked `r0 = r1 + r2` must run when `r2` equals `r0`, apart as the default conflicts with it; please confirm that reading, which decides apart versus exact-or-apart.
- Decision 30. A permissive marker (any-overlap) replaces a generated constraint instead of appending one. Please confirm that this is within "each marker expands to the constraints it stands for", with the unwritten default relation as the thing replaced.
