# Panel 2, round 2: Opus (minimalist language designer)

Lens: a construct stays only if no existing construct states the same behavior without loss.
Binding inputs: owner decisions 10, 15, 16, 17, 19, 20, 21, and the owner's input on Q2.1 (keep `set_inplace` unless shown truly redundant).

### Q2.1

Strongest opposing argument (owner input; Option B in ambiguity S5.5 and `semantics/rulings.md:8-13`): overlap alone does not settle read-after-write.
In r0 = r1 + r2, r2 may equal r0 or differ, so identical text cannot be the only carrier of "reuse", and the flag states intent the texts do not.
Second argument (Sonnet, round 1 step 8): the fault on other overlap is not forced by owner 16, since S5.2 already defines the result.

Concession:
- My round-1 Option C is wrong on one point: its blanket fault on any other overlap outlaws "may reuse".
- A three-operand ALU described as r0 = r1 + r2 and issued with r2 equal to r0 is ordinary hardware use, and C turns it into a fault.
- My round-1 Q2.2 step 3 fails with it: distinct texts that evaluate equal are not "spelled as in-place".

Rebuttal on redundancy, per intent of one input and output pair:

| Intent | Spelling with no in-place flag |
| ------ | ------------------------------ |
| must reuse | identical parsed region texts (buffer, starts, sizes) |
| may reuse | distinct texts; S5.2 defines the result when they evaluate equal |
| may reuse, hardware streams | distinct texts plus a constraint "equal starts or apart" |
| must not overlap | distinct texts plus an "apart" constraint |

- Identical text expresses only "must": both regions evaluate to the same storage at every issue, so the output lands on the input's storage, which is owner 17's meaning.
- An implementation that writes a shadow copy and commits it is invisible at the level TAIDL describes, as D18 puts streaming hazards under timing (`language/plan/decisions.md:26`).
- "May" is expressed by distinct texts, and neither the flag nor identical text expresses it; so the may/must distinction does not need the flag.
- The read-after-write hazard inside one instruction is defined by S5.2 (every input read before any output written, `language/spec-draft/instructions.md:54`); the flag does not guard it, and the spec says the mark never changes the result.
- Where hardware does not meet S5.2 under overlap, the guard is a constraint (Q2.2), with or without the flag.
- The flag's only reader is the back end's copy pass: an input marked in place is "overwritten", and a copy is inserted when that value has another reader (`act-backend/act_backend/generic/src/isel/extractor/inplace_fix.rs:9-80`, flags from `act_backend/instruction_info.py:112`).
- That fact is a function of the texts: input k is overwritten exactly when its parsed region equals output region 0, the same tree comparison the front end already makes (`taidl/taidl/instruction.py:214`).
- The flag's spec form allows differently spelled starts with an issue-time fault (`instructions.md:49-52`); each such description equals one with identical text plus a constraint of equal starts, so the flag adds no expressible behavior.
- The flag adds one failure mode: flag and texts that disagree, which today needs a rejection and a fault (`taidl/taidl/instruction.py:163-172` checks only the count).
- Edge case: r0 = r0 + r0 derives two overwritten inputs where the flag allows one; the copy pass already works per child, so the rule "every identical input is overwritten" holds.
- Aliased buffers over one storage (CP16) cannot be identical text, but the flag also requires the same buffer (`instructions.md:51`), so derivation loses nothing.

Final position: Option A: an input is in place exactly when its parsed region equals the output's; no blanket overlap fault; other overlaps follow S5.2 unless a user constraint forbids them.
Fallback if the owner keeps the flag: Option B with the flag required exactly when the parsed texts are identical, so it never disagrees with the regions and needs no issue-time fault.
Confidence: medium (high that C's blanket fault must go).
What divides: Sonnet and Fable hold C with its fault; the owner holds the flag unless redundancy is shown.

### Q2.2

Strongest opposing argument (Sonnet, Fable): without the marks users re-derive the "apart or equal starts" disjunction per dimension, which is easy to get wrong (`taidl/taidl/instruction.py:200-214`).
Concession: with C's default fault gone (Q2.1), the two conditions are no longer defaults, so users need them as text.
A documented constraint idiom in the cookbook (C10) and one spec example is documentation, not a third spelling.
Final position: Option C as Sonnet frames it: remove both marks; ship the two constraint texts as a documented idiom, with no API helper.
Changed: yes, from A to C.
Confidence: high.
What divides: nothing on removal; the idiom also serves streaming hardware under Q2.1, so its text must handle strides once L19 lands (`language/plan/extensions.md:51`).

### Q2.3

All three agree: syntax needs an idiom-failure argument plus two real, publicly documented sites in owner scope, counted per construct, invented ISAs at zero, applied to every extension including L17.
Strongest residual argument (Fable): the per-extension review of Option D is the procedure, and the threshold and idiom test are its bar.
Concession: I adopt that framing; a numeric count without a review admits weak sites, and a review without a count is ad hoc.
Counting: the L-track lead writes the per-construct count in one gap-matrix column; a reviewer outside the track checks each site's document and front-end run.
Final position: C with A, B (invented ISAs zero), D as procedure, and E.
Confidence: high on the rule; medium on who counts.
What divides: only who checks the count (Fable: the panel; me: one outside reviewer per admission).

### Q2.4

Strongest opposing argument on `scaled_dot` (Sonnet, Fable): two real MX sites exist, so the word joins the types.
Evidence check: the NKI page lists `nki.isa.nc_matmul_mx` for MXFP8 and MXFP4 on NeuronCore-v4, so Trainium3 is a public second site beside Blackwell for the types.
Fable's claim that Trainium has no MX type holds for Trainium2 only.
Rebuttal on the word: the same page defines the scale group as "8 partitions x 4 elements per partition", not one run of 32 along the contraction.
A fixed-block `scaled_dot` would need a block-shape attribute to cover both sites, while the idiom (convert, broadcast and reshape the scale to the group shape, multiply, `dot`) states any grouping exactly.
Strongest opposing argument on L23 (Fable): real indirect DMA has no bounded covering range, so the `gather` idiom must cover all of `d0`, whose size is an issue-time parameter (RA10).
Rebuttal: a covering region over all of `d0` is expressible once `d0` has one addressed dimension (Q2.9), and its cost is the back end's loss of static addresses, the cost L23's own caveat accepts.
The idiom and the syntax then lose the same static fact, so the idiom does not fail by the Q2.3 test.
On L19 (Fable, Sonnet): first in line after v3, awaiting a failure demonstration in C2; I agree.
Final position: admit `f4e2m1fn` and `f8e8m0fnu`; `scaled_dot`, L18, L19, L20, L21, L23 stay idioms; L22 cut; no `tf32` storage type.
Changed: no.
Confidence: medium on `scaled_dot` and L23; high on the rest.
What divides: `scaled_dot` (Sonnet and Fable admit it); L23 (Fable admits it late Q2).

### Q2.5

All three agree: no new state kind in v3; data-dependent addressing goes to one study (RA9), joined with L23 by Fable; async groups stay plain integer state or go with concurrency.
Final position: Options C and D, unchanged; confidence medium-high.

### Q2.6

All three agree on Option B: concurrency out of v3, program order is the only order, and a constraint used as a wait faults when false.
Final position: Option B, unchanged; confidence high.

### Q2.7

Strongest opposing argument (Fable): the descriptor-driven class has no study, NVDLA has a public register-level manual, and Ascend's instruction level is not public, so NVDLA replaces Ascend now.
Rebuttal: the class matters to the data-dependent addressing study of Q2.5, not to any v3 admission, since no construct in Q2.4 is one site short with NVDLA as its second.
Concession: NVDLA, not Ascend, is the study to queue, as an input to the RA9 study.
All three agree to correct the three verdicts, recount, and treat the Hopper description as the GPU tensor-core description (E).
Final position: C, D (only sketches that ran count), and E in v3; NVDLA queued for the RA9 study; no Ascend study.
Changed: slightly, NVDLA now named as the queued study.
Confidence: medium.
What divides: whether every sketch must run (Fable, me) or only those behind an admitted extension (Sonnet); timing of NVDLA.

### Q2.8

All three agree on Option A: remove `cholesky`, `triangular_solve`, and `batch_norm_inference`; C gives the same set.
Final position: Option A, unchanged; confidence high.

### Q2.9

Strongest opposing argument on `d0` (Sonnet): making it ordinary buys nothing, since its size is run-time (RA10), so ordinary buffers would need a run-time size form for one user.
Concession: correct; an ordinary declaration would generalize buffers for one site.
Revised: `d0` stays implicit and keeps the one issue-time size, but its implicit declaration gives one addressed dimension, so the region-check special case goes (`taidl/taidl/accelerator.py:31` declares `[]`, `taidl/taidl/instruction.py:123` patches it to one).
Strongest opposing argument on instructions without semantics (Sonnet, Fable): R11 states it as "legal only without regions", needed for configuration instructions.
Concession: my "semantics iff an output region" leaves an input-only instruction without semantics legal and meaningless; R11 rejects it.
Final position: cut presumed reshape; `d0` implicit with one declared dimension; `constraints=` only (already D34); `ROOT` unchanged on the last line; R11 plus "a semantics defines at least `%Out0`".
Changed: yes on `d0` and on E.
Confidence: high on reshape and `constraints=`; medium on `d0`.
What divides: whether `d0` appears in the object model as declared (me, Fable) or stays a spec sentence only (Sonnet).

### Q2.10

All three agree on Option A for the grammar.
Strongest opposing argument on reserved names (Sonnet, D17): a user meets the error before reading oracle docs, so the list must be in the open document.
Rebuttal: the names are reserved by one generator, not the front end, so a table in the language spec states a generator rule as language.
Concession: the spec needs one sentence that TAIDL-TO reserves names, naming where its list lives, and the error names the word and generator.
Fable's E (generate the table) removes drift but adds a build step for a list the front end does not enforce.
Final position: Option C plus that one pointer sentence; E if a second generator reserves names.
Changed: slightly, adding the pointer sentence.
Confidence: medium.
What divides: Sonnet holds D, Fable holds E.

### Q2.11

All three agree: the execution page stays normative; the object model page fixes only the names a generator reads, checked by L29; the rest moves to the `taidl` README; no serialized format.
Final position: Option C, unchanged; confidence medium.

### Q2.12

Strongest opposing argument (Sonnet): owner 14 puts the papers first, and the TAIDL paper's term is "TAIDL definition", not "ISA description".
Evidence check (term counts from the text of both owner papers):

| Paper | "ISA description(s)" | "TAIDL definition(s)" | "ISA definition(s)" | "TAIDL description(s)" |
| ----- | -------------------- | --------------------- | ------------------- | ---------------------- |
| TAIDL, MICRO 2025 | 1 | 27 | 15 | 0 |
| ACT, OOPSLA 2026 | 53 | 0 | 2 | 12 |

- Sonnet is right about the TAIDL paper, and Fable's claim that it calls the file an ISA description is wrong.
- The two owner papers disagree, and both terms are precise, so owner 14 alone does not decide it.
- I still prefer "ISA description": the newer paper uses it, and it names the artifact without the language name.
On renames, strongest opposing argument (Sonnet, Fable, D13): the grammar rule `instruction` needs regenerated parsers and buys a writer nothing visible.
Rebuttal: L17 already changes `TYPE` in `IDLV2.g4` (`language/plan/extensions.md:25`), so v3 regenerates the parsers anyway; that is Fable's own stated condition for the rename.
Final position: "ISA description", put to the owner as a choice between the two papers; rename `parameters`, `comp_attr`, `var_type`, and the rule `instruction` in v3; delete `num_dim_str`; move `array_dim_str` into the oracle.
Changed: no on substance; the file name now goes to the owner.
Confidence: medium.
What divides: the file name (Sonnet prefers "TAIDL definition"); the rule rename (Sonnet and Fable defer it).

## Escalation to the owner

- Q2.1: under owner 17 the in-place flag is redundant with parsed region identity for "must reuse".
- "May reuse" is distinct texts under S5.2, and neither the flag nor identical text expresses it.
- The flag's only reader, the back end's copy pass, needs a fact the front end can derive from the texts.
- The flag can disagree with the regions, which costs a rejection rule and an issue-time fault.
- Request: remove `set_inplace` in v3 and derive it; if kept, require it exactly when the parsed texts are identical.
- Q2.12: the two owner papers use different terms for the user's file ("TAIDL definition" in MICRO 2025, "ISA description" in OOPSLA 2026); the owner picks one.
