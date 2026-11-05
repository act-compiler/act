# Panel 9, round 1, Fable: overlap and hazard declarations

Lens: how shipped ISAs behave when an instruction's operands alias.
Every position respects owner decisions 16, 17, 22, 24, 30, 31, and 34.
Under decision 30 the markers stay as shorthand; the questions are what each expands to and how the one constraint mechanism handles overlap.

## Shipped-ISA facts this round rests on

| ISA | Input-output overlap rule | Source |
| --- | ------------------------- | ------ |
| RISC-V V | Destination may overlap a source only as the whole group at equal EEW, or at the fixed low or high part for narrowing or widening; "any instruction encoding that violates the overlap constraints is reserved" | RVV spec 1.0, section Vector Operands (fetched) |
| RISC-V V | `vslideup`, `vslide1up`, `vrgather`, `vcompress`: destination "cannot overlap the source vector register group, otherwise the instruction encoding is reserved"; reductions: "the destination vector register can overlap the source operands" | RVV spec 1.0, slide, gather, compress, reduction sections (fetched) |
| Intel AMX | `tdpbf16ps` reads and writes one tile: `tsrcdest.row[m].fp32[n] := tsrcdest.row[m].fp32[n] + tmpf32`; the operand is both input and output by encoding | Intel SDM `TDPBF16PS` operation (fetched) |
| AWS NKI | `nc_matmul` "must read inputs from SBUF and write outputs to PSUM"; `accumulate=True` adds to the destination; accumulate on an uninitialized PSUM location is undefined behavior | NKI `nki.isa.nc_matmul` page (fetched) |
| PTX wgmma | The accumulator `d` is read and written by one instruction; reading A fragments before a `wgmma.wait_group` covering the instruction is undefined behavior | LLVM NVPTX usage doc quoting PTX ISA (search result; the PTX page itself was too long to fetch) |

- In every shipped case the legal overlap is one of: the whole operand (accumulator, RVV same-EEW), a fixed part by encoding (RVV widening), or none (slide, gather, compress, NKI SBUF versus PSUM).
- In every legal case the result is computed from the operand's old value, so the value rule is "inputs are the pre-state", which panel 1 Q1.1 already fixed.
- The only hazards a shipped ISA leaves undefined are temporal (wgmma before `wait_group`, PSUM uninitialized), which Q2.6 puts out of v3 and Q1.4 makes a fault.

### Q9.1

Position: D, a closed set of four overlap relations between one input and one output region, with no separate hazard-order declaration, because the value under every legal relation is fixed by pre-state reads.

1. The three current markers already name three relations: must reuse (`set_inplace`), apart (`set_apart`), and exact-or-apart (`set_no_partial_overlap`).
   Evidence: taidl/taidl/instruction.py:163-188; spec-draft instructions.md:49-62.
2. These three are exactly what RVV, AMX, and NKI need: the accumulator tie is must reuse, the RVV same-EEW rule is exact-or-apart, and slide, gather, compress, and SBUF-to-PSUM are apart.
   Evidence: table above.
3. Owner 31 says the language must not fail a scenario a real architecture allows, so a fourth relation "any overlap, value from the pre-state" is needed as an opt-in for an instruction that permits partial overlap, such as a DMA engine moving within one buffer.
   Evidence: owner-decisions.md:29; trainium2.md:19 (`dma_copy` on `d0`, a single buffer where regions can partially overlap).
4. A hazard-order declaration (option B) adds nothing: under Q1.1 C every expression reads the pre-state and outputs commit at the end, so "inputs read before outputs written" is the only value any relation can have, and no shipped ISA defines a write-before-read value.
   Evidence: panel-1/decision.md:20-21,55; the RVV reduction rule and the AMX `tsrcdest` pseudocode compute from old values.
5. "May reuse" is not a declaration: owner 22 makes it the implementation's choice on an unmarked exact overlap, and the model cannot observe it, so it is the meaning of the default, not a relation an author picks.
   Evidence: owner-decisions.md:21; panel-2/decision.md:33.
6. Partial overlap with a defined value cannot be the default: RVV reserves it, AMX cannot express it (whole tiles), and a streaming systolic write over a region it still reads gives garbage, so it stays a fault unless the author opts into relation 4.
   Evidence: RVV overlap rule (fetched); amx.md:48-55; panel-1/decision.md:58-61.

Confidence: medium-high on the four relations; medium on whether relation 4 has a v3 site, since no case study sketches a within-buffer DMA.
Changes my mind: a shipped tensor ISA whose documentation defines a partial-overlap result that is not the pre-state value; then B's ordering declaration is needed.

### Q9.2

Position: D with one rule: every marker expands to constraints over evaluated start and size terms, through the mechanism that already gives each region its bounds constraints; the vocabulary needed is `==`, `<=`, `+`, `and`, `or` over those terms, which the grammar has.

1. One mechanism exists today in three flavors: the bounds check on every region (S1.3.3), the constraints the author writes (S1.3.5), and the constraints `keep_off` appends for the two placement markers, all faulting at issue.
   Evidence: instructions.md:27,38,59; instruction.py:190-231; overview.md:40.
2. `set_apart` on input k expands to one constraint, a disjunction over the access dimensions whose start terms differ: `out_start + out_size <= in_start or in_start + in_size <= out_start`, which `keep_off` already emits.
   Evidence: instruction.py:216-219,223-224.
3. `set_no_partial_overlap` on input k expands to one constraint per such dimension: `out_start == in_start or <the disjunction of step 2>`.
   Evidence: instruction.py:220-222; instructions.md:61-62.
4. `set_inplace` on input k expands to two static checks (same buffer and equal size terms, a rejection at declaration) and one issue constraint per access dimension, `out_start == in_start`, which is S1.3.6 restated as a constraint.
   Evidence: instructions.md:50-52; panel-1/decision.md:52.
5. Under owner 34 every term is the evaluated start or size expression, never an attribute name, so a tied pair spelled `@a.dst` and `1024 - @a.src` expands correctly, and no tool reads meaning into a name.
   Evidence: owner-decisions.md (decision 34); instruction.py:216 (`term(out_start) == term(start)` over expressions).
6. The in-place marker keeps one reader beyond its constraints: the backend copy pass and the allocator consume the pairing itself to copy a shared operand and to force equal offsets, which a constraint over terms does not tell them.
   Evidence: inplace_fix.rs:59-80; constraint.cc:206-236; panel-4/decision.md:84-86.
7. Relation 4 (any overlap) is a marker that expands to nothing and suppresses the implicit exact-or-apart constraint of Q9.3 for that pair; its vocabulary need is zero.
   Evidence: Q9.3 step 4.
8. A marker's expansion is a documented text so a user can read, replace, or extend it with strides later (L19), and the per-dimension disjunction stays generated because it is error-prone by hand.
   Evidence: panel-2/decision.md:59-62; instruction.py:209-224.

Confidence: medium-high.
Changes my mind: a strided region form in v3 whose apart condition is not a per-dimension interval disjunction; then the expansion needs a new term kind before the markers can expand.

### Q9.3

Position: a violated constraint of any origin is one outcome, a fault at issue (S1.3.5); an undeclared same-buffer input-output pair carries an implicit exact-or-apart constraint, so an exact overlap is legal with the pre-state value and a partial overlap faults, and the any-overlap marker lifts it.

1. Owner 16 forbids undefined behavior, and the fault on a false constraint is already defined; the clarification of owner 30 makes bounds, marker, and user constraints one mechanism, so a violated marker is that same fault and needs no new norm.
   Evidence: owner-decisions.md:16; instructions.md:38,52; panel-1/decision.md:165-166.
2. Rejecting an ISA specification whenever overlap is possible (option A) fails real scenarios: RVV lets a reduction destination overlap its sources, and `r0 = r1 + r2` with `r2 == r0` is owner 22's legal case, so option A contradicts owner 31 and owner 22.
   Evidence: RVV reduction rule (fetched); owner-decisions.md:21.
3. A run-time fault on every undeclared overlap (option B) fails the same cases, since an unmarked exact overlap is legal by owner 22 and panels 1 and 5 both rule it legal with the pre-state value.
   Evidence: panel-1/decision.md:55-57; panel-5/decision.md:89-90.
4. The stated default (option C) is an implicit constraint, generated like the bounds constraint for every input-output pair in one buffer: per dimension, equal starts or apart; this is the rule RVV, AMX, and the two panels converge on.
   Evidence: instruction.py:220-222 (the same text); act-oracle instruction_codegen.py:162-181 (pre-state value at no cost).
5. Partial overlap defaults to a fault, not a pre-state value, because no shipped tensor ISA promises that value and the oracle would certify a kernel the hardware corrupts, the same error Q1.4 avoided for stale reads.
   Evidence: panel-1/decision.md:58-61,69-71.
6. Owner 31 is still met: an author whose engine defines partial overlap puts the any-overlap marker on that input, the implicit constraint is not generated, and the value is the pre-state value the oracle already gives.
   Evidence: Q9.1 step 3; instruction_codegen.py:162-181.
7. The in-place marker's static checks (same buffer, equal size terms) are rejections at declaration, as owner 24 already makes the cross-buffer pair; only evaluated-start conditions are issue faults, which matches how bounds work under owner 34.
   Evidence: owner-decisions.md:23; instructions.md:51.

Confidence: medium-high.
Changes my mind: a case study whose instruction legally overlaps in part by default in hardware documentation; then the implicit constraint for that ISA class is dropped, still with the pre-state value.

### Q9.4

Position: the oracle evaluates every constraint, implicit, marker-generated, or user-written, on the host before any load and computes the pre-state value; the backend reads the same constraint list into its solver, plus the in-place pairing for its copy pass and offset equality; the corpus holds one legal, one fault, and one rejected case per relation.

1. Oracle value: it loads every input and then stores every output, so the result under all four relations is the pre-state value with no new code; Q1.1 C only moves updates after outputs.
   Evidence: instruction_codegen.py:162-181; panel-1/decision.md:28.
2. Oracle checks: `generate_constraints` already emits the constraint list first, so the implicit exact-or-apart constraints and the in-place start equalities join that list and fault through the same path as bounds and user constraints; output pairs stay disjoint or fault (S1.3.4).
   Evidence: instruction_codegen.py:32,162,170-171; instructions.md:29.
3. Backend must reuse: `fix_inplace` gives a shared overwritten operand its own copy and errors on an HBM-pinned or shared view; the allocator then forces equal offsets (B30) instead of merely dropping the interference edge.
   Evidence: inplace_fix.rs:59-80; constraint.cc:217-232; panel-4/decision.md:84-89.
4. Backend other relations: the apart disjunction is the solver's own non-overlap constraint over offsets, so an apart or exact-or-apart constraint, implicit or marked, is one more disjunction in `overlap_constraints`; the backend never emits a faulting issue (Q1.12 E), so it must honor the implicit constraint too.
   Evidence: constraint.cc:61-103 (the disjunction at line 88); panel-1/decision.md:167,170.
5. Backend any overlap: no constraint; the solver may place the regions anywhere, and the copy pass is not needed because the value is pre-state by construction.
   Evidence: Q9.1 step 4.
6. Corpus per relation: legal case (value equals the unmarked pre-state value, Q6.2 step 6), fault case (partial overlap at issue; unequal marked starts), rejected case (cross-buffer mark, owner 24; unequal size terms), and the shared-overwritten-input kernel that gates B30.
   Evidence: panel-6/decision.md:33-44; test_extraction.py:405-414,507-520 (as cited by panels 4 and 6); owner-decisions.md:23.
7. One extra corpus row per marker checks the expansion itself: the marker and its hand-written expansion give the same fault set on a swept attribute range, which is what owner 30 promises the user.
   Evidence: instruction.py:227-231 (a marker replaces exactly its appended constraints, so the expansion text is recoverable).
8. The `pending` rows of the norm catalogue (unmarked and partial overlap) close with these cases, which the v3 gate requires.
   Evidence: panel-6/decision.md:64-65.

Confidence: medium-high on oracle and corpus; medium on the backend, since B30 is not landed and the implicit disjunction in the solver is a design, not code.
Changes my mind: a backend allocation that cannot express exact-or-apart in the OR-Tools model without a blow-up on large kernels; then the backend's default becomes apart only, stricter than the oracle, which is still a producer that never faults.

## Escalation to the owner

- None against a decision.
- Owner 31 asks what the author declares; this round's answer is four relations as markers, each expanding to constraints over evaluated terms, with the pre-state value fixed for all of them by Q1.1 C, so no hazard-order declaration is offered.
