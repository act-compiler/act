# Panel 2 decision: language surface

Moderator: Opus.
Binding: owner decisions 10, 14, 15, 16, 17, 19, 20, 21, 22, 24.

| Question | Decision (one line) | Confidence | Owner? |
| -------- | ------------------- | ---------- | ------ |
| Q2.1 | Remove `set_inplace`; an input is in place exactly when its parsed region equals an output's; redundancy shown | medium-high | confirm (owner 22) |
| Q2.2 | Remove `set_apart` and `set_no_partial_overlap`; document two constraint idioms, no API helper | high | no |
| Q2.3 | Idiom failure plus two public in-scope sites whose sketches ran, per construct, invented ISAs zero, for every extension | high | no |
| Q2.4 | v3 adds `f4e2m1fn` and `f8e8m0fnu` only; `scaled_dot`, L18, L19, L20, L21, L23 stay idioms; L22 cut; no `tf32` | medium-high | no |
| Q2.5 | No new state kind in v3; data-dependent addressing is one study (RA9) joined with L23 | medium-high | no |
| Q2.6 | Concurrency out of v3; program order is the only order; a constraint used as a wait faults when false | high | no |
| Q2.7 | Correct three verdicts and recount; only sketches that ran count; fixed Hopper sketch is the GPU description; NVDLA queued, no Ascend | medium | no |
| Q2.8 | Remove `cholesky`, `triangular_solve`, `batch_norm_inference` | high | no |
| Q2.9 | Cut presumed reshape; `d0` gets one declared dimension; `constraints=` only; `ROOT` unchanged; R11 | medium-high | no |
| Q2.10 | Grammar page links the `.g4` files; no reserved-name table, one pointer sentence and a naming error | medium | no |
| Q2.11 | Execution page normative; object model page fixes only the names a generator reads; rest to the `taidl` README | medium | no |
| Q2.12 | File name goes to the owner (default "ISA description"); rename `parameters`, `comp_attr`, `var_type`, rule `instruction` | medium | yes (file name) |

## Q2.1: in-place mark

Decision: remove `set_inplace` in v3; an input region and an output region whose parsed texts (buffer, start trees, size trees) are identical are in place, and the front end derives the flag the generators read.
Redundancy with owner 17 is shown for v3.

Reasoning:
1. Owner 17 fixes the meaning as "the output reuses the marked input's storage"; owner 24 requires one buffer.
2. The draft already pins buffer and sizes by text and leaves only the starts free, checked at issue (`language/spec-draft/instructions.md:49-52`).
3. Must reuse: identical parsed text evaluates to the same storage at every issue, so the output lands on the input's storage, which is owner 17's whole content.
4. Every in-place site cited spells identical text: `mini_trn.py:92-94` with the flag, `casestudies/hopper.md:46-48` without it.
5. May reuse: a true flag forces equal starts at issue or a fault (`instructions.md:52`), so the flag cannot say "may" either.
6. So the must-versus-may split is carried by identical versus distinct text, and owner 22's `r0 = r1 + r2` is the unmarked distinct pair.
7. Owner 22's "`r0 = r1 + r0` may or may not reuse the source storage" is a choice the model cannot observe, since every input is read before any output is written (`instructions.md:54`).
8. Differently spelled starts: a tied pair `@a.src`, `@a.dst` is written with `@a.src` in both regions plus the constraint `@a.dst == @a.src`; nothing is lost, and the identical text carries the clobber fact.
9. No real ISA with a tied pair that cannot be written this way is cited; Sonnet's case is uncited and so carries no weight against step 8.
10. The flag's only reader is the back-end copy pass, which reads one static flag per input before addresses exist (`act_backend/instruction_info.py:112`, `inplace_fix.rs:59-63`).
11. The front end already compares parsed start trees this way (`taidl/instruction.py:214`), so the derived flag is available at the same point.
12. The flag adds a failure mode: today only its count is checked (`taidl/instruction.py:163-172`), so the flag and the texts can disagree, and the unmarked Hopper sketch gets no copy.
13. The one case derivation cannot cover is two buffers over one storage (CP16), which owner 24 rejects.
14. `r0 = r0 + r0` derives two overwritten inputs; the copy pass iterates per child (`inplace_fix.rs:59-63`), so this works.

Composition with panel 1's overlap norm:
- An identical parsed pair is in place and is exempt from any overlap rule.
- Every other input and output pair, including texts that differ but evaluate equal, falls under panel 1's norm, whichever option it picks.
- A per-instruction hazard is a user constraint (Q2.2 idiom) under any norm.

Fallback if the owner keeps the flag: the flag is required exactly when the parsed texts are identical, checked both ways at declaration as a rejection, with no issue-time fault.

Evidence verified: `instructions.md:49-54`, `taidl/instruction.py:163-172` and `:214`, `instruction_info.py:112`, `inplace_fix.rs:59-63`, `mini_trn.py:92-94`, `hopper.md:46-48`, `semantics/rulings.md:8-11`.

Dissent: Sonnet keeps the flag as a "must reuse" declaration for tied pairs with differently spelled starts, checked both ways (medium); its decisive case has no cited instance.
Fable rates "truly redundant" medium because of CP16, which owner 24 closes.
The round-1 blanket fault on any other overlap (Option C) has no defender left; all three withdrew it against owner 22.
Confidence: medium-high.

## Q2.2: placement marks

Decision: remove both marks; the spec and cookbook give one constraint text for "apart" and one for "identical or apart"; no API helper.
1. Each mark only appends a constraint a user can write (`taidl/instruction.py:216-218`).
2. With no default overlap fault (Q2.1), the two conditions are real hazard statements, so the idiom is the only spelling, not a third one.
3. The per-dimension disjunction (`taidl/instruction.py:200-214`) is error-prone, so it is documented, not re-derived.
4. The idiom must handle strides once L19 lands (`language/plan/extensions.md`, L19).
Dissent: none on removal; Opus first held no idiom, then conceded.
Confidence: high.

## Q2.3: extension admission rule

Decision: syntax enters only with an idiom-failure argument plus two real, publicly documented sites in owner-15 scope.
1. Counting is per construct, in one gap-matrix column, and invented ISAs count zero (CP9).
2. A site counts only after its sketch ran through the front end; `casestudies/method.md:5` records one-session authorship, and CP15 shows rank bugs.
3. "Idiom fails" means it loses information or cannot be stated, not that it is verbose.
4. The rule applies to every extension, L17 included; a per-extension review is the procedure and the rule is its bar.
5. The L-track lead writes the count; a reviewer outside the track checks each cited document and run (Fable accepts this).
Dissent: none of substance.
Confidence: high on the rule, medium on who counts.

## Q2.4: L17 to L23

Decision: v3 adds `f4e2m1fn` and `f8e8m0fnu`; no `tf32` storage type; `scaled_dot` stays an idiom.
1. Two public MX sites exist: Blackwell and NeuronCore-v4, Trainium3 (NKI `nki.isa.nc_matmul_mx` page, verified).
2. That page's scale group is "8 partitions x 4 elements per partition", unlike a 32-run block, so one fixed `scaled_dot` would need a block-shape attribute.
3. The idiom (convert, broadcast scale, multiply, `dot`) is exact for power-of-two scales (Fable, round 2 step 1), so it does not fail by Q2.3.
4. L18, L20, L21 stay idioms; L22 is cut; all three agree.
5. L19 is first in line after v3, admitted only if the C2 footprint demonstration shows the span idiom fails.
6. L23 waits for the Q2.5 study; covering all of `d0` is costly but expressible (RA10).
Dissent: Fable treats L19's over-declared footprint as near-failure already.
Confidence: medium-high; medium on L19.

## Q2.5: typed state kinds

Decision: no new state kind in v3; descriptors use existing state, attributes, and region reads; async groups stay plain integer state or go with concurrency.
1. All three agree; the real gap is data-dependent addressing (gap matrix rank 8), studied once (RA9) together with L23.
2. The study examines a descriptor written at run time and read later (Trainium DGE) first.
Confidence: medium-high.

## Q2.6: concurrency

Decision: Option B; out of v3; the spec states program order is the only order and a constraint used as a wait faults when false (CP14).
Consistent with owner 15 (no SIMT) and owner 16.
Confidence: high.

## Q2.7: evidence base

Decision: correct the `ldtilecfg`, indirect DMA, and `smstop` verdicts and recount; a sketch counts only after it ran.
1. The fixed Hopper sketch (CP13, CP15) is the GPU tensor-core description and feeds R47.
2. No Ascend study: hiascend.com was unreachable (`casestudies/method.md:7`) and audit R15 finds its reason argues against it.
3. NVDLA is the queued study, an input to the Q2.5 study, since no v3 admission is one site short with NVDLA as second.
Dissent: Fable wants NVDLA before admissions; that matters only to post-v3 constructs.
Confidence: medium.

## Q2.8: rare operation words

Decision: remove all three.
1. Only they use the `BOOL`, `TRANSPOSES`, and `NUMBER` forms (`taidl/operations.py:18-23`, `:145-156`, verified), so Option C equals A.
2. No case-study accelerator has them; they return at a second real ISA description that needs them.
Confidence: high.

## Q2.9: redundant spellings and implicit rules

Decision:
- Presumed reshape is removed; an explicit `reshape` line is required.
- `d0` stays predeclared with an issue-time size, declared with one addressed dimension, so the special case at `taidl/instruction.py:123` goes (`taidl/accelerator.py:31` declares `[]` today).
- `constraints=` is the only spelling.
- `ROOT` stays on the last line.
- R11: an instruction may omit semantics only when it has no regions; a semantics defines at least `%Out0`.
The `d0` declaration form is front-end code shape, not user-visible, and goes to the front-end plan.
Dissent: none left; Opus conceded R11, Sonnet conceded `d0`.
Confidence: medium-high.

## Q2.10: grammar and reserved names

Decision: the grammar page links the `.g4` files and keeps token tables and binding notes; S4 carries no reserved-name table.
1. The names are reserved by TAIDL-TO, not by the front end, so a table states a generator rule as language.
2. S4 gets one sentence saying TAIDL-TO reserves names and where its list lives, which reaches the user before the first run.
3. The error names the word, the generator, and that list.
Dissent: Sonnet keeps a labeled table (D17); the pointer sentence answers its "before the first run" argument.
Confidence: medium.

## Q2.11: object model and execution pages

Decision: Option C; S5.6 stays normative; the object model page fixes only `constraint_asts`, `update_asts`, `instr_input_asts`, `instr_output_asts`, `instr_expression_asts`, checked by L29; the rest goes to the `taidl` README; no serialized format.
All three agree.
Confidence: medium.

## Q2.12: file name and renames

File name: owner.
Counts from the text of both PDFs, case-insensitive, singular and plural, line breaks joined:

| Paper | "ISA description" | "TAIDL description" | "TAIDL definition" | "ISA definition" |
| ----- | ----------------- | ------------------- | ------------------ | ---------------- |
| ACT, OOPSLA 2026 | 54 (17 in title, running heads, and citation; 37 in body) | 12 | 0 | 2 |
| TAIDL, MICRO 2025 | 1 (about Sail, not the file) | 0 | 29 (28 case-sensitive) | 15 (mostly the language name) |

- The debaters' figures (53, 27 to 28) hold within the counting method.
- The two owner papers disagree and both terms are precise, so owner 14 does not decide it.
- Best argument for "ISA description": the newer paper uses it throughout, including in its title, and the draft spec already uses it (`spec-draft/overview.md:8`).
- Best argument for "TAIDL definition": it is the TAIDL paper's term for the file, and it matches the language's name, "ISA Definition Language".
- Panel default: "ISA description", with "spec" kept for the language document; all three debaters end there.

Renames, decided:
1. Rename `parameters`, `comp_attr`, and `var_type` in the v3 batch; all three agree.
2. Delete `num_dim_str`; move `array_dim_str` into the oracle.
3. Rename the grammar rule `instruction`: it sits in `IDLV2.g4:8`, the file whose `TYPE` rule (`IDLV2.g4:37`) L17 changes in v3, so the parser is regenerated anyway (`extensions.md:25`).
4. The glossary drops `add_tensor_buffer` and `add_control_register` (owner 19).
Dissent: Sonnet keeps the rule `instruction` (D13); its cost argument falls with step 3.
Confidence: medium.

## Escalation to the owner

- Q2.1 (Opus): `set_inplace` is redundant with parsed region identity under owner 17; remove it in v3, or require it exactly when the texts are identical.
- Q2.1 (Fable): every cited in-place site spells identical regions, and the only non-derivable case (CP16) is outside v3; if kept, make the flag an optional lint against identical text.
- Q2.1 (Fable): confirm that an unmarked overlap at issue is defined by read-then-write, with a per-instruction constraint stating a hazard; this is panel 1's overlap norm.
- Q2.1 (Sonnet): state whether a description may declare a tied pair whose input and output starts are spelled differently; if yes, Sonnet keeps the flag.
- Q2.12 (Opus): the two owner papers name the user's file differently; the owner picks one.
- Moderator: Q2.1 asks the owner to confirm that redundancy is shown under owner 22; Q2.12 asks the owner to pick the file name.
