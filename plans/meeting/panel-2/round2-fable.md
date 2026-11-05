# Panel 2, round 2, Fable: accelerator architect

Lens unchanged: a rule the hardware cannot observe is a cost without a hardware counterpart.
Evidence re-read this round: `language/spec-draft/instructions.md` S5.3 to S5.5, `execution.md` S5.6, `conformance.md` rows on marks, `taidl/taidl/instruction.py:163-244`, `act-backend/act_backend/instruction_info.py:112`, `semantics/rulings.md:6-13`, the in-place sketches in `casestudies/amx.md:48-55`, `gemmini.md:54-58`, `tpu.md:44-49`, `hopper.md:44-49`, and the NKI `nki.isa.tensor_tensor` page.

### Q2.1

Strongest opposing argument: the owner's input; identical region text cannot say whether the hardware may or must reuse the source storage, `r0 = r1 + r2` must stay legal when `r2` equals `r0` at issue, and a read-after-write hazard is a hardware fact the description must be able to state.
All three round-1 files chose Option C with "any other overlap is a fault"; the owner's input shows that fault rule is wrong, so I change position.

Rebuttal and concession, in steps.

1. Three cases exist in hardware, and I name them for the rest of the question: must-alias, may-alias, and hazard.
2. Must-alias is the accumulator: AMX `tdpbf16ps` reads and writes `tmm` tile 2 (`amx.md:52-54`), Gemmini `compute_accumulated` reads and writes `accm[@a.c]` (`gemmini.md:56-57`), the TPU MXU result buffer reads and writes `mrb[@a.acc]` (`tpu.md:46-48`), Hopper `wgmma` reads and writes `rf[0, @a.d]` (`hopper.md:46-48`), NKI `nc_matmul` accumulates into one PSUM tile (`trainium2.md:109`); in every one the input and output regions are spelled identically, because the hardware has no other destination.
3. Identical text expresses must-alias exactly: the two regions evaluate equal at every issue, so the output always lands where the input was, and the input value is dead after the issue; that is the whole content of the flag for the backend (`instruction_info.py:112` passes `instruction.inplace`, nothing more).
4. May-alias is `r0 = r1 + r2` with `r2` free: NKI `tensor_tensor(dst, data1, data2)` documents no rule on `dst` equal to or overlapping an input (nki.isa.tensor_tensor page; it constrains only PSUM placement), so the hardware tolerates equality and the description must not fault on it; `set_inplace` cannot express may-alias either, since a true flag forces equality at issue (S5.5, "else the issue is a fault").
5. So may-alias is the absence of a mark plus a defined result, and S5.6 steps 5 and 6 already define it: every input is read, then every output is written, so an equal or partially overlapping unmarked pair computes as if copied.
6. The hazard is a per-engine fact: a DMA engine typically forbids any source-destination overlap, a streaming vector engine typically allows whole overlap or none; neither is expressed by `set_inplace`, both are a constraint over starts and sizes, which is what the Q2.2 marks append today (`instruction.py:216-218`).
7. The flag's remaining role is Option B's run-time equality check on regions spelled differently; the front end already forces the same buffer and equal sizes by text (S5.5; `conformance.md` row "in-place input whose buffer or sizes differ"), so the only free variable is the starts, and `@a.x == @a.y` is a constraint the writer can spell with no new construct.
8. A redundant assertion creates its own error class: the Hopper sketch spells identical regions and omits the mark (`hopper.md:46-49`), and under `semantics/rulings.md:11` every `wgmma` issue would be rejected as an unmarked overlap, although the description is right about the hardware.
9. The one case where the flag is not derivable is two data models over one storage (CP16 aliased buffers), where the texts cannot be identical because the buffer names differ; v3 has no aliased buffers, so the case does not arise in v3.
10. On "may or may not reuse the source storage" in `r0 = r1 + r0`: physical renaming is invisible to a program and to a compiler, and TAIDL describes architectural storage (S5.6 region values are buffer elements), so the language should state only the architectural fact, which identical addresses already state.

Final position: derive must-alias from identical region text (parsed trees, as `keep_off` compares them); an unmarked pair that overlaps at issue is defined by S5.6 read-then-write; a hazard is a per-instruction constraint; `set_inplace` is redundant in v3 and goes; a changed position.
Confidence: medium-high on derivation and on dropping the overlap fault; medium on "truly redundant", because of the CP16 residual.
Still divides the panel: whether partial overlap of an unmarked pair is defined by S5.6 (my position) or a fault (round-1 C of all three); whether the CP16 residual justifies keeping the flag as a dormant construct.

### Q2.2

Strongest opposing argument: Opus, Option A with no helper, because the Q2.1 overlap fault already supplies `set_apart` as the default and whole-or-apart as the only other legal case.
Rebuttal: that argument rests on the fault rule the owner's input removed; once an unmarked overlap is defined by S5.6, the two conditions are real hazard statements a writer must be able to add, and the per-dimension disjunction (`instruction.py:200-214`) is the part a writer gets wrong, as Sonnet says.
The marks compare starts by spelling, which no datasheet does; the idiom compares values, which is what the hardware does.
Final position: Option C, remove both marks and ship one documented constraint idiom for apart and for whole-or-apart; unchanged.
Confidence: high.
Still divides the panel: helper or no helper; the condition itself is agreed.

### Q2.3

All three agree: idiom-failure argument plus two real, publicly documented sites in owner-15 scope, counted per construct in a matrix column, sketches counted only after a front-end run, invented ISAs count zero, applied to every extension including L17.
Final position: as above; unchanged.
Confidence: high.
Still divides the panel: nothing of substance; Opus adds an outside-track reviewer per cited site, which I accept.

### Q2.4

Strongest opposing argument: Opus, `scaled_dot` has a lossless idiom (`convert`, `multiply` by the broadcast power-of-two scale, `dot`), so only the two MX types enter v3; Sonnet and Opus both keep L23 an idiom because `gather` and `dynamic_slice` take indices from a region.
Rebuttal and concession.

1. `scaled_dot`: a power-of-two scale multiply is exact in f32 for `f4e2m1fn` times `f8e8m0fnu` (6 times 2^127 is below the f32 maximum, 0.5 times 2^-127 is a representable subnormal), so the idiom is lossless on the MX FP4 path; the hardware applies the scale per 32-element block after the block dot, which differs only in accumulation order, inside the R8 tolerance; I concede `scaled_dot` to the idiom, with the two types kept.
2. L23: my round-1 failure claim was that the covering input is all of `d0` with no static size; a writer can pass the HBM size as an attribute and cover it, so the idiom grows rather than fails, and under the Q2.3 rule it waits for the data-dependent addressing study (Q2.5 D) to show a failure; I move L23 from "admit late Q2" to "late Q2 slot kept, admission gated on that study".
3. L19 stays first in line after v3: strides are universal (TMA `globalStrides`, NKI `dma_copy`, AIE-ML descriptors, Gemmini `config_ld`), and the span idiom over-declares the footprint, which is exactly the hazard constraint of Q2.2 reading a wrong extent; that is the failure demonstration C2 should run.
4. L18, L20, L21 idioms and L22 cut are agreed by all three.

Final position: L17 types `f4e2m1fn` and `f8e8m0fnu` in v3, `scaled_dot` as idiom, no `tf32` storage type; L19 first after v3 pending the C2 footprint demonstration; L23 slot kept but gated on the Q2.5 study; L18, L20, L21 idioms; L22 cut; changed on `scaled_dot` and L23.
Confidence: medium.
Still divides the panel: whether L19's footprint over-declaration already counts as idiom failure.

### Q2.5

All three agree: no new state kind in v3, descriptors through existing state and regions, async groups wait for the concurrency study, data-dependent addressing is one study (RA9) joined with L23.
Final position: Options C and D; unchanged.
Confidence: medium-high.
Still divides the panel: nothing; Opus's "would change my mind" (a descriptor written at run time and read later) is the Trainium DGE case (`dge_mode`), which the study must examine first.

### Q2.6

All three agree: Option B, out of v3, program order is the only order, a constraint-as-wait faults and the spec says so.
Final position: Option B; unchanged.
Confidence: high.
Still divides the panel: nothing.

### Q2.7

Strongest opposing argument: Sonnet, no new study before v3, because the recount and the fixed Hopper sketch decide v3 contents, and Opus, NVDLA only if an admission turns on the descriptor-driven class.
Rebuttal: the evidence base feeds the Q2.3 rule for the whole of Q2, and without one descriptor-driven engine every count is drawn from register-file ISAs, so a construct that class needs can never reach two sites; one NVDLA sketch is one file against a public register-level manual.
Concession: Ascend is out, and the GPU tensor-core evidence is the fixed Hopper sketch (CP13, CP15) run through the front end, not a new file; R47's description then reuses it.
Final position: Options C and D, plus one NVDLA sketch after the recount, no Ascend; unchanged in substance, narrowed on E.
Confidence: medium.
Still divides the panel: whether NVDLA is written before or only when a count needs it.

### Q2.8

All three agree: Option A, remove all three; Option C coincides since `BOOL`, `TRANSPOSES`, and `NUMBER` are used only by these words (`taidl/taidl/operations.py:18-23`, `:145-156`, verified).
Final position: Option A; unchanged.
Confidence: high.
Still divides the panel: nothing.

### Q2.9

Strongest opposing argument: Sonnet, keep `d0` special in code but state it in the spec, since its size is run-time either way and an ordinary declaration buys the user nothing.
Rebuttal: a writer models `hbm` as a byte array like every other buffer; a special case at `instruction.py:123` is a second rule for the same thing, and the issue-time size (RA10) is the only difference, which an ordinary declaration can carry as well; the user-visible text is the same under both, so this is a code-shape question the panel can leave to the front-end plan.
Agreed by all three: remove presumed reshape, keep `constraints=` only, leave `ROOT` where it is, semantics present exactly when an output region exists (R11).
Final position: unchanged; `d0` as an ordinary buffer with one addressed dimension and an issue-time size, with the divide noted as non-user-visible.
Confidence: medium-high.
Still divides the panel: `d0` declaration form only.

### Q2.10

Strongest opposing argument: Opus, reserved names belong to one generator, so the spec cuts the table and the error message names the word and the generator; Sonnet, keep the table labeled since the user meets the error before any oracle doc.
Concession: I drop the generated table; a build step for one short list is not worth it (Sonnet), and front-end rejection would make the names language, which they are not (Opus).
The user's first encounter is the error, so the error text must name the word, the generator, and where its list lives; that satisfies D17 without a table.
Final position: Option A for the grammar; Option C for reserved names with the error text carrying the word and the generator; changed on reserved names.
Confidence: medium.
Still divides the panel: table in S4 (Sonnet) versus none (Opus and me).

### Q2.11

All three agree: the execution page (S5.6) stays normative, the object model page fixes only the names a generator reads (Option C list), the rest moves to the `taidl` README, no serialized format.
Final position: as above; unchanged.
Confidence: medium.
Still divides the panel: nothing.

### Q2.12

Strongest opposing argument: Sonnet, the TAIDL paper's own term is "TAIDL definition" and owner 14 puts the papers first; Opus counters that the ACT paper uses "ISA description" 53 times.
Rebuttal: I could not verify either count this round; under owner 14 the precise term wins, and "ISA description" names what the file is to an architect (an instruction set and its storage), matches the draft spec (`spec-draft/overview.md:8`), and is the newer paper's usage if Opus's count holds; the moderator should verify the two counts before ruling.
On renames all three agree on `parameters`, `comp_attr`, `var_type` now and the grammar rule `instruction` later (D13); Opus's point that L17 regenerates parsers anyway (`extensions.md:25`, `TYPE` in `IDLV2.g4`) removes the cost argument for deferring the rule rename, so I accept it in the batch if the visitor updates fit.
Final position: "ISA description" (Option E as R1), Option B for renames with the grammar rule rename admitted if L17's parser regeneration lands in the same batch; slightly changed.
Confidence: medium.
Still divides the panel: the file name's paper evidence; the size of the rename set.

## Escalation to the owner

- Decision 17 fixes the meaning of the mark, and the owner's Q2.1 input keeps `set_inplace` unless shown redundant; the evidence above shows that in v3 every in-place site spells identical regions (`amx.md:52-54`, `gemmini.md:56-57`, `tpu.md:46-48`, `hopper.md:46-48`, `mini_trn.py:92-94`), that the front end already pins buffer and sizes by text so the flag's only free content is a start equality a constraint can spell, and that the one non-derivable case (aliased buffers, CP16) is not in v3.
- Keeping the flag as a required assertion costs one error class with no hardware counterpart (the unmarked Hopper `wgmma` sketch would be rejected at every issue under `semantics/rulings.md:11`); if the owner keeps it, I ask that it be optional and checked as a lint against identical text, never a condition for aliasing.
- The round-1 Option C fault on any other overlap contradicts the owner's `r0 = r1 + r2` input; I ask the owner to confirm that an unmarked overlap at issue is defined by S5.6 read-then-write, with a per-instruction constraint as the way to state a hazard.
