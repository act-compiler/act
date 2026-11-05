# Panel 2, round 1: Sonnet (new user writing an ISA description)

Lens: I write a description for my own accelerator; I want few constructs, no silent surprises, and enough power to say what my hardware does.

Assumptions:

- A new user learns from `examples/mini_trn.py` and the spec, and never reads generator code.
- A construct costs the user one concept; a rule that needs a footnote costs more than a rule that needs a line.
- Generators (backend allocator, oracle) can only act on what the front end states once.

External evidence used:

- TAIDL paper (MICRO 2025), https://act-compiler.github.io/assets/pdf/2025-micro-taidl.pdf: the user-facing unit is called a "TAIDL definition" with "two main components -- the data model definition and the instruction semantics"; the paper's own examples (TPUv1, AMX) use only data models, calling attributes, control registers, and XLA-HLO semantics.
- StableHLO spec, https://openxla.org/stablehlo/spec: `cholesky` says "behavior is undefined" for a non-Hermitian-positive-definite input, `convert` out-of-range float to integer is "TBD", `bitcast_convert` layout is implementation-defined, and `output_operand_aliases` is an attribute "not yet described".
- AWS NKI ISA index, https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.isa.html: predicated ops (`tensor_copy_predicated`, `select_reduce`, `affine_select`), index-driven ops (`local_gather`, `nc_n_gather`), MX ops (`quantize_mx`, `nc_matmul_mx`), DMA ops with a descriptor-generation mode (`dge_mode`); no rounding-mode parameter is listed.
- PTX ISA (cited at `casestudies/hopper.md:4`): `cp.async.bulk.tensor`, `mbarrier`, `wgmma.mma_async`.

### Q2.1

Position: Option C, with aliasing decided by identical region text and any other overlap of an input and an output at issue a fault.

1. Every input is read before any output is written, so aliasing never changes the computed result (`language/spec-draft/instructions.md:54`, S5.2).
2. So the mark carries no meaning for a user; it exists for the allocator, which needs one static fact (same storage).
3. The mini_trn softmax already writes identical input and output regions (`examples/mini_trn.py:92-93`); under A or C the user writes nothing extra and the mark call vanishes.
4. A separate flag can disagree with the regions (flag set, regions differ); the front end rejects that today (`taidl/instruction.py:163-172`, `spec-draft/conformance.md:30`).
5. Deriving from text removes a whole error class; the remaining trap is two texts that evaluate equal.
6. Option D (alias by evaluated starts) cannot serve the allocator, which works from attribute expressions before addresses exist.
7. C answers the trap with a fault at issue, so the user gets a named error instead of a silent non-alias; the cost is one sentence, "same text means same storage, any other overlap is a fault".
8. Faulting on other overlap is not forced by owner 16 (read-before-write already defines the result); it is a choice that keeps the allocator sound, and the spec should say so.

Evidence: `taidl/instruction.py:163-172`, `act-backend/tests/test_extraction.py:507`, `act-backend/examples/mini_trn_double/mini_trn_double.py:136`, `reviews/spec/ambiguity.md:23`, StableHLO `output_operand_aliases` not specified (URL above).
Confidence: medium.
Would change my mind: an ISA where the same bytes are legitimately reached by two different texts and the user needs aliasing there (then B).

### Q2.2

Position: Option C, remove both marks and ship a documented constraint idiom for the two conditions.

1. Each mark only appends a constraint text the user can write (`taidl/instruction.py:216-218`); about 70 lines for no example caller (`:174-244`).
2. The marks also evaluate at a different state point than user constraints (`reviews/spec/ambiguity.md:19`), which is a hidden second rule for a user to learn.
3. Removing them without a replacement leaves users to re-derive the "apart or equal starts" disjunction, which is easy to get wrong (see the per-dimension logic at `taidl/instruction.py:200-214`).
4. A cookbook entry (C10 in `casestudies/plan/c10_modeling_cookbook.md`) gives the text once; that beats an API call because the user sees the constraint they are really asserting.
5. Only tests call the marks (`act-oracle/tests/test_constraints.py:41`, `act-backend/tests/test_generation.py:220-221`), so migration cost is internal.

Evidence: as cited; owner 10 allows removal.
Confidence: high.
Would change my mind: a real ISA description (not a test) that needs the marks in more than one instruction.

### Q2.3

Position: Option C with B's weighting: an extension needs an idiom-failure argument plus two real documented sites; invented ISAs count zero.

1. The language already has 78 operation words (`IDLV2.g4:45-74`); each construct is a concept the new user must learn, so the bar is "cannot be said", not "is verbose".
2. D28 shows the pattern: write masks have an idiom (output as input, `select`, in place), so no syntax (`language/plan/decisions.md:36`).
3. A bare count of studies is gameable: Eyeriss and CGRA have no public ISA (CP9), and rows bundle constructs (CP5).
4. A counted site must be a public ISA, so any reader can check it; NKI and PTX are public (URLs above).
5. The idiom-failure test is concrete: the idiom is lossy, or it makes generators give wrong answers (for example, the allocator over-approximates a strided footprint).
6. A per-extension review with no threshold (D) invites ad hoc admission; a numeric rule plus a failure argument is predictable.
7. The rule applies to items already scheduled (E), so L17 must also pass; I think it does (Q2.4).

Evidence: `reviews/casestudies-priorart/review.md:26,30` (CP5, CP9), `language/plan/extensions.md:5` (pending panel), `casestudies/gaps.md` rank list.
Confidence: medium.
Would change my mind: evidence that a failure-argument requirement blocks a construct that three real ISAs need and whose idiom is merely tedious.

### Q2.4

Position: L17 in v3 (shrunk); L19 and L23 are the two to decide next, both documented as idioms in v3; L18, L20, L21, L22 not in v3.

1. L17 has no idiom: an element type cannot be spelled by existing means, and a missing type blocks a whole ISA. NKI lists `nc_matmul_mx` and `quantize_mx` (URL above), Blackwell and XDNA2 are the other sites (`extensions.md` L17), so two real sites hold.
2. Keep only `f4e2m1fn`, `f8e8m0fnu`, `scaled_dot`; the rest has at most one user.
3. L18 masks: the NKI predicated ops are real, but the idiom works (D28); no syntax.
4. L21 rounding: NKI lists no rounding parameter; `clamp` then `convert` covers saturation; stochastic rounding needs seeded state and has no StableHLO reference; no syntax.
5. L20 convolution: `dot` plus `pad` and `slice` describes it (D27); defer to a real conv-engine ISA.
6. L22: names no accelerator and lets bodies into the grammar; cut.
7. L19 strided regions: gap rank 2 is High with six accelerators, and the idiom (cover the span, `slice` with strides) is correct but over-declares the footprint, which hurts overlap constraints. I rank it the strongest syntax candidate, but the idiom is expressible today, so under Q2.3 it waits for a failure demonstration in C2.
8. L23: `gather` and `dynamic_slice` already take indices from a region (CP3, `taidl/operations.py:120-122`), so the residual need is only static addresses; defer.

Evidence: `language/plan/extensions.md` L17 to L24, `reviews/spec/minimalist.md` verdict rows, `casestudies/gaps.md` ranks 2, 4, 8.
Confidence: medium (low on L19).
Would change my mind: C2 shows the span idiom makes the allocator reject a valid TMA box; then L19 enters v3.

### Q2.5

Position: Option C, no new state kind in v3; a separate design study for data-dependent addressing (D).

1. A user describing a descriptor wants a bundle of numbers that an instruction reads; existing buffers plus `gather` and `dynamic_slice` already read indices from a region (CP3).
2. A typed state kind is a second declaration form (state variable, buffer, now descriptor kind) for one need that C8 also targets (CP6); two mechanisms for one idea is the worst outcome for a learner.
3. Async groups model concurrency, which is out of v3 (Q2.6), so a kind for them has nothing to describe.
4. The backend rejects every state read and update today (`act-backend/act_backend/spec_checks.py:372-384`), so any new state kind would be unusable in the backend until that lands.
5. Indexed state (`@s.rows[@c.t]`): AMX `LDTILECFG` fits one instruction with 17 attributes and 17 ordered updates (CP2), clumsy but flat; I accept flat names until instruction families (gap rank 7) are decided, because indexing and families are one ergonomics problem.
6. TMA tensor maps live in memory and `cp.async.bulk.tensor` takes an address (PTX, `casestudies/hopper.md:4`), which matches "a region holds the descriptor".

Evidence: `reviews/casestudies-priorart/review.md:23-27`, `casestudies/gaps.md` ranks 7, 8, 10, `reviews/rulings/audit.md` R14 (as cited by the agenda).
Confidence: low to medium.
Would change my mind: a Hopper or Trainium2 sketch run through the front end that fails to express a descriptor the hardware mutates between issues.

### Q2.6

Position: Option B: out of v3, and the spec states plainly that program order is the only order, naming the fault a constraint-as-wait produces.

1. Timing is already outside TAIDL (`language/spec-draft/overview.md:10`, S1); a user must not guess whether `mbar_wait` blocks.
2. Under owner 16 a violated constraint is a fault (R55), so a wait written as a constraint faults in a sequential program (CP14); a user who copies a Hopper sketch gets a surprising fault unless the spec says it.
3. Owner 15 excludes SIMT threads; a core-grid construct (C) has no source syntax and would be the largest addition for one rank-9 row.
4. D (permanent exclusion) is stronger than the evidence: tile arrays are not SIMT, so I leave the door open for the study.
5. The idioms of L24 (grid as an extra addressed dimension, async DMA as two instructions sharing a token) are the user's answer now.

Evidence: `reviews/casestudies-priorart/review.md:35` (CP14), `casestudies/gaps.md` rank 9, `language/plan/extensions.md` L24.
Confidence: high.
Would change my mind: a public ISA whose single instruction semantics depends on another core's concurrent write.

### Q2.7

Position: Option C plus D for sketches behind an admitted extension; no new study before v3.

1. The evidence base is the rule's input (Q2.3); wrong verdicts inflate it (CP2, CP3, CP12), so correcting and recounting comes first.
2. A sketch that does not parse is weak evidence (R16); running every sketch is large, running the ones behind L17 and any later candidate is cheap and sufficient.
3. A study of Ascend (A) has a stated reason that argues against it (audit R15); NVDLA or Gaudi TPC (B) fills a real class gap (CP7) but is not needed to decide v3 contents.
4. The Hopper sketch already is the GPU tensor-core description (E); it needs its missing attributes and rank fixes (CP13, CP15), not a new file.
5. For me as a user, a descriptor-engine ISA is the study I would most like to see, so queue NVDLA or Gaudi TPC after v3.

Evidence: `reviews/casestudies-priorart/review.md:23-28,34`, `casestudies/README.md:5-17`.
Confidence: medium.
Would change my mind: a descriptor-engine study showing a High-severity gap that the ten sketches cannot reveal.

### Q2.8

Position: Option A: remove all three in v3 (C gives the same set).

1. The forms `BOOL`, `NUMBER`, `TRANSPOSES` are used only by these three words (`taidl/operations.py:18-23`, `:145-156`; no other row uses them), so C and A coincide.
2. StableHLO defines `cholesky` as undefined behavior for non-positive-definite input (URL above), which cannot stand under owner 16; keeping it means inventing a fault rule for a word no accelerator has.
3. Operand order for these words is unanswered in S8.2 (implementer Q32); the user cannot write them correctly from the spec today.
4. Removal deletes three attribute forms from the language the user must learn; reintroduction at the second real need follows the Q2.3 rule.
5. D (unsupported marker) keeps the cost and adds a coverage table to read.

Evidence: `taidl/operations.py:145-156`, `reviews/spec/minimalist.md` verdict row, `act-oracle/docs/coverage.md`.
Confidence: high.
Would change my mind: a tensor-accelerator ISA with a hardware solver or normalization instruction.

### Q2.9

Position: A remove presumed reshape; B keep `d0` special but state it; C keep the keyword only; D defer `ROOT`; E per R11.

1. Presumed reshape (S7.2): the spec says a line may give a region value another shape with the same element count (`spec-draft/semantics.md:48`); an invisible reshape is exactly the silent surprise a new user hits, and 196 explicit `reshape` lines show users write it anyway (minimalist row). Remove, require one `reshape`.
2. `d0`: the user never declares it (`taidl/accelerator.py:31`, `spec-draft/declarations.md:17`); the special case at `taidl/instruction.py:123` is invisible if the spec says "hbm is a `u8` buffer with one addressed dimension of run-time size". Making it an ordinary model buys the user nothing, since its size is run-time (RA10).
3. `add_constraints` beside `constraints=` (`taidl/accelerator.py:66`, `taidl/instruction.py:321-323`): two spellings of one list; keep the keyword, since the user then sees the whole instruction in one call.
4. `ROOT`: it keeps a semantics pasteable as XLA-HLO (TAIDL paper section 2.4 relies on XLA-HLO text); defer, change nothing (R9 would add a rule for no user need).
5. Instruction without semantics: legal only without regions (R11); a state-only configuration instruction is something every ISA has (Gemmini config, AMX `ldtilecfg`), and a region without meaning is a mistake the front end should reject.

Evidence: as cited; `reviews/spec/minimalist.md` rows `d0`, `add_constraints`, `ROOT`, presumed reshape.
Confidence: medium (high on A and E).
Would change my mind: a sketch where presumed reshape removal inflates semantics lines across many instructions.

### Q2.10

Position: A for the grammar (name the `.g4` files, keep token tables and binding notes); D for reserved names (keep the table in S4, labeled as the oracle's list).

1. A copied rule set kept equal by CI is a second source; it already drifted once (`taidl/antlr4/IDLV2Expr.g4:25`, `compare`).
2. A user does not need ANTLR rules; the user needs token spellings, precedence, and the sentence "bare words are only attribute names or values".
3. A user hits a reserved-name error at generation time, before reading oracle documentation (D17), so the list must be in the document they have open.
4. Cutting the table (C) makes the first error unexplained; generating it (E) is better but needs a build step not justified by a short list.
5. A label ("these names are reserved by the oracle") keeps the spec honest that the front end does not enforce them.

Evidence: `language/plan/decisions.md:17,25`, `language/spec-draft/declarations.md:44-48`, `act-oracle/docs/coverage.md`.
Confidence: medium.
Would change my mind: a second generator with its own reserved list (then E).

### Q2.11

Position: Option C: the spec fixes only the names a generator must read; the rest of the object model lives in the `taidl` README.

1. The user of Q2.11 is the description author; they never touch `constraint_asts` or tree accessors, so these do not belong in the document that teaches the language (minimalist row "S6 Parsed trees").
2. A generator author needs a small, stable contract; five names (`constraint_asts`, `update_asts`, `instr_input_asts`, `instr_output_asts`, `instr_expression_asts`) are enough to state normatively (`taidl/instruction.py:131-140`).
3. Fixing every attribute (A) freezes front-end internals that a v3 rewrite (L3, R62) will change.
4. Keep the abstract issue model (S5.6) in the spec: a user must know the order of reads, updates, and faults in one issue (`spec-draft/execution.md`).
5. A serialized format (D) has no non-Python consumer yet (R42).

Evidence: `reviews/spec/minimalist.md` row S6, `language/plan/spec.md:24,122`, `spec-draft/object_model.md` (50 lines), `spec-draft/execution.md` (41 lines).
Confidence: low to medium.
Would change my mind: the generator-author panel showing that a third-party generator cannot be written from five names plus the README.

### Q2.12

Position: E with a caveat (the paper's term is "TAIDL definition"); rename user-visible and generator-read Python names once in v3, delete unread names, leave the grammar rule `instruction` (B/C hybrid).

1. File name: owner 14 puts papers first when precise. The TAIDL paper writes "TAIDL definition" throughout and never "ISA description" as its own term (the one hit is "scalar ISA description languages"), so R1's reason does not hold on the text.
2. "TAIDL definition" is precise, in the paper, and matches the glossary pattern (the user's file defines an accelerator); "ISA description" is what the draft spec already uses (`spec-draft/overview.md:8`, `glossary.md:10`). I weakly prefer the paper's term; the owner decides.
3. Renames: the first thing a user types is `add_instruction(name, computation_attr, addressing_attr)` (`taidl/accelerator.py:60-62`); names the user sees in the signature should match the glossary, and v3 is the one breaking window (owner 10).
4. Rename Python names the user or a generator reads (`parameters`, `comp_attr`, `var_type`); delete names no generator reads (`num_dim_str`, minimalist L15).
5. The grammar rule `instruction` needs parser regeneration (D13) and no user sees it; leave it.
6. The glossary must follow owner 19 (buffers, state) before any rename starts; the audit says it still renames to `add_tensor_buffer` (`reviews/rulings/audit.md` R2).

Evidence: TAIDL paper text (URL above); `language/plan/terminology.md`; `language/plan/review.md:30`; `language/plan/decisions.md:21`.
Confidence: low (file name), medium (renames).
Would change my mind: the ACT paper using "ISA description" as its term for the user's file.
