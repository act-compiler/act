# Panel 2 agenda: language surface

Scope: constructs to keep or remove, extensions and their admission rule, scope, and spec presentation.
Debaters: Sonnet (new user writing an ISA description), Opus (minimalist language designer), Fable (accelerator architect).
Moderator: Opus.
Draft answers (R-numbers) are one option among others and carry no default.
Binding owner decisions for the whole panel:

- 10: breaking changes are allowed, batched into one coordinated major release (TAIDL v3 and matching generator releases).
- 15: scope is tensor accelerators plus GPU tensor-core instructions (mma, wgmma, TMA); no SIMT thread model.
- 16: no undefined behavior in v3; every run-time violation is a defined error or a defined behavior under one simple norm.
- 17: the in-place mark means the output reuses the marked input's storage.
- 19: a data model has buffers (not "tensor buffers") and state (not "initial state"); `hbm` stays `hbm`.
- 20: an instruction's view of a buffer is a "region" (not "data slice").
- 21: the state reference prefix `@s.` stays.

Source key:

- Ambiguity review: `reviews/spec/ambiguity.md`.
- Implementer review: `reviews/spec/implementer.md`, questions Q1 to Q38.
- Minimalist review: `reviews/spec/minimalist.md`.
- Case-study review: `reviews/casestudies-priorart/review.md`, findings CP1 to CP20.
- Gap matrix: `casestudies/gaps.md`, ranks 1 to 18.
- Proposals: `language/proposals.md` and `language/plan/extensions.md`, items L17 to L24.
- Decisions: `language/plan/decisions.md`, items D9, D13, D17, D28.
- Audit: `reviews/rulings/audit.md`, rows R-n and RA1 to RA13.
- Draft answers: `minutes/rulings.md`.
- No `language/review-response.md` exists.

## Questions

### Q2.1: How is an in-place input tied to its output in v3?

- Context: owner decision 17 fixes the meaning as shared storage; the open point is how a description says which input and which output share it.
- Context: the spec does not say which output aliases which input, or whose starts win when they differ (ambiguity S5.5); the front end checks neither (`taidl/instruction.py:163-172`).
- Context: back-end and oracle tests call the mark today (`act-backend/tests/test_extraction.py:507` marks a `d0` input in place for a `d2` output).
- Option A: remove `set_inplace`; an input and an output alias exactly when their region texts are identical (R5, minimalist verdict "merge, derive it").
- Option B: keep `set_inplace` as an explicit call; the spec states that flag k makes output region 0 the same storage as input region k, same data model and sizes, equal starts when issued or a fault (ambiguity S5.5).
- Option C: derive aliasing from region identity as in A, and add that any other overlap of an input and an output at issue is a fault, including equal regions spelled differently (audit R5, RA2).
- Option D: derive aliasing from the evaluated starts and sizes at issue rather than from text; texts that differ but evaluate equal alias (raised in audit R5 as the unaddressed case).
- Evidence: R5, R21, R22; audit rows R5, R21, RA2, RA3; ambiguity S5.5 row; minimalist verdicts on `set_inplace`; `act-backend/examples/mini_trn/mini_trn.py:92-99`; `semantics/rulings.md:8-13` keeps `set_inplace`; implementer Q36 (no example uses marks).
- Evidence: L24 masked-write idiom depends on an in-place input (`language/plan/extensions.md` L24, D28).
- Binding: owner 17, owner 16 (a mismatch must be a rejection or a fault), owner 10 (removal is allowed in the v3 batch).

### Q2.2: Are the placement marks `set_apart` and `set_no_partial_overlap` kept in v3?

- Context: each mark only appends one constraint text that a user can write (`taidl/instruction.py:216-218`); the front end spends about 70 lines on them (`taidl/instruction.py:174-244`).
- Context: no example or tutorial calls them; tests in all three repos and spec-draft pages do (audit R58).
- Context: placement-mark constraints read the state at a different point than user constraints (ambiguity S5.3/S5.4/S5.5 row; audit R7).
- Option A: remove both marks; the same condition is written as a constraint (R58, minimalist "cut").
- Option B: keep both marks and fix their constraint evaluation point in S5.4 (ambiguity row).
- Option C: remove both marks and ship a documented constraint idiom or helper for the two conditions, so users do not re-derive them (implied by audit R58 migration list).
- Option D: keep one mark and remove the other (no source argues for it; listed only to fix the option space).
- Evidence: R58; audit R58 migration list (`taidl/tests/test_instruction.py`, `taidl/tests/test_expressions.py:183`, `act-oracle/tests/test_constraints.py:41`, `act-backend/tests/test_generation.py:220-221`, spec-draft `instructions.md`, `conformance.md`, `glossary.md`); minimalist cuts list; L19 caveat on overlap rules with strides.
- Binding: owner 10 (breaking removals allowed); owner 16 (a violated constraint is a defined error either way).

### Q2.3: What rule admits a language extension, and who counts the evidence?

- Context: R59 admits an extension in v3 only if at least two case studies need it, and the plan file calls it pending a panel.
- Context: the counts in the gap matrix are disputed: bit operators and indexed state have a second site once descriptors and barriers are counted (CP5), and rows from invented ISAs inflate ranks (CP9).
- Context: the plan's own text says no extension changes the grammar this quarter without its own review (`language/plan/extensions.md` header).
- Option A: two case studies per construct, counted per construct in a new matrix column (R59, CP5).
- Option B: two real accelerators with public documentation, with invented-ISA studies (Eyeriss, CGRA) counted at half weight or not at all (CP9).
- Option C: no syntax unless an idiom cannot describe it; an extension needs an idiom-failure argument plus two sites (minimalist "speculative generality" section, D28).
- Option D: a per-extension review with no numeric threshold (extensions.md header).
- Option E: the rule applies to the whole of the extension plan, including items already scheduled for Q2 (minimalist "apply it to the whole of Q2").
- Evidence: R59; CP4, CP5, CP9, CP11; gap matrix ranks 10 and 14 to 18; minimalist "Speculative generality"; D28.
- Binding: owner 15 (accelerator scope); no owner decision fixes a numeric threshold.

### Q2.4: Which of L17 to L23 enter v3, and which stay idioms or are cut?

- Context: eight proposals exist; the plan schedules L17 in Q1, L23 late Q2, and defers or cuts the rest (`language/plan/extensions.md` table).
- Context: five case-study items duplicate L rows under different syntax (CP4); minimalist finds only L17 and L23 add something no idiom gives.
- Context: each item below has a sourced position; the panel weighs them under the rule chosen in Q2.3.
- Option A, L17 (types, `scaled_dot`): keep in v3 for `f4e2m1fn`, `f8e8m0fnu`, `scaled_dot`, defer `f6*`, `s1`, `u1` (minimalist, extensions.md); or ask the owner whether Blackwell and XDNA2 count as the second site (CP11); or drop `tf32` as a storage type (CP11).
- Option B, L18 (write masks), L20 (`convolution`), L21 (rounding and saturation on `convert`): describable today by idiom (D28, minimalist) and deferred; or keep one of them as syntax because of its named users (proposals.md rows 2, 5, 6); L21 stochastic rounding has no StableHLO reference.
- Option C, L19 (strided regions): idiom of covering the span and a strided `slice` (minimalist); or syntax, with overlap rules extended to strides (extensions.md L19, CP18 places stride as a region property).
- Option D, L22 (sub-computations): cut from v3 (extensions.md, minimalist); no source argues for admission.
- Option E, L23 (indirect regions): late Q2 second extension (extensions.md); or no syntax because `gather` and `dynamic_slice` already take indices from a region (CP3); or admit with the loss of static addresses in the allocator accepted (L23 caveats).
- Evidence: `language/proposals.md` ranks 1 to 8; `language/plan/extensions.md` L17 to L24; CP3, CP4, CP10, CP11; minimalist verdict rows L17 to L24; D28.
- Binding: owner 15 (SME and SVE are CPU vector extensions outside scope per D28); owner 10 (additive items are not breaking).

### Q2.5: Do descriptors, async groups, and indexed state become typed state kinds in v3?

- Context: R14 puts typed state kinds for descriptors and async groups (P11) in v3, citing Hopper TMA and Trainium DMA descriptors.
- Context: the audit finds async groups model concurrency and have fewer than two users, while the case studies' real gap is data-dependent addressing (gap matrix rank 8).
- Context: C8 (read a descriptor from a region) and P11 (declare it as a state kind) solve one need twice (CP6); indexed state has one site by `gaps.md` and a second by CP5.
- Option A: add typed state kinds for descriptors and async groups in v3 (R14).
- Option B: add descriptor kinds only; defer async groups to the concurrency study (audit R14).
- Option C: add no new state kind; describe descriptors with existing state variables, `gather`, and `dynamic_slice` (CP3 expressibility argument, gap matrix rank 8 verdict).
- Option D: treat data-dependent addressing as a separate design study (audit RA9), outside v3 unless two case studies need it.
- Option E: add indexed state (`@s.rows[@c.t]`) because AMX tiles, Hopper `mbarrier`, and Trainium2 semaphores need it (CP5); or express AMX `LDTILECFG` as one instruction with 17 attributes and 17 ordered updates and leave state flat (CP2).
- Evidence: R14, R27, R59; audit R14, RA9; CP2, CP3, CP5, CP6, CP12, CP13; gap matrix ranks 8, 9, 10, 14; minimalist "State variables and updates: keep, do not extend"; backend rejects every state read or update (`act-backend/act_backend/spec_checks.py:372-384`).
- Binding: owner 15 (TMA is in scope); owner 16; owner 21 (`@s.` stays); owner 19 (state, not "initial state").

### Q2.6: Is multi-core and tile-array concurrency in or out of the v3 language?

- Context: R13 puts it outside v3 as a Q2 study with implementation after v3; L24 documents idioms (a core grid as an extra addressed dimension, async DMA as two instructions sharing a state token).
- Context: the spec states timing is outside TAIDL (S1); a constraint used as a wait (`mbar_wait`, `lock_acq0`) becomes a fault under a sequential program order (CP14).
- Context: gap matrix rank 9 lists Trainium2, Hopper, AIE-ML, and Tensix as needing concurrency or instances.
- Option A: concurrency is out of v3; the idioms of L24 are the documented answer and a design study follows (R13, L24).
- Option B: out of v3, and the spec states plainly that program order is the only order, naming the fault produced by a constraint-as-wait (CP14).
- Option C: add a core-grid or instance construct in v3 for tile arrays (gap matrix rank 9 supporters; no source gives syntax).
- Option D: out of scope permanently as a SIMT-like thread model, with spatial mappings such as Groq, Cerebras, and SambaNova named as outside (CP7, proposals.md "Already describable").
- Evidence: R13, R14; L24; D28; CP7, CP14; gap matrix rank 9; audit R14; `language/spec-draft/overview.md:10`.
- Binding: owner 15 (no SIMT thread model); owner 16.

### Q2.7: Which case studies form the evidence base for the extension rule?

- Context: the ten sketches are faithful but three verdicts are wrong in the direction that inflates the gap matrix (`ldtilecfg`, indirect DMA, `smstop`), and shape bugs show the sketches were not run (CP2, CP3, CP12, CP15).
- Context: the descriptor-driven engine class (NVDLA, Gaudi MME and TPC, Apple ANE, MTIA fixed-function) has no study except an invented command set (CP7).
- Context: R15 adds an Ascend study at the start of C12; the audit finds its stated reason argues against it.
- Option A: add an Ascend study (R15).
- Option B: add NVDLA or Gaudi TPC beside or instead of Ascend, chosen for public docs (CP7).
- Option C: add no study; correct the verdicts and recount the matrix (CP2, CP3, CP5, CP8, CP9, CP12, CP16).
- Option D: run every sketch through the front end before it counts as evidence (R16, C11), or only the sketches that support an admitted extension.
- Option E: add a GPU tensor-core description as the extra real ISA description (R47 in the test panel scope; named here because it fixes the evidence base).
- Evidence: R15, R16, R47; audit R15; CP1 to CP3, CP7, CP9, CP12, CP13, CP15; `casestudies/README.md`; gap matrix.
- Binding: owner 15 (GPU tensor-core in scope; SIMT out).

### Q2.8: Are `cholesky`, `triangular_solve`, and `batch_norm_inference` removed?

- Context: each appears once, in a back-end or oracle test; no case-study accelerator has such an instruction.
- Context: they alone need the `BOOL`, `TRANSPOSES`, and `NUMBER` attribute forms (`taidl/operations.py:18-23`, `:145-156`) and the oracle's `f16` partial row.
- Context: S8.2 gives operand counts only; operand order for these words is unanswered (implementer Q32).
- Option A: remove all three in v3 and reintroduce at the second real ISA description that needs them (R60, minimalist).
- Option B: keep all three, with operand order and element types stated in S8.2 and S8.3 (implementer Q32).
- Option C: remove only the words that need a form no other word uses; keep any whose forms are shared (follows from the forms listed in `taidl/operations.py`).
- Option D: keep the words but mark them as unsupported by generators that lack them (coverage rows in `act-oracle/docs/coverage.md`).
- Evidence: R60; minimalist verdict "cut"; implementer Q32; `taidl/operations.py:18-23`, `:41-165`, `:145-156`; `act-oracle/docs/coverage.md`.
- Binding: owner 10; owner 15 (accelerator scope).

### Q2.9: Which redundant spellings and implicit rules are removed or made explicit?

- Context: the minimalist review lists constructs with two spellings or implicit rules; each is small, but together they change the batch of breaking edits.
- Context: the same constructs have partial support in the generators (presumed reshape in the oracle, `d0` special case in `taidl/instruction.py:123`).
- Option A, presumed reshape (S7.2): remove it and require one explicit `reshape` line (minimalist); or keep it and state the rule (S7.2 as written; implementer Q22 asks for a worked example).
- Option B, `d0`: declare it as an ordinary data model with one addressed dimension (minimalist); or keep it special and state it as `data_model[0]` with empty lists (implementer Q3, Q12; audit RA10 sets its size as an issue-time parameter).
- Option C, `add_constraints` beside the `constraints=` keyword: keep the keyword only (minimalist); or keep both.
- Option D, `ROOT`: defer any change (minimalist); remove it; or allow it on a non-last line (R9).
- Option E, instruction without semantics: legal only without regions, with regions rejected (R11); or require all of semantics, inputs, and outputs together or none (ambiguity S7.2/S5.2 row).
- Evidence: R9, R11; minimalist verdicts for `d0`, `add_constraints`, `ROOT`, presumed reshape; ambiguity S7.2/S5.2 row; implementer Q3, Q12, Q19, Q21, Q22; `taidl/accelerator.py:31`, `:66`; `taidl/instruction.py:99-100`, `:123`, `:254-257`, `:276-318`.
- Binding: owner 10; owner 16; owner 20 (region vocabulary).

### Q2.10: How does the spec present the grammar and the reserved names?

- Context: S3 copies the ANTLR rules of both `.g4` files and a CI test keeps the copy equal; the copy had drifted once (S3.2 omitted rule `compare`).
- Context: S4 lists names reserved by TAIDL-TO; the oracle's `coverage.md` already points at the oracle's README for the same list, and a user meets the error before reading the oracle's docs (D17).
- Option A, grammar: the grammar page links the `.g4` files and keeps the token tables and binding notes (R61, D9, minimalist).
- Option B, grammar: keep the copied rules and extend the CI test to diff them against the `.g4` files (L8 as written, minimalist notes the second source).
- Option C, reserved names: cut the table from the spec (R61, minimalist).
- Option D, reserved names: keep the table in S4, labeled as the oracle's list (D17).
- Option E, reserved names: keep a table but generate it from the generators' own lists, so it cannot drift.
- Evidence: R61; D9, D17; minimalist rows S2/S3 and S4; ambiguity "Grammar page against the .g4 files"; `taidl/antlr4/IDLV2Expr.g4:25`; `act-oracle/docs/coverage.md`.
- Binding: no owner decision fixes the page layout.

### Q2.11: Do the object model and execution pages stay in the language spec, and how much of the front-end API do they fix?

- Context: S4.6 (`object_model.md`) and S5.6 (`execution.md`, programs and issues) are planned pages; L29 adds a contract test for the object model.
- Context: R42 says the front-end-to-generator contract is a documented, typed Python object model rather than a serialized format.
- Context: the implementer could not answer attribute names, region storage forms, tree node types, the abstract issue model, or the exception for an unsupported feature from the spec alone (implementer Q1 to Q10, Q35).
- Context: minimalist says "Parsed trees" in S6 is front-end API, not language, and belongs in the `taidl` README.
- Option A: both pages stay in the spec as normative, with attribute tables, tree accessors, and the issue model (implementer Q1 to Q10, R42, L29).
- Option B: the object model moves to the `taidl` README; the spec keeps only the language and the abstract issue model (minimalist on S6).
- Option C: the spec fixes only names a generator must read (`constraint_asts`, `update_asts`, `instr_input_asts`, `instr_output_asts`, `instr_expression_asts`) and leaves other attributes unspecified.
- Option D: a serialized interchange format replaces the Python object model (R42 rejects it for now; no non-Python generator exists).
- Evidence: R42; L29 in `language/plan/spec.md:122`; `language/plan/spec.md:24`; implementer Q1 to Q10, Q18, Q35, and its "does well" list; minimalist row "S6 Parsed trees"; `taidl/accelerator.py:11-15`, `taidl/data_model.py:33-50`, `taidl/instruction.py:131-140`, `:249-252`, `:364`.
- Binding: owner 19 (buffers, state), owner 20 (region), owner 10.

### Q2.12: What is the user's file called, and which internal names are renamed in v3?

- Context: R1 names the user's file an "ISA description" and keeps "spec" for the language document; R2 renames internal names in v3 per the glossary.
- Context: the audit finds R2 points at a glossary file that still renames to `add_tensor_buffer` and `add_control_register`, which owner decision 19 forbids.
- Context: the review lists internal names that clash with the terms: `parameters`, `comp_attr`, `var_type`, the grammar rule `instruction`, and the generator-only `array_dim_str` and `num_dim_str` (`language/plan/review.md`).
- Option A: rename all of them in v3 under the reconciled glossary and update B and O visitors (R2, L15).
- Option B: rename the Python attributes now; leave the grammar rule `instruction` for later because it needs regenerated parsers (D13).
- Option C: rename `parameters` and delete `num_dim_str` only; skip the rest as churn in three repos (minimalist L15 "shrink").
- Option D: rename nothing; fix the glossary to the code's names.
- Option E, file name: "ISA description" (R1); or another term from the owner's papers if it is more precise (owner 14 order: papers first).
- Evidence: R1, R2; audit R2; D13; `language/plan/terminology.md`; `language/plan/review.md` weaknesses table; minimalist L15 row; implementer Q2, Q4; `taidl/instruction.py:34-35`; `taidl/data_model.py:33-50`.
- Binding: owner 14 (precedence), 19, 20, 21 (`@s.` stays), 10.

## For the authors

- `taidl/docs` S3.2 omits `compare : EQ | NE | LT | LE | GT | GE ;` (`taidl/antlr4/IDLV2Expr.g4:25`); add it if the grammar page keeps rules.
- S2.2 never states longest-match and earlier-rule-wins for expression tokens and lists no `WS` row (`IDLV2Expr.g4:62`).
- S2.1 lists no `COMMENT` or `WS` row, and its NUMBER note hides that `-inf` and `-nan` lex as `NUMBER` (`IDLV2.g4:91`, `:96-97`).
- S2.1: `INT` is decimal; leading zeros do not change its value.
- S7.1: replace the bare-word sentence with "a bare word stands only as an operation attribute's name or value, or as `constant`'s `true` or `false`".
- S7.1: a semantics defines at least `%Out0`; exactly one line carries `ROOT`.
- S5.1: each list is a list of `str`; a bare string or a non-`str` entry is rejected with `TypeError`.
- S4.1: a name is not in `keyword.kwlist` of a named Python version; soft keywords are allowed.
- S8.2: state the default of `type` when absent; add an operand-order column or the sentence "as the reference operation orders them" (implementer Q28, Q32).
- S8.3: state `dot` operand and result element types, and the operand checked for `select`, `clamp`, `compare`, and `is_finite`.
- S8.4: add "`copy` returns its operand" (implementer Q37); pin the StableHLO and XLA-HLO versions (implementer Q26, Q27).
- S8.5: the text gives every shape as a function of the instruction's attributes and state.
- S9: add the missing rows listed in ambiguity "Missing S9 entries" and move the S1.3 rows that are statically checkable to S9.
- Add one example per S5 feature (state, constraints, updates, marks, multiple outputs, in-place) (implementer Q36, `gaps.md` doc gaps).
- Add a worked example for a scalar data model and for `constant` with a backtick expression (implementer Q22, `gaps.md` doc gaps).
- `casestudies/gaps.md` rank 1: move to Medium after a fixed byte order (CP8); rank 12: rename to aliased buffers over one storage (CP16); rank 16: delete after `smstart` and `smstop` are fixed (CP12).
- `casestudies/amx.md`: rewrite `ldtilecfg_t0` as one instruction (CP2); `trainium2.md` `dma_copy`: change the verdict to expressible with workaround (CP3).
- `casestudies/sme.md`: zero `z` and `p` in `smstart` and make a ZA access with `@s.za_on == 0` a fault (CP12).
- `casestudies/hopper.md`: add `scale-d`, `imm-scale-a`, `imm-scale-b`, `imm-trans-a`, `imm-trans-b` and the group-wait sequence; fix `slice` rank bugs (CP13, CP15).
- Mark invented-ISA rows in the gap matrix (CP9); trim C5 to `convolution` and a bounded `scan` (CP10); drop `tf32` from C6 (CP11).
- P2 taxonomy: replace "undefined" by the words rejected, fault, and defined value (CP1).
- P7 and C11: one sentence each naming the other and the boundary (CP17); stride as a region property, alignment as a buffer property (CP18).
- L3 says "70 words"; the grammar has 78 operation words (`IDLV2.g4:45-74`).
- RA1: R34 is missing from `minutes/rulings.md`; restore it or mark the number retired.
- Audit R26: the reason for keeping `Buffer` is the owner's term (owner decision 19), not the papers' term.
- The six A tutorials still use the retired `ENTRY` and `parameter(0)` syntax (`act/tutorials/pldi26/exercise3/solution/QKV.py:25-26`); no fail point relies on a tutorial run until they move (R63).
