# Panel 2, round 1: Opus (minimalist language designer)

Assumptions:
- A construct earns its place only if no existing construct describes the same behavior without loss, at two real sites.
- A construct that only re-spells another is a second source of truth and a second thing every generator must handle.
- Owner decisions 10, 15, 16, 17 bind; v3 is the only breaking window, so a cut deferred past v3 is a cut not made.

### Q2.1

Position: Option C: aliasing is derived from identical parsed region texts, and any other input/output overlap at issue is a fault.
1. Under owner 17 an in-place input and its output share storage, so they are one region; the flag restates what the two region texts already say.
2. Text identity is static, so B's allocator still sees the flag before any issue, as it reads it today.
3. Option D makes aliasing a run-time property no generator can see at build time, and two meanings (alias versus no alias) then hang on attribute values.
4. Any undeclared overlap is a fault under owner 16: a description declares exactly where the hardware supports reuse, and the oracle checks the rest on concrete values.
5. Equality is taken on parsed trees, as `keep_off` already compares starts, so spacing never changes the answer.
6. The one test that breaks marks a `d0` input in place for a `d2` output, which cannot share storage under owner 17; that test is ill-formed and migrates.
7. The L24 masked-write idiom keeps working: the input region is spelled as the output region, so the flag is derived.
Evidence:
- 1: owner decision 17; `act-backend/examples/mini_trn/mini_trn.py:97-99` (input and output texts equal, then `set_inplace([True])`).
- 2: `act-backend/act_backend/instruction_info.py:112` (B reads the flags); minimalist review, "Order and sizes".
- 3, 4: audit R5 and RA2; owner decision 16; `language/spec-draft/execution.md` (S5.6 program and issue model).
- 5: `taidl/taidl/instruction.py:214` (`out_tree.getText() == tree.getText()`).
- 6: `act-backend/tests/test_extraction.py:507` (`add_load(acc, 'load_d2', 'd2').set_inplace([True])`).
- 7: D28 in `language/plan/decisions.md:36`; MLIR rationale prefers removing a redundant distinction "to make the IR simpler" (https://mlir.llvm.org/docs/Rationale/Rationale/).
Confidence: high.
Would change my mind: a real ISA whose in-place form needs input and output regions spelled differently (for example different data models naming one storage), which CP16's aliased buffers would raise.

### Q2.2

Position: Option A: remove both marks with no helper, because Q2.1's overlap fault already states both conditions by default.
1. Each mark only appends a constraint text a user can write; 70 lines of front end exist for that.
2. Under Q2.1 C, an input that is not in place and overlaps the output faults at issue; that is `set_apart` as the default.
3. `set_no_partial_overlap` permits "equal starts or apart"; equal is now spelled as in-place and apart is the default, so its condition is also covered.
4. A helper (Option C) is a third spelling of the same rule and needs its own docs and tests.
5. The marks also carry a second evaluation point for state (audit R7); removal deletes that ambiguity instead of specifying it.
Evidence:
- 1: `taidl/taidl/instruction.py:174-244`, `:216-218`.
- 2, 3: Q2.1 position; audit RA2.
- 5: audit R7; ambiguity row S5.3/S5.4/S5.5.
- Migration list: audit R58 (`taidl/tests/test_expressions.py:183`, `act-oracle/tests/test_constraints.py:41`, `act-backend/tests/test_generation.py:220-221`).
Confidence: high if Q2.1 C is adopted; medium otherwise, where Option A still holds with a documented constraint line.
Would change my mind: a real ISA that permits overlap of an input and output that is neither identical nor disjoint and defines the result.

### Q2.3

Position: Options C with A and E: syntax enters only with an idiom-failure argument plus two real, publicly documented accelerators in owner scope, counted per construct, for every extension.
1. The engineering rule is abstraction at the second use site; the first site is served by an idiom.
2. A count is evidence only if the site is real: invented command sets (Eyeriss, CGRA) are designs the track wrote, so they count zero, not half.
3. A count is evidence only if the idiom fails: a construct with a lossless idiom is verbosity, which two sites do not justify alone.
4. Counting per construct stops bundling from hiding a cut (C3 with indexed state, C9 with bit operators).
5. A site counts only when its sketch runs through the front end, since unrun sketches carried shape bugs and wrong verdicts.
6. The L-track lead writes the count in one gap-matrix column; a reviewer outside the track checks each cited site before admission.
7. StableHLO keeps a closed op set and MLIR stores a variant as an attribute rather than a new op, so growth by attribute or idiom is the norm in the reference systems.
Evidence:
- 1: `/home/user/.claude/.claude/engineering.md` "Things to avoid"; R59.
- 2: CP9; `casestudies/gaps.md` ranks 3 and 13.
- 3: minimalist "Speculative generality"; D28.
- 4: CP5.
- 5: CP15; R16.
- 7: https://openxla.org/stablehlo/spec ("a closed set of high-level operations"); https://mlir.llvm.org/docs/Rationale/Rationale/ (comparison kind as an attribute, not ten ops).
Confidence: high on the rule; medium on who counts.
Would change my mind: a construct with a lossless idiom whose idiom size is shown to block real users (for example an instruction count that grows past the allocator's limits).

### Q2.4

Position: admit only the L17 types `f4e2m1fn` and `f8e8m0fnu` in v3; `scaled_dot`, L18, L19, L20, L21, L23 stay idioms; L22 is cut.
1. The two MX types have no idiom: a storage type with a 4-bit or exponent-only layout cannot be spelled with existing types, and StableHLO already defines both.
2. `scaled_dot` has an idiom: `convert` each operand, `multiply` by the broadcast power-of-two scale, then `dot`; the scale multiply is exact, so the idiom is lossless.
3. `tf32` is dropped as a storage type; `reduce_precision` models it (CP11).
4. L18 has the D28 idiom; its named sites are SME and SVE, CPU extensions outside owner 15.
5. L19 has the span-plus-strided-`slice` idiom; syntax would also force stride-aware overlap rules into Q2.1.
6. L20 is `dot` plus `pad` and `slice` (D27); L21 saturation is `clamp` then `convert`, and stochastic rounding has no StableHLO reference.
7. L23: `gather` and `dynamic_slice` read indices from a region, and `dynamic_update_slice` writes at a data-dependent start, so indirect access is a covering region plus those words; syntax would cost B its static addresses.
8. L22 names no accelerator and is the change that lets bodies into the grammar.
Evidence:
- 1: https://openxla.org/stablehlo/spec (lists `f4E2M1FN` and `f8E8M0FNU` MX types); `language/plan/extensions.md` L17.
- 2: `taidl/taidl/operations.py` (`convert`, `multiply`, `broadcast`, `dot` rows); L17 names `scaled_dot` as XLA's, not StableHLO's.
- 3: CP11. 4: D28 (`language/plan/decisions.md:36`); owner decision 15.
- 5: minimalist L19 row; extensions.md L19 caveat. 6: D27; minimalist L20 and L21 rows.
- 7: `taidl/taidl/operations.py:120-122`; CP3; L23 caveats.
- 8: extensions.md table row L22.
Confidence: high on L18 to L22; medium on `scaled_dot` and L23.
Would change my mind: for `scaled_dot`, a hardware accumulation order that the idiom cannot match within the R8 tolerance; for L23, a real indirect write whose covering region exceeds a buffer.

### Q2.5

Position: Options C and D: no new state kind in v3; descriptors use existing state, attributes, and `gather`; async groups go with concurrency; data-dependent addressing is a separate study.
1. B refuses every state read or update today, so a new state kind would extend a feature no generator executes.
2. One `LDTILECFG` is one instruction with 17 attributes and 17 ordered updates under S5.4, so AMX needs no indexed state for configuration.
3. A TMA tensor map is encoded on the host and fixed per kernel; its fields enter a load instruction as attributes, which describes the copy.
4. Async groups model completion order in time, which S1 places outside TAIDL.
5. C8 and P11 are two solutions for one need, so admitting P11 now commits to one before the RA9 study compares them.
6. Indexed state has AMX `tileloadd` as its one non-concurrency site; Hopper `mbarrier` and Trainium2 semaphores count only once concurrency is in.
Evidence:
- 1: `act-backend/act_backend/spec_checks.py:372-384`. 2: CP2.
- 3: owner decision 15 (TMA in scope); CP3; `casestudies/hopper.md`.
- 4: `language/spec-draft/overview.md:11`; audit R14. 5: CP6; audit RA9. 6: `casestudies/gaps.md` rank 10; CP5.
Confidence: medium.
Would change my mind: a TMA or Trainium2 descriptor that is written by a kernel at run time and read by a later instruction, with no attribute form.

### Q2.6

Position: Option B: concurrency is out of v3, and the spec states that program order is the only order and that a constraint-as-wait faults.
1. S1 already excludes timing and overlap in time; a core grid or instance construct (Option C) has no proposed syntax and no semantics.
2. Under owner 16 a constraint used as a wait has a defined result only under one stated order, and that result is a fault.
3. The L24 idioms (core index as an addressed dimension, async copy as two instructions sharing state) describe the data effect.
4. Declaring it permanently out (Option D) prejudges the R13 study; v3 needs only the present boundary.
Evidence:
- 1: `language/spec-draft/overview.md:11`; R13. 2: CP14; owner decision 16.
- 3: extensions.md L24. 4: R13; owner decision 15 (only SIMT is excluded).
Confidence: high.
Would change my mind: an in-scope instruction whose data result, not its timing, depends on another engine's progress.

### Q2.7

Position: Options C, D, and E: correct the verdicts and recount, count only sketches that run, and add the GPU tensor-core description; no Ascend study now.
1. Three verdicts inflate the matrix (CP2, CP3, CP12); fixing them is cheaper and more accurate than a new study.
2. A sketch that does not parse carried shape bugs, so it is not evidence until it runs.
3. Owner 15 names mma, wgmma, and TMA, and R47 already plans that description; it serves as test and evidence at once.
4. R15's own reason argues against Ascend; NVDLA joins only if an admission decision turns on the descriptor-driven class.
Evidence:
- 1: CP2, CP3, CP12. 2: CP15; R16.
- 3: owner decision 15; R47. 4: audit R15; CP7.
Confidence: medium.
Would change my mind: a construct whose admission is one site short and whose second site is only in NVDLA or Ascend.

### Q2.8

Position: Option A: remove all three; Option C reduces to A because each needs a form no other word uses.
1. `BOOL` appears only in `cholesky` and `triangular_solve`, `TRANSPOSES` only in `triangular_solve`, and `NUMBER` only in `batch_norm_inference`.
2. Removing the words removes three attribute forms, their checker branches, and the oracle's `f16` partial row.
3. No case-study accelerator has such an instruction; each word has one test caller.
4. Option D keeps text no generator supports, which owner 16 then has to define.
Evidence:
- 1, 2: `taidl/taidl/operations.py:18-23`, `:145-156`, `:253-256`.
- 3: minimalist verdict row; R60. 4: `act-oracle/docs/coverage.md`.
Confidence: high.
Would change my mind: a real ISA description in scope with a triangular solve or batch-norm instruction.

### Q2.9

Position: cut presumed reshape, make `d0` an ordinary buffer, keep only `constraints=`, keep `ROOT` fixed on the last line, and require semantics exactly when an output region exists.
1. Presumed reshape is an implicit rule with partial oracle support, and users already write explicit `reshape` lines.
2. `d0` is declared with no dimensions and then special-cased to one; one addressed dimension in its declaration removes the case, and its size is an issue-time parameter (RA10).
3. `add_constraints` and `constraints=` are two spellings of one list.
4. `ROOT` carries no information but keeps a semantics pasteable as HLO text; allowing it elsewhere (R9) adds a rule, so it stays where S7.1 puts it.
5. A semantics defines at least `%Out0`, so R11 and the all-or-none rule coincide as: semantics present if and only if an output region exists.
Evidence:
- 1: `language/spec-draft/semantics.md:48`; minimalist presumed-reshape row.
- 2: `taidl/taidl/accelerator.py:31`; `taidl/taidl/instruction.py:123`; audit RA10.
- 3: `taidl/taidl/accelerator.py:66`. 4: minimalist `ROOT` row; R9.
- 5: panel agenda "For the authors" S7.1 note; R11; MLIR removes a needless distinction to simplify the IR (https://mlir.llvm.org/docs/Rationale/Rationale/).
Confidence: high on 1 to 3; medium on 4 and 5.
Would change my mind: a description that needs a value reshaped without a line, or a generator that emits HLO with `ROOT` elsewhere.

### Q2.10

Position: Option A for the grammar, with any rule listing generated from the `.g4` files; Option C for reserved names, with the error text naming the reserved word and the generator.
1. A copy kept equal by a CI test is a second source; it already drifted once.
2. Sail's manual prints a grammar generated from the parser definition, so a rendered listing need not be a hand copy.
3. The token tables and binding notes are not in the `.g4` files in readable form, so they stay.
4. Reserved names belong to one generator, not the language; a second generator would add a second list.
5. D17's concern is that a user meets the error first; the fix is in the error message, which reaches that user.
Evidence:
- 1: D9 (`language/plan/decisions.md:17`); `taidl/antlr4/IDLV2Expr.g4:25`.
- 2: https://alasdair.github.io/manual.html (grammar "automatically generated from the Menhir parser definition").
- 3: minimalist S2/S3 row. 4, 5: D17 (`decisions.md:25`); `act-oracle/docs/coverage.md`.
Confidence: high on the grammar; medium on reserved names.
Would change my mind: front-end rejection of reserved names, which would make them language.

### Q2.11

Position: the execution page stays normative; the object model page stays but fixes only the names a generator reads (Option C), checked by L29; the rest moves to the `taidl` README.
1. The issue model is semantics: owner 16 defines faults per issue, so the spec cannot drop it.
2. A generator needs a contract; a contract wider than what generators read freezes front-end internals for no reader.
3. The parsed-tree types follow from the `.g4` rules, so naming the entry rule per place fixes them.
4. A serialized format (Option D) adds a second representation with no non-Python consumer.
Evidence:
- 1: `language/spec-draft/execution.md:6-12`; owner decision 16.
- 2: `language/spec-draft/object_model.md:4`; minimalist row "S6 Parsed trees"; L29 (`language/plan/spec.md:122`).
- 3: `taidl/taidl/instruction.py:386-393`. 4: R42.
Confidence: medium.
Would change my mind: a generator that needs a front-end attribute outside that list.

### Q2.12

Position: "ISA description" for the file; in v3 rename only names the S4.6 contract publishes that clash with spec terms, delete `num_dim_str`, and move `array_dim_str` into the oracle.
1. Owner 14 puts the papers first; the ACT paper uses "ISA description(s)" 53 times, the TAIDL paper "ISA definition" a few times, so the newer, dominant term wins.
2. Q2.11 fixes the contract names; a published name that clashes with a spec term would need a mapping table forever.
3. v3 is the only breaking window, and L17 regenerates the parsers anyway, so the rule rename costs only visitor updates.
4. `num_dim_str` has no reader; `array_dim_str` has one, in the oracle, so it belongs there.
5. Renames pass through owner 19: no `add_tensor_buffer` or `add_control_register`.
Evidence:
- 1: https://act-compiler.github.io/assets/pdf/2026-oopsla-act.pdf (term counts from its text); https://act-compiler.github.io/assets/pdf/2025-micro-taidl.pdf ("given ISA definitions in TAIDL").
- 2: `language/plan/review.md:30-31`; `language/spec-draft/object_model.md`.
- 3: owner decision 10; extensions.md L17 changes `TYPE` in `IDLV2.g4`; D13.
- 4: `act-oracle/taidl_to/api_generator.py:25`; minimalist L15 row. 5: audit R2; owner decision 19.
Confidence: high on the file name; medium on the rename set.
Would change my mind: migration cost in B and O counted above one engineer-week for the renames beyond `parameters`.
