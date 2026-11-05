# Panel 4 agenda: backend architecture

Scope: act-backend parsers, e-graph, rewrites, extraction, allocation, and state register support.
Lenses: incremental maintainer (Sonnet), e-graph and instruction selection expert (Opus), outside compiler architect (Fable).
Moderator: Fable.
Questions Q4.1 to Q4.8 are blockers or majors; Q4.9 to Q4.12 are majors or smaller design choices.
Options are listed in no order of preference; a draft answer is one option among others.
Only the owner decisions bind the panel.

## Owner decisions that bound this panel

- Decision 10: breaking changes are allowed, batched into one coordinated major release.
- Decision 11: size for 3-4 engineers over two quarters; the backend plan is sized for two of them.
- Decision 15: tensor accelerators plus GPU tensor-core instructions; no SIMT thread model.
- Decision 16: no undefined behavior in v3; every run-time violation is a defined error or a defined behavior under one simple norm.
- Decision 17: the output of an in-place mark reuses the marked input's storage.
- Decision 19: a data model has buffers (not "tensor buffers") and state (not "initial state"); `hbm` stays `hbm`.
- Decision 21: the `@s.` prefix stays.

## Questions

### Q4.1: How are overlapping IR-to-ISA rewrite rules handled?

- Context: two rules whose left-hand sides match the same expression are alternatives for the extractor.
- Context: the fixture `in_place()` declares `add_in_place` and `add_d1` on the same sum on purpose.
- Context: the current tie-break between equal-cost candidates follows e-node order.
- Context: TAIDL has no syntax to declare a priority.
- Option A: reject overlapping rules at generation; allow explicit priority only where declared (R22).
- Option B: allow overlap; the cost extractor chooses; break cost ties by a stated total order such as declaration order, under the deterministic mode of Q4.2.
- Option C: allow overlap and break ties by a stated order that is not declaration order (for example instruction name or rule id).
- Option D: reject only overlaps that are exact duplicates (same pattern, same conditions, same result) and allow the rest.
- Evidence: minutes/rulings.md R22; reviews/rulings/audit.md row R22 (verdict wrong); backend/review-response/binding.md row R22 (objection); reviews/backend/architect.md (extractor and tie-break findings); backend/plan/extractor.md B10; act-backend/tests/test_extraction.py:405-414; act-backend/act_backend/generic/src/isel/extractor/fast.rs:24-26.
- Binding: none directly; decision 16 needs a defined selection, not a particular one.

### Q4.2: What gate proves a structural step unchanged?

- Context: search is bounded by a wall clock (`pipeline.rs:155-180`, `lib.rs:18`) and stops on a ratio of elapsed times, so the chosen candidate can depend on machine speed.
- Context: goldens are simulated kernel outputs (`README.md:109`), so equal goldens do not prove an equal selection.
- Context: the plan adds a deterministic mode (B29) and an oracle-differential run (B3) as the gates.
- Option A: add the deterministic mode first; gate on identical chosen assembly and identical `--log` candidate files, plus goldens (R17).
- Option B: Option A, plus an oracle-differential run over `tests/test_extraction.py` and `tests/test_matching.py` as the gate for every behavioral step.
- Option C: keep goldens as the gate and add the oracle-differential run only for B7, B9, and B11.
- Option D: derive the gate from translation validation of each compiled kernel against the oracle, with no chosen-assembly comparison.
- Option E: the deterministic mode is a flag on top of the wall-clock default, and the two modes must pick the same candidate on the examples (B29 caveat), or the goldens are re-recorded once.
- Evidence: minutes/rulings.md R17; backend/plan/architecture.md B3; backend/plan/search.md B29; reviews/backend/architect.md plan step 1; reviews/backend/critic.md rows B3 and "Differential testing as the gate"; backend/review-response/architect.md step 1; backend/review-response/critic.md rows B3.
- Binding: decision 11 (gate cost counts against two engineers).

### Q4.3: In what order are the e-graph payload, shape, and constant changes made?

- Context: payloads are strings re-split at each condition by `payload_ints` (`egraph.rs:63-94`), and `Language::matches` compares only the discriminant (`egraph.rs:163-165`).
- Context: `Analysis::make` computes no shape (`egraph.rs:396-401`), so each rewrite sets shapes in `set_shapes_N` and the reader makes 17 `set_analysis_data` calls.
- Context: each extraction pass clones the e-graph (`pipeline.rs:234-240`) because the extractors add `DetectedConst` nodes (`slow.rs:21-24`).
- Context: the empty shape doubles as "unset" (`egraph.rs:405-415`, D8), which makes rank-0 shapes unreliable.
- Option A: typed payloads (B31), then shape and type in `make` (B7), then no clone (B9), then rewrite values (B6), all in Q1 (R18).
- Option B: rewrite values (B6) first, as the original plan ordered, with string payloads kept until later.
- Option C: attributes become child e-nodes so egg's own matching sees them (critic), with typed fields only where an attribute is not a node.
- Option D: constant-ness and the folded value live in class data and candidate building emits the node (B9 as planned).
- Option E: the constant node is added in `Analysis::modify` (critic), accepting growth against `NODE_LIMIT`.
- Option F: shape is `Option` in class data (B26) with `merge` asserting agreement, or shape keeps the empty-vector convention with an explicit rank-0 flag.
- Evidence: minutes/rulings.md R18; backend/plan/isel.md B31, B7, B9, B6; backend/plan/defects.md B26; reviews/backend/architect.md plan steps 4 and 8; reviews/backend/critic.md rows B6, B7, B9 and "Missing: typed payload"; backend/review-response/critic.md rows B6 and B9; backend/coverage-table.md (rank-0 row); reviews/backend/verification.md D8, D11.
- Binding: decision 10 (typed payloads change `--log` internals only; no user-facing break); decision 11.

### Q4.4: Are rewrite conditions generated code or data?

- Context: IR-to-ISA rewrites are text lines bound by number to three hand-indexed Rust functions (`applier.rs:15-58`); conditions index pattern positions (`ir2ir_rewrites.rs:78-85,103-135`).
- Context: the custom applier re-matches the left-hand side to recover e-nodes egg does not bind (`applier.rs:106-113,168-215`).
- Context: the architect recommends data for IR-to-ISA rules; the critic recommends generated Rust and drops the Q2 data-driven experiment.
- Option A: data per instruction (pattern, payload conditions as small expression trees, result attributes) read by one generic applier, decided now (R19).
- Option B: generated Rust, one value per rule with variables bound by name, as TableGen and ISLE compile rules to code (critic).
- Option C: hybrid: IR-to-IR rules are hand-written named Rust; IR-to-ISA rules are generated Rust.
- Option D: hybrid: IR-to-IR rules are hand-written named Rust; IR-to-ISA rules are data with one applier.
- Option E: defer the choice behind a Q2 experiment (the plan before the architect review).
- Evidence: minutes/rulings.md R19; backend/plan/isel.md B6; backend/plan/decisions.md (Rewrite form row); reviews/backend/architect.md plan step 7 and target architecture; reviews/backend/critic.md rows B6 and "Q2 data-driven applier experiment"; backend/review-response/architect.md step 7; backend/review-response/critic.md row on the Q2 experiment.
- Binding: decision 11 (B6, B12, B13 are generator work shared by one lane).

### Q4.5: One extractor or two, and what ranks the candidates?

- Context: the fast extractor is a cost-blind first-feasible DFS that clones its memo per tried e-node (`fast.rs:88-128`).
- Context: the slow extractor enumerates every tree up to a growing node limit (`slow.rs:117`, `pipeline.rs:502`).
- Context: the default cost is the count of `api.` lines (`cpp/cost/count.cc:1-22`), but the cost ABI is pluggable (`src/cost/c.rs:7`).
- Context: after saturation the loop re-extracts an unchanged e-graph (`pipeline.rs:544-547`).
- Option A: merge into one cost-guided k-best extractor on (class, buffer) ranked by a node-count proxy; the real cost model picks among those that allocate (R20, B10).
- Option B: Option A, with candidates ranked through the cost ABI, or stated to be re-ranked when a non-default cost model is loaded (audit).
- Option C: merge the two files into one module and keep both strategies, since cost is known only after allocation (critic).
- Option D: keep both extractors unchanged and only remove the duplicated constant pass.
- Option E: any of A to D, plus deduplication of candidates by content across passes (B34), with the content key including every node's buffer.
- Evidence: minutes/rulings.md R20; backend/plan/extractor.md B10; backend/plan/search.md B34; reviews/rulings/audit.md row R20; reviews/backend/architect.md plan step 6; reviews/backend/critic.md rows B10; backend/review-response/critic.md row B10; act-backend/act_backend/generic/src/isel/extractor/.
- Binding: none directly; decision 16 needs drop reasons to stay defined.

### Q4.6: How is in-place aliasing realized in allocation and extraction?

- Context: the allocator only skips the interference edge between an in-place output and its marked input (`constraint.cc:214-230`); it does not force the same offset.
- Context: the extractor patches liveness after extraction in `fix_inplace` (`inplace_fix.rs:9-18`).
- Context: a marked input with a later reader is overwritten unless a copy is built; the audit pairs equality and copies as one item.
- Context: R5 derives the mark from identical region texts and removes `set_inplace`; that ruling is a language-track matter, but the backend reads the mark (`ir_egraph_rs_generator.py:94,167-171`).
- Option A: an equality constraint on the two offsets in the OR-Tools solver (B30, R21), with the extractor fix-up kept until B11.
- Option B: Option A plus B11 copies at candidate building, delivered as one item and gated on the shared-overwritten-input oracle test.
- Option C: keep overlap permitted and verify aliasing after allocation, dropping a candidate with a reason when offsets differ.
- Option D: handle aliasing only in the extractor, giving the output node the input's allocation as a pre-placed region.
- Option E: B30 in Q1 and B11 in Q2 as the plan sequences them, with a candidate dropped with a reason in between.
- Evidence: minutes/rulings.md R21, R5; reviews/rulings/audit.md rows R21, R5; backend/plan/extractor.md B30, B11; reviews/backend/architect.md plan step 10; reviews/backend/critic.md row B11; backend/review-response/architect.md step 10; backend/review-response/binding.md (decision 17 row, R5/R58 row); act-backend/tests/test_extraction.py:507.
- Binding: decision 17.

### Q4.7: How does the backend support state registers and `@s.*`?

- Context: today every read and update of state in an instruction with semantics is rejected (`spec_checks.py:249-251,274-292,328-367,374-385`); declaring state is accepted.
- Context: an instruction without semantics is dropped unchecked (`generator.py:119-122`).
- Context: B16 treats state as compile-time configuration: a kernel compiles under one known state, state-only instructions are emitted where a selected instruction needs a state value, and `@s.` reads fold to integers.
- Context: R7 (state read order) is contested in the audit, and RA8 and RA9 (state width, data-dependent addressing) have no ruling.
- Option A: implement in v3 Q2 after the front end's typed instruction model, as compile-time configuration (R27, B16).
- Option B: implement in v3 for read-only `@s.` in starts, sizes, and constraints folded at compile time; updates and state-only instruction emission follow after v3.
- Option C: keep the rejection in v3 and document it in the coverage table; implement after v3.
- Option D: state is a run-time value the generated code tracks (allocator and selection treat it as an input), not folded at compile time.
- Option E: for an instruction with regions and no semantics, reject (R11); for a state-only instruction, either emit it (B16) or keep leaving it out with a message (B24 as written).
- Evidence: minutes/rulings.md R27, R11, R14, R7; reviews/rulings/audit.md rows R7, R14 and RA8, RA9; backend/plan/coverage.md B16; backend/plan/defects.md B24; backend/coverage-table.md (state rows); reviews/backend/verification.md (questions table, C32, C33, D4, D12); reviews/backend/critic.md row B16; plans/casestudies/sme.md.
- Binding: decision 21 (`@s.` stays); decision 16 (a dropped or rejected state use must be a defined result, not silent); decision 19 (state vocabulary).

### Q4.8: Which track owns the checks of meaning, and when do the backend's copies go?

- Context: the backend checks region element counts (`check_regions`, `spec_checks.py:394-418`) only for a literal count above the region's.
- Context: the backend never reads `var_type` and never checks shape or type agreement of semantics lines (D2, D3); an ill-typed semantics is silently never selected.
- Context: the shape rules would exist once in Rust (B7) and once in Python unless generated from one table.
- Context: R62 sizes the shared shape module as M; the audit sizes it L or more.
- Option A: region element-count checks move to the front end (L5); the backend keeps its literal check until then (R25).
- Option B: line agreement (B23) moves to the language track, and the backend asserts that the semantics IR is typed (architect).
- Option C: the backend keeps B21 to B23 as its own generation-time rejections, built from the B7 rule table.
- Option D: one rule table is generated for both the front end and the backend, so neither copy is hand-written.
- Option E: the backend rejects now (breaking, in the v3 batch) and deletes its copy when the front end lands the same check.
- Evidence: minutes/rulings.md R25, R62, R31; reviews/rulings/audit.md row R62; backend/plan/defects.md B21, B22, B23; backend/plan/bugs.md D1, D2, D3; backend/review-response/architect.md row on B23; reviews/backend/architect.md plan step 9; reviews/backend/verification.md D1 to D3.
- Binding: decision 16 (an ill-typed semantics is a defined error); decision 10 (rejections are batched into the major release).

### Q4.9: What does the backend assume, enforce, or initialize under the no-undefined-behavior norm?

- Context: R4 and R10 say the backend "may assume" an out-of-bounds region or a violated constraint never happens; the audit calls that undefined-behavior wording.
- Context: R12 says every buffer starts zero-filled; the audit limits it to on-chip buffers and notes no backend item plans zeroing or a read-before-write check.
- Context: R56 makes integer arithmetic exact; the backend checks every intermediate in i64 (`expression_writer.py:72,94`), so exact arithmetic would need i128 or big integers in generated Rust.
- Context: H4 (`plans/semantics/holes.md`) keeps every intermediate in i64 and faults on leaving it.
- Option A: keep R4 and R10 as drafted: the backend may assume the violation never happens.
- Option B: reword: the backend never emits a violating region or constraint, and its allocator enforces the bound (audit).
- Option C: for unwritten storage, the backend zeroes on-chip buffers it exposes, or rejects a kernel that reads unwritten storage.
- Option D: for unwritten storage, the backend does neither and the contract is stated as the oracle's alone.
- Option E: i64 per intermediate (H4) in generated Rust; or exact arithmetic with i128 or big integers (R56); or exact in the oracle and i64-checked in the backend with the difference documented.
- Evidence: minutes/rulings.md R4, R10, R12, R56; reviews/rulings/audit.md rows R4, R10, R12, R56 and RA2, RA3; plans/semantics/holes.md (H1, H2, H4, H11); act-backend/act_backend/expression_writer.py:72,94; backend/plan/coverage.md B15 (S1.3 cases).
- Binding: decision 16.
- Cross-panel: panel 1 owns the semantic rulings; this question covers only the backend's obligation.

### Q4.10: Which engine and rule-scheduling strategy does the e-graph use?

- Context: four IR-to-IR rules have a bare-variable left-hand side and fire on every class (`ir2ir_rewrites.txt:3,4,5,8`); saturation ends at `NODE_LIMIT` 10000 (`lib.rs:20`) or the 1000 ms limit.
- Context: `N` is 1 (`lib.rs:24`), so extraction and allocation run after every iteration.
- Context: rule 0 hard-codes tile size 16 and `Dtype::I32` inside the generic crate (`ir2ir_rewrites.rs:20-27,100`).
- Option A: keep egg; borrow rule sets, explanations, and stop reasons (R23).
- Option B: keep egg; separate `Runner`s per phase, expansion rules then selection rules (B32).
- Option C: keep egg; apply `BackoffScheduler` or make the bare-variable rules conditional, with no phases.
- Option D: move to egglog for rulesets and schedules.
- Option E: for rule 0, generate the tile and dtype from the spec's instructions, or drop the rule where no instruction needs it (B33).
- Evidence: minutes/rulings.md R23; backend/plan/search.md B32, B34; backend/plan/isel.md B33; backend/plan/decisions.md (Rule scheduling, Engine rows); reviews/backend/architect.md (Missing: rule scheduling); reviews/backend/critic.md (Missing: rule sets and phasing, target constants); backend/review-response/binding.md row R23.
- Binding: decision 11 (egglog is a rewrite of the core; two lanes cannot absorb it).

### Q4.11: What is the shape of the parser, FFI, and instruction-model boundaries?

- Context: the C++ `parser.cc` reads back text Rust printed one call earlier (`act_malloc.cc:45-66`); kernel facts cross as `metadata.json` (`pipeline.rs:457-469`).
- Context: `hold` and `hold_tensors` (`parser.cc:146,335`) are allocator logic, not parsing.
- Context: each generator re-walks the TAIDL ANTLR tree (`isel_ir2isa_rewrites_txt_generator.py:56-65`), and `spec_checks.py` imports emitter internals (`spec_checks.py:9-18`).
- Context: the front-end object model is a language-track item (R42); B13 builds from today's `instruction.py` objects until it lands.
- Option A: one typed `#[repr(C)]` call carrying candidate and kernel facts; delete the C++ candidate parser; keep `create_instruction`, `hold`, `hold_tensors`; the Rust printer stays only for `--log` in today's exact line format (B2, B4).
- Option B: Option A, with the struct emitted for Rust and C++ from the B13 model so one schema exists.
- Option C: keep the text boundary and harden it (return errors, not `assert`; B25 only).
- Option D: B13 builds from the front end's typed Python object model (R42), jointly with the language track.
- Option E: B13 builds from today's `instruction.py` objects first and re-points later.
- Option F: B5 as a reader split in three layers (original), or defer it, or redefine it as "reader checks annotations, analysis infers" after B7.
- Evidence: minutes/rulings.md R42; backend/plan/architecture.md B2; backend/plan/parser.md B4, B5; backend/plan/generators.md B13, B14; backend/plan/defects.md B25; reviews/backend/architect.md plan steps 3, 5 and "What the plan would lose"; reviews/backend/critic.md rows B2, B4, B5, B13; backend/review-response/architect.md steps 3, 5 and B5 row; act-backend/tests/test_extraction.py:109,135,496.
- Binding: decision 11; decision 10 (the C++ boundary is internal, not user-facing).

### Q4.12: How is generated Rust and C++ laid out?

- Context: 79 templates exist, many in several layouts chosen by line width (`TWINS`, `template_loader.py:14`) so output passes rustfmt and clang-format.
- Context: no gate checks the generated crate's layout (`tests/lint.py:19,27`); the CI format check covers `git ls-files` only (`README.md:110`).
- Option A: one template per fragment, no width logic, no formatter run at generation (B12, critic).
- Option B: delete the twins and run rustfmt at generation; clang-format is not required (R24).
- Option C: delete the twins and run both rustfmt and clang-format at generation.
- Option D: keep the twins.
- Option E: apply any choice only to emitters that survive the Q4.4 decision.
- Evidence: minutes/rulings.md R24; backend/plan/generators.md B12; reviews/backend/critic.md row B12; reviews/backend/architect.md plan step 7; backend/review-response/critic.md row B12.
- Binding: none directly; the engineering rules require formatting of committed files, not of generated output.

## For the authors

Findings that need no debate; the panel may skip them.

- R26 reason: keep `Buffer`; the reason is owner decision 19, not "the papers' term" (audit row R26, verdict wrong).
- R26: rename `TensorInfo`; `is_const` already lives in it (`egraph.rs:370-374`).
- B28: treat the `--verbose` label renames ("Phase 1 Module 3", "Info:") as non-breaking diagnostics unless the owner says otherwise.
- B27: the `@s.` size and constraint errors name the state, as the start and semantics errors do (`spec_checks.py:287-290,364-367`).
- B25: replace the `assert(false)` paths with error returns through `why` (`parser.cc:14,188`, `constraint.cc:115`); plain `cmake ..` builds lack `NDEBUG` handling (D5).
- B26: `PiiGraph` construction returns `Result` (`ir/pii.rs:36,42,48`, `ir/buffer.rs:15`).
- B17: `concatenate` takes one or more operands (S8.2), not exactly two (`egraph.rs:13`); the coverage table needs a `partial` row.
- B18: sub-byte types need packing rules in the allocator's byte sizes.
- B19: egg patterns have one root; several outputs need a multi-pattern or a tuple op, and the item may slip past Q2.
- B20: float literal range check matches the integer check (`spec_checks.py:97-105`).
- Coverage table: rows for `parameter`, `@a.` in a size, constraint integers past 64 bits, narrow `var_type`, float literal range, rank-0 shape, and S1.3 cases are added (verification review).
- D13: the one-element `broadcast` guard assumes the broadcast operand is the previous assignment (`isel_ir2isa_rewrites_rs_generator.py:348-350`); B6 removes it.
- D9: "refuse" occurs in docstrings, not error messages (`spec_checks.py:1,23,180`).
- R58: tests and spec pages that call `set_apart` and `set_no_partial_overlap` must migrate (`act-backend/tests/test_generation.py:220-221`).
- R9: backend coverage row for `ROOT` on a non-last line stays `partial` until fixed.
- RA1: R34 is missing from the numbering; restore it or mark it retired.
- RA12: no ruling sums the backend Q1 and Q2 items (about 52-56 engineer-weeks) against decision 11; panel 8 owns the roadmap.
