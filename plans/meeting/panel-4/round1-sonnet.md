# Panel 4, round 1: Sonnet (incremental maintainer)

Lens: every step must leave act-backend generating, building, and compiling the examples.
Assumptions used throughout:
- A1: two engineers, two quarters (owner decision 11), so a step that cannot be verified by an existing test is expensive.
- A2: tests/conftest.py already simulates assembly against the oracle (`simulated`, tests/conftest.py:197), so the oracle-differential run reuses existing machinery.
- A3: external sources cited: egg docs.rs/egg (Analysis, BackoffScheduler), Cranelift ISLE language-reference.md (bytecodealliance/wasmtime, cranelift/isle/docs).

### Q4.1
Position: Option B, plus the exact-duplicate rejection of Option D as a generation-time lint.
1. The fixture `in_place()` declares `add_in_place` and `add_d1` on one sum on purpose, so rejecting overlap breaks a test the project owns (tests/test_extraction.py:405-414).
2. Rejecting overlap also rejects any ISA with in-place and out-of-place variants of one operation, which decision 15 hardware has (audit row R22).
3. Cranelift ISLE allows overlap: rules are an unordered list, higher explicit priority wins, and equal priority is allowed (wasmtime cranelift/isle/docs/language-reference.md).
4. ISLE calls its specificity heuristic "not part of the specified language semantics"; so the order must be stated by us, not left to e-node order.
5. Today the tie-break is e-node order, and the fixture comment says "Declared first, so its e-class holds it first" (tests/test_extraction.py:407); it works by insertion accident, not by contract.
6. Decision 16 needs a defined selection: make the key (cost, instruction declaration index), applied in the extractor, in the deterministic mode of Q4.2.
7. Declaration order keeps today's observed choice on the fixture, so no golden moves.
8. Reject only the exact-duplicate case (same pattern, conditions, result): it is dead code with no cost to detect, and Option D agrees with B there.
Evidence: tests/test_extraction.py:405-414; act_backend/generic/src/isel/extractor/fast.rs:24-26 ("e-node order keeps the numbering stable"); audit row R22.
Confidence: medium.
Change my mind: a spec where two overlapping rules differ in result shape such that declaration order silently picks a wrong-cost one with no way to override; then explicit priority is needed.

### Q4.2
Position: Option B (A for structural steps, plus the oracle-differential run for every behavioral step), with the one-time mode-equivalence check of Option E.
1. Search stops on wall clock (`TIME_LIMIT` 1000 ms, act_backend/generic/src/lib.rs:18) and on `SATURATION_FACTOR` x elapsed (pipeline.rs:157-170), so the chosen candidate depends on machine speed.
2. A gate that is flaky is worse than none for a maintainer; the deterministic mode must exist before any step (B29 first).
3. Equal goldens are simulated outputs and do not prove equal selection (README.md:109 per agenda), so compare chosen assembly and `--log` candidate files too.
4. The oracle-differential run costs little: conftest already generates oracle and backend together (tests/conftest.py:56) and simulates (tests/conftest.py:197).
5. Option C's narrower use (B7, B9, B11 only) misses B6, B10, B32, B30, which also change selection; behavioral steps all change what the extractor can pick.
6. Option D replaces a cheap equality check with per-kernel translation validation, which is a new subsystem; it also cannot say a structural step is unchanged.
7. Option E's caveat is real: run default mode and deterministic mode on every example once; if they differ, re-record goldens once in a commit of its own, before any other step.
Evidence: lib.rs:18; pipeline.rs:155-180; tests/conftest.py:56,197; plans/backend/plan/architecture.md B3; plans/backend/plan/search.md B29.
Confidence: high.
Change my mind: measuring that the oracle run exceeds a few minutes in CI, which would make every-step gating too slow for two engineers.

### Q4.3
Position: Option A order, with D (folded value in class data, no node) and F (`Option` shape).
1. B6 first regenerates `set_shapes_N` that B7 deletes (applier.rs:43-56 shows the hand-indexed `set_shapes_fn`); work done twice.
2. Payloads are strings re-split at each condition (`payload_ints`, egraph.rs:63-94), and `matches` compares discriminants only (egraph.rs:163-165), so typed payloads (B31) are internal to `--log` and carry the least risk; it goes first.
3. egg's `Analysis::make` is the documented place to derive class data from an e-node; here it computes no shape (egraph.rs:396-401), so shape is set from outside by every rewrite.
4. `merge` panics only when both shapes are non-empty and differ (egraph.rs:405-415); the empty vector means "unset", so rank-0 cannot be told from unset.
5. `Option<Shape>` with a `merge` that asserts agreement removes that ambiguity in one small change; an explicit rank-0 flag keeps the bug-prone convention.
6. The clone per extraction exists because extractors insert `DetectedConst` nodes (pipeline.rs:234-240, slow.rs:21-24); keeping constants in class data and emitting the node at candidate building removes the mutation, so the clone goes.
7. Option E (add the node in `modify`) grows the e-graph against `NODE_LIMIT` 10000 (lib.rs:20), which already ends saturation on every kernel.
8. Option C (attributes as child nodes) changes the language arity and every rewrite string: not a small step.
Evidence: egraph.rs:63-94,163-165,396-415; pipeline.rs:234-240; slow.rs:21-24; applier.rs:43-56; egg docs Analysis ("Makes a new Analysis data for a given e-node").
Confidence: medium.
Change my mind: finding that the constant fold needs a node in the class (for example the allocator needs constants as nodes), which would favor E.

### Q4.4
Position: Option D (IR-to-IR hand-written named Rust; IR-to-ISA data with one applier), decided after a two-rule spike; if the spike fails, fall back to Option C.
1. Today three hand-indexed match tables (`precond_N`, `metadata_N`, `set_shapes_N`) bind rule number to function (applier.rs:15-58); this is the defect to remove, not generated-vs-data.
2. Conditions index pattern positions (ir2ir_rewrites.rs:78-85,103-135), so a name-bound variable is needed in either form.
3. The custom applier re-matches the left-hand side to recover e-nodes egg does not bind (applier.rs:106-113,168-215); both forms need that logic once, and a data form has one applier to fix.
4. IR-to-IR rules are few (10 lines, ir2ir_rewrites.txt) and hand-written, one of them hard-codes tile 16 and I32 (ir2ir_rewrites.rs:20-27,100); they gain nothing from a rule data format.
5. ISLE and TableGen compile rules to code because the target is a general compiler with thousands of rules; here the ISA rule count is per accelerator and small.
6. A generic applier shrinks the generator to emitting data tables, and one applier is testable on its own; each emitted function is a new template with width twins (Q4.12).
7. The critic's drop of the experiment is sound only if the generated form is cheaper; no evidence in the plan measures that.
8. A two-rule spike on `add_in_place` and one dot rule costs days and is verified by the Q4.2 gate.
Evidence: applier.rs:15-58,106-113; ir2ir_rewrites.rs:20-27,78-135; plans/reviews/backend/critic.md (Q2 experiment row); wasmtime ISLE docs (rules compiled to Rust).
Confidence: low.
Change my mind: the spike showing the condition expression trees need more than integer arithmetic and comparison over attributes, which pushes toward generated Rust.

### Q4.5
Position: Option C now (one module, both strategies), then Option A/B after B9 and the Q4.2 gate exist; Option E dedup afterwards.
1. The fast extractor is first-feasible and cost-blind and clones its memo per tried e-node (fast.rs:88-128); the slow one enumerates by cartesian product (slow.rs:117).
2. Cost is known only after allocation: the default cost counts `api.` lines of the assembly (cpp/cost/count.cc:11-17), so a node-count proxy before allocation is an approximation.
3. The cost ABI is pluggable (src/cost/c.rs:7), so a proxy can mislead a non-default model; Option B's re-rank is needed in the merged design.
4. The merge is behavioral: the order of `--log` files changes and some candidates disappear; it must wait for the deterministic gate and B9.
5. Merging the files first is structural: no candidate set changes, so it passes the structural gate.
6. After saturation the loop re-extracts an unchanged e-graph (pipeline.rs:544-547); removing that duplicate pass is a one-line, behavior-preserving win available at once (Option D's piece).
7. k-best must cover candidates the allocator drops, so k is measured on the examples before it is fixed.
Evidence: fast.rs:88-128; slow.rs:117; count.cc; pipeline.rs:502,544-547; plans/backend/plan/extractor.md B10.
Confidence: medium.
Change my mind: measurement showing the node-count proxy picks the same winner as the cost model on all examples and a second accelerator family.

### Q4.6
Position: Option B, with B30 first and a candidate dropped with a reason in between (the plan's Option E sequencing).
1. Decision 17: the output reuses the marked input's storage; today the allocator only skips the interference edge (constraint.cc:214-230), so offsets can differ.
2. An equality constraint in the solver (B30) is one local change in one C++ file and is verified by comparing output and input offsets in the `--log` allocation.
3. Equality alone is unsafe: a marked input with a later reader is overwritten; `fix_inplace` (inplace_fix.rs:9-18) patches that after extraction by building a copy.
4. Keep `fix_inplace` until B11 moves the copy into candidate building; between the two, a candidate whose input is live is dropped with a reason (decision 16: defined result).
5. Option C (verify after allocation, drop on mismatch) leaves aliasing a luck of the solver, contradicting decision 17.
6. Option D (extractor pre-places the region) puts allocation logic into the extractor, spreading one concern over two stages.
7. The two must ship as one item only for the final state; in between, the drop-with-reason keeps the compiler correct.
8. R5 (derive the mark from identical region text) only changes where the mark comes from; the backend reads it either way (ir_egraph_rs_generator.py:94,167-171), so B30 is independent.
Evidence: constraint.cc:214-230; inplace_fix.rs:9-18; tests/test_extraction.py:507; plans/backend/plan/extractor.md B30, B11.
Confidence: medium.
Change my mind: a case where a legal in-place kernel becomes infeasible under the equality because the marked input is pinned in HBM, and no copy is possible.

### Q4.7
Position: Option B (read-only `@s.` in starts, sizes, constraints folded at compile time in v3; updates and state-only emission after), with Option C as the floor.
1. Today every state read in an instruction with semantics is rejected (spec_checks.py:249-251), and an instruction without semantics is silently dropped (generator.py:119-122); the second is a defect under decision 16.
2. A read-only fold is the smallest piece: the writer already folds integer attributes and checks i64 overflow, and `@s.` reads become constants under a known state.
3. Updates need S5.4 (when updates take effect), R7 is contested in the audit, and RA8/RA9 have no ruling; implementing before those are settled would encode a guess.
4. State-only instruction emission needs the allocator and selection to schedule a configuration instruction before its user: new search logic, which fits no small step.
5. Decision 21 and 19 only fix names; they do not require support in v3.
6. Whichever option wins, the silent drop must become a message or a rejection now (Option E first half, R11); that part is a small step and needs no state design.
7. Option D (run-time state tracked by generated code) multiplies allocator and selection inputs; it is the largest and has no case-study demand (plans/casestudies/sme.md uses compile-time configuration).
Evidence: spec_checks.py:249-251,274-292; generator.py:119-122; plans/backend/plan/coverage.md B16; audit rows R7, RA8, RA9.
Confidence: medium.
Change my mind: a case study kernel in scope (SME, TMA descriptors) that cannot compile without state updates, which would force Option A.

### Q4.8
Position: Option A with Option D as the long-run target; the backend keeps its literal check until the front end lands the shared table.
1. `check_regions` only rejects a literal count above the region's (spec_checks.py:394-418) and skips any non-literal (`if None in (tile, region): continue`); it is a safe, tested floor.
2. The backend never reads `var_type` and never checks line agreement (D2, D3), so an ill-typed semantics is silently never selected; decision 16 makes that a defined error.
3. Writing the rule table twice (Rust B7 and Python) invites drift; R62 says derive it once from the oracle shape module.
4. The size dispute (M in R62, L or more in audit) shows nobody has measured it; I cannot lower the L risk for a backend step.
5. Option E (reject now, delete later) is acceptable only for rules whose rejections need no new table: those are cheap and within the v3 batch (decision 10).
6. Option B (the backend asserts the IR is typed) is the cheapest safeguard against silent never-selected rules: one assertion, no table.
7. Order: B asserts typing now, front end owns checks of meaning (R25, R31), backend deletes its literal check in the same commit batch that lands the front-end rule.
Evidence: spec_checks.py:394-418; plans/minutes/rulings.md R25, R31, R62; audit row R62; plans/backend/plan/defects.md B21-B23.
Confidence: medium.
Change my mind: the front end schedule slipping past v3, which would force Option E or C to avoid shipping a silent never-selected case.

### Q4.9
Position: Option B for R4 and R10; Option C for unwritten storage (zero on-chip buffers or reject read-before-write); Option E third variant (exact in the oracle, i64-checked in the backend) for integers, documented.
1. "May assume it never happens" is undefined-behavior wording, and decision 16 forbids undefined behavior; the allocator already holds sizes in 64 bits (spec_checks.py:147,217,253), so enforcement is feasible.
2. Reword: the backend never emits a violating region or constraint, and the allocator rejects an allocation that would (audit rows R4, R10).
3. R12 as drafted binds the backend to emit zeroing or prove no read-before-write; no backend item plans either.
4. The smallest honest backend rule is to reject a kernel that reads unwritten on-chip storage, because zeroing costs real instructions.
5. `d0` holds inputs and is not zero-filled by the backend; the rule applies to on-chip buffers only.
6. R56 exact arithmetic needs i128 or big integers in generated Rust; the writer checks each operator in i64 (expression_writer.py:72,94 `checked_add`, `checked_mul`).
7. Exact in the oracle and i64-checked in the backend differ only when an intermediate leaves i64 and the final value returns inside; the backend then errors where the oracle succeeds, a defined, safe difference to document.
8. i128 in generated Rust is cheap but the C++ allocator holds 64 bits, so it would not remove the limit end to end.
Evidence: expression_writer.py:72,94; spec_checks.py:147,217,253; plans/semantics/holes.md H4, H11; audit rows R4, R10, R12, R56.
Confidence: medium.
Change my mind: a case study with a legal intermediate beyond i64 that the final value brings back inside, which would force i128.

### Q4.10
Position: Option E plus Option B, keeping egg; fix rule 0 first, then phases, with Option C as a stopgap.
1. Four bare-variable rules fire on every class (ir2ir_rewrites.txt:3,4,5,8), so saturation ends at `NODE_LIMIT` rather than at saturation.
2. egg ships `BackoffScheduler`: a rule over its match limit (default 1000) is banned for a number of iterations (default 5), with doubling; it targets exactly this explosive-rule problem (docs.rs/egg BackoffScheduler).
3. Backoff is a one-line runner setting and is a behavioral step gated by Q4.2; it is the stopgap before phases.
4. Phases (B32) with separate `Runner`s need no engine change and keep every rule; they cost one extra loop and need B8 to name the rules.
5. egglog is a rewrite of the core; two lanes cannot absorb it (decision 11), and the plan found no feature needing it.
6. Rule 0 hard-codes tile size 16 and I32 in the generic crate (ir2ir_rewrites.rs:20-27,100), which is a latent bug on any other ISA; generating the constants from the spec, or dropping the rule when no instruction needs it, is a correctness fix.
7. `N` is 1 (lib.rs:24), so extraction runs after every iteration; raising N is a free tuning knob after dedup (B34) and measurement.
Evidence: ir2ir_rewrites.txt:3-8; lib.rs:20,24; ir2ir_rewrites.rs:20-27,100; docs.rs/egg BackoffScheduler; plans/backend/plan/search.md B32.
Confidence: medium.
Change my mind: measurement showing phase order loses a candidate that interleaved firing finds on the examples (the B32 caveat).

### Q4.11
Position: Option A, with Option E for the model source; Option F redefined (reader checks annotations) after B7.
1. The C++ parser reads text Rust printed one call earlier (act_malloc.cc:45-66), and a bad line drops the candidate with a message, so the round trip is lossy and a source of drift.
2. A typed `#[repr(C)]` call removes the text parse while `hold` and `hold_tensors` stay in the allocator (parser.cc:146,169), where they are allocator logic.
3. The Rust printer stays for `--log` in today's exact format, because tests read those files (test_extraction.py:109,135,496).
4. The struct from the B13 model (Option B) needs the model first; B13 builds from today's `instruction.py` objects (Option E) and re-points when R42 lands, since R42 says a typed Python object model.
5. ABI mismatch is the fail point; a round-trip test that prints the struct from C++ detects it.
6. Option C (harden the text) is a safe fallback and cheap (B25: `assert(false)` to error returns at parser.cc:14,188), and it should land anyway.
7. The boundary is internal (decision 10), so no user-facing break.
Evidence: act_malloc.cc:45-66; parser.cc:146,169; pipeline.rs:249-253,457-469; tests/test_extraction.py:109,135,496; plans/backend/plan/architecture.md B2.
Confidence: medium.
Change my mind: the ABI struct needing fields that vary per accelerator, making one schema unstable.

### Q4.12
Position: Option E: decide after Q4.4; absent that, Option B (delete twins, run rustfmt at generation, no clang-format).
1. 79 templates, many in layout twins chosen by width (`TWINS`, template_loader.py:14), exist only to pass rustfmt and clang-format.
2. No gate checks generated layout (tests/lint.py:19,27), so the twin logic has no consumer other than the formatter lint on committed output.
3. If Q4.4 picks a data applier, the number of emitted fragments drops, and the twin question shrinks with it; deciding twice is wasted effort.
4. rustfmt is already a project dependency and the generated crate is Rust, so running it at generation removes the width logic with no new tool.
5. clang-format is not needed: generated C++ is not committed and CI checks `git ls-files` only (README.md:110).
6. Twin deletion is structural: the Q4.2 gate (identical assembly) proves it; only layout differs.
7. Deleting twins removes `fits`/`LINE_WIDTH` logic in template_loader.py:1-20 and the width checks in `CheckedWriter` (expression_writer.py:76,90).
Evidence: template_loader.py:1-20; tests/lint.py:19,27; plans/backend/plan/generators.md B12; plans/minutes/rulings.md R24.
Confidence: medium.
Change my mind: a requirement that generated sources be committed and diff-reviewed, which would need stable layout without a formatter dependency.
