# Panel 4, round 1, Fable (outside compiler architect)

Lens: how production and research selectors (TableGen DAGISel and GlobalISel, Cranelift ISLE, egg and egglog, MLIR ODS) solve the same problems act-backend faces.
Assumption throughout: the default cost stays an instruction count unless a cost library is loaded, and decision 11 caps the backend at two engineers.

### Q4.1

Position: Option B; overlap is the normal case for a selector, and the deterministic mode supplies the tie order.

1. Every production selector accepts overlapping patterns and resolves them by cost or priority, never by rejection: DAGISel sorts by pattern complexity, ISLE by explicit priority then specificity.
2. In an e-graph selector, overlap is the point: both rules add e-nodes to one class and the extractor picks by cost; rejecting overlap rejects any ISA with in-place and out-of-place variants.
3. The fixture `in_place()` builds exactly that pair, so R22 contradicts the backend's own tests.
4. The real defect is the tie-break by e-node order, which is iteration order, not a declared order; a stated total order fixes it without new TAIDL syntax.
5. Declaration order is what ISLE and DAGISel users already reason about, so it is the least surprising order; instruction name (Option C) ties the result to renames.
6. Option A would need a priority syntax TAIDL lacks; Option D removes only duplicates, which are harmless to an e-graph anyway.

Evidence: ISLE language reference, rule priorities and the warning that the specificity heuristic is not part of the semantics; LLVM WritingAnLLVMBackend, generated selector comment "Pattern complexity = 13 cost = 1"; act-backend/tests/test_extraction.py:405-414; act-backend/act_backend/generic/src/isel/extractor/fast.rs:24-26 and 50-54; backend/plan/search.md B29.
Confidence: high.
Would change my mind: an owner decision adding a priority syntax to TAIDL, or a measured case where declaration order produces a worse kernel than name order on the examples.

### Q4.2

Position: Option B, with the oracle-differential run scoped to the extraction and matching tests, and the deterministic mode as the only CI mode.

1. A wall-clock stop makes the chosen candidate machine-dependent, so a golden on the simulated output cannot prove the selection unchanged; the plan already says this.
2. LLVM gates structural changes on identical `llc` output under lit, a textual comparison of chosen assembly, which is Option A's content.
3. A textual gate misses semantic errors that happen to change the text legitimately (B10 reorders candidates), so a semantics gate is needed too; translation validation (Alive2 for LLVM) is the research answer, and here the oracle run is the cheap equivalent.
4. Option D alone drops the only cheap signal that the selection did not silently change; Option C leaves B10's reordering ungated for the other steps.
5. Option E keeps two modes that must agree, which doubles the surface a two-engineer team must debug; make deterministic the CI mode and wall-clock the user default only.
6. The cost of Option B over A is one more pytest entry over existing fixtures, which fits decision 11.

Evidence: act-backend/act_backend/generic/src/pipeline.rs:153-180 and 540-548; act-backend/act_backend/generic/src/lib.rs:17-24; act-backend/README.md:109; LLVM lit and FileCheck testing guide; Alive2 translation validation paper (Lopes et al., PLDI 2021); backend/plan/search.md B29; backend/plan/architecture.md B3.
Confidence: medium.
Would change my mind: a measurement that the oracle run over the two test files exceeds a few minutes in CI, or that the deterministic mode changes the chosen kernel on an example.

### Q4.3

Position: Option A with Option F's `Option` shape, and constants in class data per Option D; attributes as child e-nodes (Option C) only where patterns must match them.

1. `Language::matches` compares discriminants only, so payload strings never participate in matching and every condition re-parses them; typing them is a precondition for correct conditions and costs nothing user-facing.
2. egg's own `Analysis` design computes data in `make` from children; shapes set per rewrite in `set_shapes_N` is the pattern egg exists to replace, so B7 before B6 avoids generating code B7 deletes.
3. egg's constant folding idiom is `modify` adding a node; but here the folded node is only needed by candidate building, so class data plus emission at build time (Option D) avoids `NODE_LIMIT` growth and removes the clone.
4. The empty shape as "unset" conflates rank-0 with unknown; `Option<Vec<i64>>` is how GlobalISel's LLT distinguishes untyped from scalar.
5. Option C is right in principle (ISLE and egglog match attributes structurally) but turns every op into several e-nodes; do it only for attributes patterns need, such as slice offsets, after B31 lands.

Evidence: act-backend/act_backend/generic/src/ir/egraph.rs:63-94, 163-165, 396-416; docs.rs egg `Analysis` page, `make`, `merge`, `modify` and the constant folding example; act-backend/act_backend/generic/src/pipeline.rs:232-240; act-backend/act_backend/generic/src/isel/extractor/slow.rs:21-24; backend/plan/isel.md B31, B7, B9, B6.
Confidence: high on order, medium on Option C's scope.
Would change my mind: a count showing that typed payloads force a rewrite of every pattern string, making B6 first cheaper.

### Q4.4

Position: Option B, generated Rust with variables bound by name, for both rule kinds where practical; IR-to-IR rules stay hand-written until B8 names them (Option C as the landing state).

1. Every mature selector compiles rules to code: TableGen emits `XXXGenDAGISel.inc`, ISLE emits one Rust function per term with a decision trie; none interprets rule data at compile time.
2. Generated code gets the host type checker and clippy for free; a data applier needs its own expression-tree evaluator, type checker, and error messages, which is a second language inside the backend.
3. The current applier already re-matches the left-hand side to recover bindings egg does not expose; a generated rule binds names from the `Subst` directly and deletes `find_matches`.
4. A data form (R19) is justified only if rules change at run time or come from users; here rules come from the generator at build time, so the data form buys nothing.
5. Hand-indexed `precond_N` arms are the real defect; name binding fixes it in either form, so the choice should follow tooling, which favors code.

Evidence: ISLE language reference, "Every term with rules in ISLE becomes a single Rust function" and decision trie; LLVM WritingAnLLVMBackend, TableGen patterns to `XXXGenDAGISel.inc`; act-backend/act_backend/generic/src/isel/rewrites/applier.rs:15-58 and 106-113; act-backend/tests/lint.py:19-27; backend/plan/isel.md B6.
Confidence: medium.
Would change my mind: a requirement that a user supply rules without regenerating the crate, or a prototype showing the data applier under 300 lines with typed errors.

### Q4.5

Position: Option B plus E; one k-best extractor ranked through the cost ABI, with deduplication keyed on every node's buffer.

1. egg's `Extractor` is greedy one-best per class; neither extractor here uses it, and the fast one is cost-blind, so the pair gives no cost guarantee today.
2. Two strategies in one loop mean two memo layouts and two drop-reason paths; one k-best enumeration on (class, buffer) is what LpExtractor-style and ILP selectors do, and k covers allocator drops.
3. Ranking by node count is right only under `count.cc`; a loaded cost library can disagree, so rank by the ABI where it is cheap, or document the re-rank.
4. Re-extracting an unchanged e-graph after saturation and recomputing identical candidates wastes the time budget; content deduplication (B34) fixes both.
5. Option C keeps both because cost is known after allocation; but allocation feasibility is a filter, not a ranking, so k-best then filter is the standard shape.

Evidence: docs.rs egg `Extractor` page, greedy cheapest representative per e-class; act-backend/act_backend/generic/src/isel/extractor/fast.rs:24-60; act-backend/act_backend/generic/cpp/cost/count.cc:1-22; act-backend/act_backend/generic/src/pipeline.rs:540-548; backend/plan/extractor.md B10; backend/plan/search.md B34; reviews/rulings/audit.md row R20.
Confidence: medium.
Would change my mind: a measurement that the slow extractor finds a cheaper allocating candidate than k-best with k up to 8 on an example.

### Q4.6

Position: Option B; equality constraint and copy insertion at candidate building land as one item gated on the shared-overwritten-input oracle test.

1. Decision 17 is an equality, and the solver already models offsets, so an equality constraint is the direct encoding; skipping an interference edge only permits overlap.
2. Register coalescing in LLVM joins a tied operand and its result into one live range and inserts a copy where the source is still live; that is exactly B30 plus B11.
3. Shipping the equality before the copy (Options A and E) leaves a window where a live marked input is overwritten and only the oracle notices, which violates decision 16's defined-behavior norm.
4. Option C turns a constraint into a drop reason, losing candidates the solver could have placed; Option D moves allocation into the extractor, duplicating the solver.
5. The copy's cost must be visible to the extractor, so it belongs in candidate building, not in a post-pass fix-up.

Evidence: act-backend/act_backend/generic/cpp/malloc/src/constraint.cc:214-230; act-backend/act_backend/generic/src/isel/extractor/inplace_fix.rs:9-18; LLVM CodeGenerator docs, register coalescing and tied operands; backend/plan/extractor.md B30, B11; act-backend/tests/test_extraction.py:507.
Confidence: high.
Would change my mind: a sizing showing B11 exceeds one quarter, in which case Option E with the drop reason is the fallback.

### Q4.7

Position: Option B; compile-time folding of read-only `@s.` in v3, updates and state-only emission after v3, with every unsupported use a named rejection.

1. Compile-time state is how ArmSME and AMX are modeled in MLIR: the dialect assumes streaming mode or tile configuration is set by an enclosing op, and the lowering emits the configuration instruction; it does not track state as a run-time value.
2. Read-only folding needs only the front end's typed model and the expression writer; updates need the ruling on read order (R7, contested) and state width (RA8, unruled), so they cannot be sized yet.
3. Option D makes state an input to allocation and selection, which is a new dimension of the search; two engineers cannot absorb it.
4. Option C keeps the deficiency the owner named, so it fails the owner's bar; Option A promises updates on rulings that do not exist.
5. The `@x.` writer path would compile `@s.n` as attribute `n` (D12), so the rejection must stay loud until each use is supported.

Evidence: MLIR ArmSME dialect docs, streaming mode as an enclosing-function property; act-backend/act_backend/spec_checks.py:249-251; act-backend/act_backend/generator.py:119-122; backend/plan/coverage.md B16; backend/plan/bugs.md D12; reviews/rulings/audit.md rows R7, RA8; plans/casestudies/sme.md.
Confidence: medium.
Would change my mind: panel 1 ruling R7 and RA8 in time for Q2, which makes Option A feasible.

### Q4.8

Position: Option D as the end state, Option B as the contract now: one generated rule table, and the backend asserts typed input rather than re-checking.

1. MLIR ODS encodes an op's facts once and generates verifiers, builders, and docs from the record; two hand-written copies of shape rules in Python and Rust is the anti-pattern ODS exists to end.
2. The backend never reads `var_type`, so today an ill-typed semantics is silently never selected, which is undefined behavior under decision 16.
3. The front end owns meaning (R25); the backend's obligation is to fail loudly on untyped input, an assertion, not a parallel checker.
4. The B7 `make` rules and the front end's shape rules are the same function over the same table; generating both from one table costs one generator, less than two hand copies plus their drift tests.
5. Option E would ship a rejection the front end later duplicates, two breaking changes for one error.

Evidence: MLIR Defining Dialects, Operations, "encode all facts regarding an operation into the record" and generated `verifyInvariants`; act-backend/act_backend/spec_checks.py:394-418; backend/plan/bugs.md D2, D3; backend/plan/defects.md B21-B23; reviews/rulings/audit.md row R62.
Confidence: medium.
Would change my mind: the language track declining to own a generated table, in which case Option C with the B7 table is the fallback.

### Q4.9

Position: Option B for bounds, Option C's reject branch for unwritten storage, and i64 per intermediate (H4) in both backend and oracle.

1. "May assume it never happens" is LLVM's `poison` wording; decision 16 bans it, so the backend's obligation must be stated as what it never emits and what the allocator enforces.
2. The allocator already models every region offset and size, so enforcing the bound is a constraint it has, not new machinery.
3. Zeroing on-chip buffers costs instructions in every kernel; a read-before-write check at allocation is a liveness query the solver's data already answers, so reject is the cheaper defined behavior.
4. The backend checks every intermediate with `checked_add` in i64; exact arithmetic (R56) would need i128 or big integers in generated Rust and a matching oracle, two implementations of one number system.
5. One norm in both executors (H4) is the only way the differential gate of Q4.2 can compare them.

Evidence: LLVM LangRef, poison values and undefined behavior; act-backend/act_backend/expression_writer.py:70-75 and 92-96; act-backend/act_backend/generic/cpp/malloc/src/constraint.cc:214-230; plans/semantics/holes.md H4, H11; reviews/rulings/audit.md rows R4, R10, R12, R56.
Confidence: medium.
Would change my mind: panel 1 ruling exact arithmetic binding, in which case the backend documents i64-checked and the difference.

### Q4.10

Position: Options B and E together; keep egg, two `Runner` phases with a stated stop reason, and target constants out of the generic crate.

1. egg's `Runner` has iteration, node, and time limits and a `StopReason`; it has no rulesets, so phases are separate `run` calls, which is what B32 proposes.
2. egglog has rulesets and schedules (`seq`, `saturate`, `run(ruleset)`) natively, but adopting it is a core rewrite that decision 11 excludes.
3. Four bare-variable rules fire on every class, so saturation always ends at `NODE_LIMIT`; `BackoffScheduler` only delays them, phases remove them from the selection phase.
4. Rule 0 hard-codes tile 16 and `Dtype::I32` in the generic crate, which is a target constant in generic code; TableGen and ISLE put such constants in the target description.
5. `N` equal to 1 runs extraction after every iteration, which phases make unnecessary: extract once per phase end.

Evidence: docs.rs egg `Runner` page, limits and `StopReason`, `BackoffScheduler`; egglog Python high-level reference, rulesets and schedules; act-backend/act_backend/generic/src/isel/rewrites/ir2ir_rewrites.txt:3-9; act-backend/act_backend/generic/src/isel/rewrites/ir2ir_rewrites.rs:15-30; act-backend/act_backend/generic/src/lib.rs:17-24; backend/plan/search.md B32.
Confidence: high.
Would change my mind: a cost rise on an example under phased runs that interleaving avoids.

### Q4.11

Position: Option B; one typed `#[repr(C)]` struct emitted from the B13 model for both Rust and C++, candidate text kept only for `--log`.

1. Printing text in Rust and re-parsing it in C++ one call later is a serialization boundary with no consumer but the parser; MLIR and LLVM cross such boundaries with typed structs, text only for debugging.
2. `hold` and `hold_tensors` are allocator logic and stay; the parser's `assert(false)` paths become unreachable once the struct is typed.
3. One schema emitted for two languages is what cbindgen and ODS do; two hand-written structs drift.
4. Building B13 from the front end's typed model (Option D) blocks the backend on R42; build from `instruction.py` first (Option E) and re-point, since the struct schema is the stable seam.
5. Option C hardens a boundary that should not exist.

Evidence: act-backend/act_backend/generic/cpp/malloc/src/act_malloc.cc:45-66; act-backend/act_backend/generic/cpp/malloc/src/parser.cc:146, 335; cbindgen documentation, one header from Rust types; backend/plan/parser.md B4; backend/plan/generators.md B13.
Confidence: medium.
Would change my mind: the language track landing R42 before B13 starts.

### Q4.12

Position: Option A, applied under Option E only to emitters that survive Q4.4.

1. Generated code is not committed, so the engineering rules do not require it formatted; the lint gate (clippy with warnings denied) checks meaning, not layout.
2. TableGen and ISLE emit unformatted code and nobody formats `.inc` files; running rustfmt at generation adds a tool to the user's generation path for no reader.
3. Five twins per fragment is width logic in Python that re-implements rustfmt badly; one template per fragment deletes it.
4. If a reader needs the generated crate formatted, `cargo fmt` on the output is one command they already have.

Evidence: act-backend/act_backend/template_loader.py:10-20; act-backend/tests/lint.py:19-27; act-backend/README.md:110; LLVM TableGen BackEnds page, generated `.inc` selectors; backend/plan/generators.md B12.
Confidence: high.
Would change my mind: a stated owner preference to read generated crates in review.
