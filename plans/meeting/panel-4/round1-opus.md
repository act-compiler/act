# Panel 4 round 1: Opus (e-graph and instruction selection lens)

Assumptions:
- The e-graph holds alternatives; selection happens only at extraction, then allocation, then the cost model.
- egg stays the engine unless Q4.10 says otherwise; decision 11 bounds the work to two engineers.

### Q4.1
Position: Option B; overlap is allowed, the extractor chooses by cost, and ties break by declaration order recorded as an explicit rule index, not by e-node insertion order.
1. A first-match selector such as ISLE commits to one rule, so equal-priority overlap is ambiguous and ISLE rejects it.
2. ISLE exempts multi-constructors, which return every match, from the overlap check; an e-graph rewrite is that case, since every matching rule adds an alternative.
3. The fixture declares `add_in_place` and `add_d1` on the same sum on purpose, so rejection (A) breaks a deliberate design.
4. Today the tie-break is e-node order in the class, which depends on rewrite application order and union-find history, not on the spec.
5. A stated total order needs a key the spec owns; declaration index is already what the fixture relies on, so store it on the ISA e-node and compare it last.
6. Exact duplicates (D) can be reported as a generation warning, since they add nodes and no choice.
Evidence: 1-2 docs.rs cranelift-isle overlap.rs (equal priority is an overlap error; multi-constructors not checked); 3 tests/test_extraction.py:405-414; 4 isel/extractor/fast.rs:24 ("e-node order keeps the numbering stable"); 5 tests/test_extraction.py:409 ("Declared first, so its e-class holds it first").
Confidence: high.
Would change: a case where cost and declaration order pick a candidate the owner calls wrong, which would argue for declared priority.

### Q4.2
Position: Option B with E's mechanics; the deterministic mode is a flag that tests always use, and goldens are re-recorded once if it picks differently.
1. Search stops on `TIME_LIMIT` and on elapsed time over the best candidate's timestamp times `SATURATION_FACTOR`, so the chosen candidate depends on machine speed.
2. A structural step must leave the selection unchanged; only identical chosen assembly and `--log` files prove that, and goldens of simulated outputs do not.
3. egg's `Runner` already stops on iteration and node limits; a deterministic mode drops the time limit and the wall-clock hook stop, so it is cheap.
4. A behavioral step changes selection by design, so identical assembly cannot gate it; the oracle-differential run on the two test files checks meaning instead.
5. D drops the chosen-assembly comparison, which is the only gate that catches a silent selection change for structural steps.
6. Hidden nondeterminism (hash-map order) is the main risk; two runs on two machines catch it.
Evidence: 1 generic/src/pipeline.rs:154-180, lib.rs:18; 3 pipeline.rs:490-495 and docs.rs egg Runner (stop reasons: saturation, iteration, node, time limits); 6 backend/plan/search.md:18-21.
Confidence: medium.
Would change: the oracle-differential harness costing more than about two engineer-weeks, which would argue for C.

### Q4.3
Position: Option A order, with F's `Option` shape in class data and D for constants.
1. egg's `Language::matches` should compare the operator and arity; a slice's offsets are part of its operator.
2. Here `matches` compares only the discriminant, so payload checks fall to string re-splitting in conditions; typing payloads is the base every later step builds on.
3. `Analysis::make` gets the e-node and its children's data, so shape and type inference belongs there once payloads are typed; this removes the per-rewrite `set_shapes_N`.
4. `merge` panics only when both shapes are non-empty, so a rank-0 shape reads as unset; `Option<Shape>` with an asserting `merge` separates the two cases.
5. Constant-ness is a pure function of children and the fixed input set, so it is class data; the folded value can live there too.
6. egg's constant-folding idiom adds the node in `modify` (E), but a `DetectedConst` node then competes in extraction and counts toward `NODE_LIMIT`; emitting it at candidate building (D) keeps the e-graph unchanged and removes the clone.
7. C (attributes as child e-nodes) lets rewrites union attribute classes and enlarges every match; I reject it.
8. B (rewrite values first) regenerates code that A then deletes.
Evidence: 1 docs.rs egg Language::matches ("only consider the operator and the arity"); 2 ir/egraph.rs:163-165; 3 docs.rs egg Analysis::make; ir/egraph.rs:396-401; 4 ir/egraph.rs:404-410; 5 isel/extractor/fast.rs:20-22; 6 docs.rs egg Analysis (modify constant-folding example); pipeline.rs:235.
Confidence: high.
Would change: a measured case where class-level constant data cannot express a constant the extractor needs.

### Q4.4
Position: Option D; IR-to-IR rules are named hand-written Rust, and IR-to-ISA rules are data read by one generic applier, decided now.
1. TableGen and ISLE compile rules to code to build a fast matcher automaton; here egg does the e-matching, so generated code buys no matching speed.
2. egg itself treats patterns as data (`Pattern` parsed from text) and attaches conditions through the `Applier` trait, so a data rule plus one applier is idiomatic.
3. The applier re-matches the left-hand side to recover matched e-nodes; with typed payloads this becomes one generic routine, written and tested once, instead of a copy per generated rule.
4. Conditions are small affine expressions over payload fields and pattern positions, which a small expression tree covers.
5. IR-to-IR rules are few and target-independent, so names and hand-written Rust serve them better than a format.
6. Deferring (E) means B6 and B12 are done twice; one lane cannot afford that.
Evidence: 1-2 docs.rs cranelift-isle overlap.rs (codegen relies on rule order); docs.rs egg Runner and Analysis pages; 3 isel/rewrites/applier.rs:100-116; 4 isel/rewrites/ir2ir_rewrites.txt:0-9; 6 minutes/rulings.md:43.
Confidence: medium.
Would change: a condition kind in the corpus that a small expression tree cannot express without an escape to code.

### Q4.5
Position: Option B plus E; one cost-guided k-best extractor ranked by a stated proxy, re-ranked by the cost ABI after allocation, deduplicated by a content key that includes buffers.
1. The fast extractor is cost-blind and stops at the first feasible tree, so it never ranks.
2. The slow extractor enumerates trees up to a growing limit, which grows exponentially with the limit.
3. Optimal DAG extraction is NP-hard in general, so a k-best tree-cost extractor with a proxy is the standard compromise; egg's own `Extractor` is a greedy tree-cost extractor.
4. The critic's point holds: the real cost is known only after allocation, so k must exceed one and allocation failures must leave a defined drop reason; that is why the proxy picks k and the cost ABI picks one.
5. When a non-default cost model is loaded, the proxy can mis-rank, so the plan must state that the k are re-ranked by the loaded model.
6. Re-extracting an unchanged e-graph after saturation is waste; content deduplication across passes removes it.
7. The enumerating extractor can stay as a test-only cross-check on small e-graphs.
Evidence: 1 isel/extractor/fast.rs:42-128; 2 pipeline.rs:497-500 (limit grows per pass); 4 cost/c.rs:5-11; 6 pipeline.rs:540-547; docs.rs egg Extractor.
Confidence: medium.
Would change: a corpus kernel where the best allocated candidate is outside any reasonable k under the proxy.

### Q4.6
Position: Option B; B30 equality and B11 copies land as one item, gated on the shared-overwritten-input oracle test.
1. Decision 17 requires the output to reuse the input's storage; skipping an interference edge only permits overlap.
2. Equality alone overwrites a marked input that has a later reader, so it is wrong without the copy.
3. `fix_inplace` already inserts a copy after extraction, but the copy's cost is invisible to the extractor.
4. In e-graph terms, "copy then in-place" and "out-of-place instruction" are two candidates; building the copy at candidate time lets cost choose, as Q4.1 does for `add_in_place` and `add_d1`.
5. Splitting the item (E) leaves a window where candidates are dropped that the current code accepts.
Evidence: 1 cpp/malloc/src/constraint.cc:214-230; 3 isel/extractor/inplace_fix.rs:6-17; 4 tests/test_extraction.py:405-414; backend/plan/extractor.md:24-28.
Confidence: high.
Would change: B11 sized well above M, which would argue for E with a defined drop reason.

### Q4.7
Position: Option B with E; v3 folds read-only `@s.` in starts, sizes, and constraints; updates and state-only emission come after v3; every other use is a named rejection.
1. An e-graph represents pure terms; egg has no effect ordering, so a state update needs a scheduling pass outside equality saturation.
2. A read of known compile-time state folds to an integer, which is just a payload condition, with no new mechanism.
3. State read order (R7) is contested and state width and data-dependent addressing (RA8, RA9) are unruled, so updates have no settled meaning to implement.
4. Decision 16 forbids silent drops, so an instruction with regions and no semantics is rejected with a message, and a state-only instruction is reported as not emitted.
5. D (run-time state) needs allocator and scheduler changes two engineers cannot absorb.
Evidence: 1 docs.rs egg Runner (rewrites only; no ordering); 3 minutes/rulings.md R7 and audit rows RA8, RA9 (as cited by the agenda); 4 agenda context on generator.py:119-122.
Confidence: medium.
Would change: a case study (plans/casestudies/sme.md) whose kernels need a state update inside one kernel to be useful at all.

### Q4.8
Position: Options B and E; the language track owns checks of meaning, the backend rejects in the v3 batch now and deletes its copy when the front end lands it.
1. An ill-typed semantics that is silently never selected violates decision 16 today.
2. B7 needs shape rules in Rust anyway, for nodes that rewrites create (slice, concat), so the backend copy is not only a check.
3. With typed input, `make` can assert that its inferred shape equals the annotated one; disagreement is an internal error, not a user check.
4. One generated table (D) removes drift but is sized L or more and is not needed to meet decision 16.
Evidence: 2 isel/rewrites/ir2ir_rewrites.txt:0 (rule creates `slice` and `concat`); 3 ir/egraph.rs:396-410; minutes/rulings.md R62.
Confidence: medium.
Would change: front-end and backend rule tables drifting in a test, which would make D worth its size.

### Q4.9
Position: Option B for R4 and R10; for unwritten storage the backend checks read-before-write on the candidate rather than zeroing; integer checks follow panel 1's norm, i64 per intermediate today.
1. "May assume" is undefined-behavior wording; the backend's real obligation is never to emit a violating region, which the allocator enforces.
2. In a selected dataflow graph every read has a producing node, so on-chip read-before-write arises only from partial writes and in-place reuse.
3. Those cases are visible on the candidate, so a check with a defined drop reason costs less than zeroing every on-chip buffer.
4. Address and constraint expressions are small and affine; i128 would cover exact arithmetic if panel 1 rules for it, so the backend does not force the choice.
Evidence: 1 cpp/malloc/src/constraint.cc:214-230; 2 isel/extractor/fast.rs:42-128; 4 act_backend/expression_writer.py:72,94 (as cited by the agenda).
Confidence: medium.
Would change: a kernel whose partial writes are data-dependent, which would make a static read-before-write check impossible.

### Q4.10
Position: Options A and E, plus making the bare-variable rules conditional; phases (B32) only after B29 measures a gain; no egglog.
1. Rules with a bare-variable left-hand side, such as `(?x) => (copy ?x)` and `(?x) => (reshape_? (reshape_? ?x))`, match every class and never saturate; the node limit is the real stop.
2. `Runner::default()` already uses egg's `BackoffScheduler`, so C's first half is in place and does not fix this.
3. Guarding those rules on the ISA having an instruction that consumes the produced operator bounds growth at the source.
4. Rule 0 hard-codes tile 16 and `I32` in the generic crate; generating it from the spec, or dropping it, is target correctness, not tuning.
5. egglog's schedules are attractive but rewrite the core, which decision 11 rules out.
Evidence: 1 isel/rewrites/ir2ir_rewrites.txt:2-4,7; lib.rs:20; 2 pipeline.rs:490 and docs.rs egg Runner ("The default one is BackoffScheduler"); 4 agenda context on ir2ir_rewrites.rs:20-27,100.
Confidence: high.
Would change: B29 measurements showing node-limit stops absent after the guards, which would make phases unnecessary.

### Q4.11
Position: Options A, E, and F (defer B5 and redefine it after B7).
1. The C++ parser reads text that Rust printed one call earlier, so a typed `#[repr(C)]` call removes a format without users.
2. `hold` and `hold_tensors` are allocator logic and stay on the C++ side.
3. One struct written in Rust with its C header derived from it gives one schema without coupling to B13.
4. B13 from today's `instruction.py` unblocks the backend; re-pointing at R42 later is a small change.
5. After B7, the reader no longer computes shapes, so B5's three-layer split loses its reason.
Evidence: 1 agenda context on act_malloc.cc:45-66; 2 cpp/malloc/src/parser.cc (agenda lines 146,335); 5 ir/egraph.rs:396-401.
Confidence: low.
Would change: the front-end object model landing before B13 starts.

### Q4.12
Position: Options A and E; one template per fragment for surviving emitters, no width logic, no formatter run at generation.
1. Under Q4.4 data rules, most IR-to-ISA templates disappear, so the choice applies only to what remains.
2. Generated output is not committed, so the engineering rules require no formatting of it.
3. Width-chosen twins are layout logic that tests do not gate.
4. A developer reading the crate can run the formatter by hand.
Evidence: 2 engineering.md (format edited files before committing); 3 agenda context on template_loader.py:14 and tests/lint.py:19,27.
Confidence: low.
Would change: generated code being read often in debugging, which would favor B.
