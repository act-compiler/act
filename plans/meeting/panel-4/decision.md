# Panel 4 decision: backend architecture

Moderator: Fable.
Paths below are under /home/user/act-backend unless marked; G = act_backend/generic.
The owner's in-meeting input binds: the in-place mark (`set_inplace`) stays, and the backend decisions below work with the mark kept.

| Question | Decision (one line) | Confidence | Owner? |
| -------- | ------------------- | ---------- | ------ |
| Q4.1 | Option B: overlap allowed, cost chooses, ties broken by a declaration index stored on the ISA e-node; exact duplicates warned (D) | high | no |
| Q4.2 | Option B: deterministic mode first and the only test mode; chosen assembly plus `--log` gate structural steps; oracle run gates behavioral steps; one equivalence check in B29 | high | minor |
| Q4.3 | Option A order (B31, B7, B9, B6) with F (`Option` shape) and D (constants in class data); no C | high | no |
| Q4.4 | Option C: IR-to-IR hand-written named Rust; IR-to-ISA generated Rust bound by name with one shared re-match helper; a bounded two-rule spike of D may overturn it | medium | no |
| Q4.5 | Option B plus E as the end state; file merge and duplicate-pass removal first, k-best after B29 and B9, enumerator kept test-only | medium-high | no |
| Q4.6 | Option E: B30 equality first with `fix_inplace` kept, B11 later moves the copy into candidate building; gated on the test at tests/test_extraction.py:507 | medium-high | no |
| Q4.7 | Option B plus E: read-only `@s.` folded at compile time in v3, updates after v3, every unsupported use a named rejection, silent drop fixed first | medium-high | no |
| Q4.8 | Option B now and A for the literal check; D as the end state if the language track owns the table; E only where no new table is needed | medium | no |
| Q4.9 | Option B for R4 and R10; reject read-before-write on the candidate with a drop reason; i64 per intermediate in the backend, norm owned by panel 1 | medium | no |
| Q4.10 | Options A and E: keep egg, rule 0 constants from the spec, guards on the bare-variable rules, B32 phases only if B29 still shows node-limit stops | high | no |
| Q4.11 | Option A with E and redefined F; the Rust struct is the schema and the C header is derived from it | medium-high | yes |
| Q4.12 | Option A under E: one template per fragment, no width logic, no formatter run at generation | high | minor |

## Q4.1

- Decision: Option B, with Option D's exact-duplicate case as a generation warning.
- 1. The fixture `in_place()` declares `add_in_place` and `add_d1` on one sum on purpose (tests/test_extraction.py:405-414), so rejecting overlap breaks a deliberate design.
- 2. The tie-break today is e-node order in the class (G/src/isel/extractor/fast.rs:24, "e-node order keeps the numbering stable"), which depends on rewrite application history, not on the spec.
- 3. Decision 16 needs a defined selection, so the key is (cost, declaration index), the index stored on the ISA e-node and compared last.
- 4. Declaration order keeps the fixture's observed choice (tests/test_extraction.py:409, "Declared first, so its e-class holds it first"), so no golden moves; a name order (C) ties the result to renames.
- 5. A duplicate adds e-nodes and no choice, so a warning suffices.
- Settled fact: the ISLE language reference says the system "allows multiple applicable rules to exist with the same priority" and that "the tie-breaking heuristic is not part of the specified language semantics"; ISLE's overlap.rs reports an overlap only when `a.prio == b.prio` and exempts multi-constructors because "the ctor returns every match".
- The two sources describe two layers; the e-graph case is the multi-constructor case, so both support B.
- Dissent: none of substance.
- Confidence: high.

## Q4.2

- Decision: Option B; the deterministic mode (B29) lands first and is the only test and CI mode; identical chosen assembly and `--log` candidate files gate structural steps; the oracle-differential run over tests/test_extraction.py and tests/test_matching.py gates behavioral steps.
- 1. The search stops at `TIME_LIMIT` 1000 ms (G/src/lib.rs:18) and when elapsed time exceeds the best candidate's timestamp times `SATURATION_FACTOR` (G/src/pipeline.rs:155-170), so the chosen candidate depends on machine speed.
- 2. The `Runner` is built with `with_time_limit(TIME_LIMIT)` and a hook calling `check_termination` (G/src/pipeline.rs:491-505), so both the stop and the extraction cadence are wall-clock bound today.
- 3. Goldens are simulated outputs and cannot prove an unchanged selection; a behavioral step changes selection by design, so only the oracle run can gate it.
- 4. Two gating modes double what two engineers debug (decision 11), so the equivalence check of Option E runs once inside B29, and goldens are re-recorded once in a commit of their own if the modes differ.
- 5. Option D replaces a cheap equality check with a new subsystem and cannot say a structural step is unchanged; Option C leaves B6, B10, B30, and B32 ungated.
- Dissent: none; the three converged.
- Owner: whether the wall-clock mode stays the user default outside tests is a value judgment, listed in the final section.
- Confidence: high.

## Q4.3

- Decision: Option A order (typed payloads B31, shape and type in `make` B7, no clone B9, rewrite values B6), with F's `Option` shape and D's constants in class data; Option C is rejected for v3.
- 1. `matches` compares only `std::mem::discriminant` (G/src/ir/egraph.rs:163-165), so payload strings never take part in matching and typed payloads change nothing egg sees; B31 is the lowest-risk base.
- 2. `make` computes no shape (G/src/ir/egraph.rs:396-401), so every rewrite sets shapes from outside; B6 before B7 would regenerate `set_shapes_N` code that B7 deletes (G/src/isel/rewrites/applier.rs:43-56).
- 3. `merge` panics only when both shapes are non-empty and differ (G/src/ir/egraph.rs:405-411), so rank-0 reads as unset; `Option<Shape>` with an asserting `merge` separates the cases.
- 4. Each extraction clones the e-graph (G/src/pipeline.rs:234-240) because the extractors add `DetectedConst` nodes (G/src/isel/extractor/slow.rs:21-24); constants in class data with the node emitted at candidate building removes the mutation and the clone, and avoids `NODE_LIMIT` growth that Option E would add.
- 5. Attributes as child e-nodes (C) would let rewrites union attribute classes with unrelated integers and change every operator's arity and rewrite string; no rule in ir2ir_rewrites.txt needs it.
- Dissent: none after Fable withdrew the scoped C.
- Confidence: high.

## Q4.4

- Decision: Option C; IR-to-IR rules stay hand-written named Rust, IR-to-ISA rules are generated Rust with variables bound by name, and one shared re-match helper serves every rule.
- A two-rule spike of Option D (`add_in_place` plus one dot rule) may overturn this only if it lands under about 300 lines with typed errors, before B6 continues; the spike is the first two B6 rules, not a separate experiment.
- 1. The defect is the three hand-indexed tables binding rule number to `precond_N`, `metadata_N`, `set_shapes_N` (G/src/isel/rewrites/applier.rs:15-58); name binding fixes it in either form.
- 2. Settled fact: egg's `Subst` holds `(Var, Id)` pairs only (egg-0.10.0 src/subst.rs:100-102), so no form can read matched e-nodes' payloads from it; `find_matches` (G/src/isel/rewrites/applier.rs:106-113,168-215) survives in both forms as one helper.
- 3. The generator already compiles conditions to Rust: `ConditionWriter` with `condition`, `conjunction`, `comparison` and `generate_constraint_checks` (act_backend/isel_ir2isa_rewrites_rs_generator.py:415-528), over `CheckedWriter`'s checked i64 operators (act_backend/expression_writer.py:71-96).
- 4. Generated code is therefore the smaller delta: replace the tables with one generated value per rule; Option D writes a data format, an evaluator, and its error paths and deletes working emitters, which decision 11 weighs against.
- 5. Fable's claim that the evaluator is "about the same size" as today's Python is unmeasured, so it carries no weight against step 4; Sonnet's spike is the cheap measurement.
- 6. Opus's round-1 point that generated code buys no matching speed here is true and does not decide the question, since the cost is in generator work, not matching.
- Dissent: Fable and Sonnet prefer D (Sonnet after the spike), on the ground that the condition grammar is closed and one evaluator carries the i64 and division norms once.
- Confidence: medium.

## Q4.5

- Decision: Option B plus E as the end state, in Sonnet's order: merge the two files and remove the duplicate post-saturation pass now; one k-best extractor on (class, buffer) after B29 and B9; the enumerating extractor kept as a test-only cross-check.
- 1. The fast extractor is cost-blind first-feasible DFS (G/src/isel/extractor/fast.rs:42-60) and the slow one enumerates up to a limit that grows by one per pass (G/src/pipeline.rs:489-502), so neither ranks.
- 2. The default cost counts `api.` lines of assembly (G/cpp/cost/count.cc:11-19), which exist only after allocation, so a pre-allocation proxy can only approximate it; the k candidates are re-ranked through the cost ABI after allocation.
- 3. Allocation feasibility is a filter, not a ranking, so k-best then filter replaces both strategies; the critic's "cost known after allocation" argues for k greater than one, not for two extractors.
- 4. After saturation the loop re-extracts an unchanged e-graph until the time limit (G/src/pipeline.rs:544-547); removing that is behavior-preserving and lands at once; content dedup (B34) keyed on every node's buffer covers the rest.
- 5. The merge changes `--log` order and drops candidates, so it is behavioral and waits for the Q4.2 gate; k is measured on the examples before it is fixed.
- Dissent: none after round 2.
- Confidence: medium-high.

## Q4.6

- Decision: Option E; B30 (equality of the two offsets in the OR-Tools solver) lands first with `fix_inplace` kept, and B11 later moves the copy into candidate building so its cost is visible to the extractor; each step is gated on the shared-overwritten-input oracle test (tests/test_extraction.py:507-520).
- 1. Decision 17 says the output reuses the marked input's storage; today the allocator only skips the interference edge (G/cpp/malloc/src/constraint.cc:217-231), which permits overlap and does not force one offset; equality is the direct encoding.
- 2. Settled fact: `fix_inplace` runs inside both extractors before allocation (G/src/isel/extractor/fast.rs:196, slow.rs:281), gives a copy to every overwritten operand with more than one use (G/src/isel/extractor/inplace_fix.rs:64-72), and returns `Err` for an operand pinned in HBM or a view with another reader (inplace_fix.rs:38-40,66-68), which drops the candidate with a reason.
- 3. So a candidate reaching the allocator never has a shared overwritten input, and the "window" Fable and Opus named in round 1 does not exist; both conceded in round 2.
- 4. Equality only makes some allocations infeasible, each a solver drop with a reason, which is defined behavior under decision 16; a candidate the overlap-permitting solver placed at different offsets was wrong under decision 17 anyway.
- 5. B11 is a cost-quality improvement, not a correctness prerequisite, so it need not block B30; Fable's fallback (E with a drop reason) is today's behavior.
- 6. Options C and D are rejected: C turns a constraint into a luck of the solver, D moves allocation into the extractor.
- The read-after-write meaning of the mark belongs to panels 1 and 2; the backend reads the mark whichever way it is declared, so this decision stands with `set_inplace` kept.
- If panels 1 and 2 weaken the mark to "may reuse", the equality becomes a preference rather than a constraint; that is a cross-panel dependency, not an open backend question.
- Dissent: Fable holds that B30 and B11 land as one item to avoid an interim regression on kernels with a live marked input; the code shows the interim state is correct and the regression is only in cost, so sequencing wins.
- Confidence: medium-high.

## Q4.7

- Decision: Option B plus E; v3 folds read-only `@s.` in starts, sizes, and constraints at compile time; updates and state-only emission follow after v3; every unsupported use is a named rejection; the silent drop is fixed first.
- 1. Every state read in an instruction with semantics is rejected today (act_backend/spec_checks.py:249-251), and an instruction without semantics is silently left out (act_backend/generator.py:119-122), which decision 16 makes a defect; a message or rejection there is a small step needing no state design.
- 2. A read of known compile-time state folds to an integer, which is a payload condition the writer already handles.
- 3. An e-graph holds pure terms and egg's `Runner` orders no effects, so an update needs a scheduling pass outside saturation; with R7 contested and RA8, RA9 unruled, updates have no settled meaning to implement in v3.
- 4. Option D makes state an input of allocation and selection, a new search dimension two engineers cannot absorb; Option C keeps a deficiency the owner named.
- 5. The `@x.` writer path would compile `@s.n` as attribute `n` (D12), so the rejection stays loud until each use is supported.
- Dissent: none; timing depends on panel 1 ruling R7 and RA8.
- Confidence: medium-high.

## Q4.8

- Decision: Option B as the contract now (the backend asserts typed semantics IR, and under B7 `make` asserts that the inferred shape equals the annotation); Option A for the literal element-count check; Option D as the end state if the language track owns one generated table; Option E only for a rejection that needs no new table.
- 1. `check_regions` rejects only a literal count above the region and skips any non-literal (act_backend/spec_checks.py:394-418), so it is a floor with no table behind it and stays until the front end lands the same check.
- 2. The backend never reads `var_type` (D2, D3), so an ill-typed semantics is silently never selected; an assertion on typed input meets decision 16 without a parallel checker.
- 3. B7 needs shape rules in Rust for nodes the rewrites create (`slice_?`, `concat_0` in G/src/isel/rewrites/ir2ir_rewrites.txt:1), so that subset exists under every option; D's value is keeping the two subsets from drifting.
- 4. A backend rejection now (E) needs the same table B21 to B23 and the front end will write; decision 10 makes two rejections one user-visible break, but not one item of work for two engineers.
- Dissent: Opus rejects in the v3 batch now and deletes later, on decision 16; the assertion meets decision 16 cheaper.
- Confidence: medium.

## Q4.9

- Decision: Option B for R4 and R10; for unwritten storage, a read-before-write check on the candidate with a defined drop reason, no zeroing; integers stay i64 per intermediate in the backend, with the norm owned by panel 1.
- 1. "May assume it never happens" is undefined-behavior wording and decision 16 bans it; the backend's obligation is never to emit a violating region or constraint, and the allocator enforces the bound it already models.
- 2. In a selected dataflow graph every read has a producer, so read-before-write arises only from partial writes and in-place reuse, which are visible on the candidate; zeroing would cost instructions in every kernel.
- 3. The writer checks every operator with `checked_add`, `checked_sub`, `checked_mul` and floored division helpers returning `None` past i64 or by zero (act_backend/expression_writer.py:71-96), so overflow is a defined error today.
- 4. Exact arithmetic (R56) would need i128 or big integers in generated Rust and a matching oracle, and the C++ allocator holds 64 bits, so it would not remove the limit end to end.
- 5. If panel 1 rules exact arithmetic for the oracle, the difference is one-sided (the backend errors where the oracle succeeds) and the Q4.2 gate compares only runs where both succeed or both error.
- Dissent: Fable and Opus want i64 in both executors; that ruling is panel 1's.
- Confidence: medium.

## Q4.10

- Decision: Options A and E; keep egg; rule 0's tile and dtype come from the spec or the rule is dropped where no instruction needs it; the bare-variable rules are guarded on the ISA having a consumer of the produced operator; B32 phases only if B29 still records node-limit stops; no egglog.
- 1. Settled fact: `BackoffScheduler` is egg's default (egg-0.10.0 src/run.rs:65, 352, 406) and the pipeline builds `Runner::default()` with no `with_scheduler` (G/src/pipeline.rs:491-495), so Option C's scheduler half is already in force and does not stop the node-limit stops.
- 2. Rules 2, 3, 4, and 7 have a bare-variable left-hand side (G/src/isel/rewrites/ir2ir_rewrites.txt:3-5,8) and match every class, so saturation ends at `NODE_LIMIT` 10000 (G/src/lib.rs:20); a guard bounds growth at the source and is smaller than phases.
- 3. Rule 0 hard-codes tile 16 and `Dtype::I32` in the generic crate (G/src/isel/rewrites/ir2ir_rewrites.rs:20-27,100), which is target correctness, not tuning.
- 4. egg's `Runner` has no rulesets, so phases are separate `run` calls; they are a behavioral step that can lose a candidate interleaving finds (B32 caveat), so they wait for the B29 measurement.
- 5. egglog has rulesets and schedules but is a core rewrite that decision 11 excludes.
- Dissent: Fable holds phases as the standard shape regardless; the B29 measurement decides it.
- Confidence: high.

## Q4.11

- Decision: Option A (one typed `#[repr(C)]` call carrying candidate and kernel facts; the C++ candidate parser deleted; `create_instruction`, `hold`, `hold_tensors` kept; the Rust printer kept for `--log`), with Option E for B13's model source and Option F redefined after B7 as "reader checks annotations, analysis infers"; the Rust struct is the single schema and the C header is derived from it.
- 1. `act_malloc.cc:45-66` reads the candidate line by line through `parse_line`, `create_instruction`, and `hold_tensors`, and any `why` drops the candidate; the text was printed by Rust one call earlier, so the round trip has no consumer but the parser.
- 2. The struct carries candidate lines, buffers, offsets, and kernel facts, none owned by the B13 instruction model, so emitting it from B13 (Option B) couples two unrelated schemas.
- 3. B13 built from today's `instruction.py` unblocks the backend; re-pointing at the R42 model later is small.
- 4. Deriving the header needs cbindgen or a small generator; cbindgen is absent from G/Cargo.toml (egg, itertools, libc, serde, serde_json), so it is an ask, and the fallback is a hand-kept header with a size and offset assertion test.
- 5. B25 (`assert(false)` paths to error returns) lands first anyway as the cheap step.
- Dissent: none of substance.
- Confidence: medium-high.

## Q4.12

- Decision: Option A under Option E; one template per fragment, no width logic, no formatter run at generation, applied only to emitters that survive Q4.4.
- 1. `LINE_WIDTH` 100, `CALL_WIDTH` 60, and five `TWINS` (act_backend/template_loader.py:8-14) re-implement rustfmt's layout in Python.
- 2. No gate consumes the generated layout: the lint runs clippy on meaning, and the CI format check covers `git ls-files` only; the one rustfmt test (tests/test_generators.py:166-185) gates the twins' own output and goes with them.
- 3. rustfmt at generation (B) adds a run-time requirement to a user's generation path for no reader; `cargo fmt` on the output is one command for a reader who wants it.
- Dissent: Sonnet's fallback to B if generated crates are read in review.
- Confidence: high.

## Escalation to the owner

No debater filed an "Escalation to the owner" section; the items below are the value judgments and dependency asks the panel surfaced.

- Q4.2: whether the wall-clock mode stays the user default outside tests, or the deterministic mode becomes the only mode; best case for keeping it: a user gets the best kernel the time budget allows; best case for dropping it: one mode is one surface to debug.
- Q4.6: the read-after-write meaning of the in-place mark is for panels 1 and 2; the backend decision stands with `set_inplace` kept and treats decision 17 as an equality constraint.
- Q4.9: the integer norm (i64 per intermediate in both executors, or exact in the oracle) is panel 1's; the backend states i64 checked either way.
- Q4.11: adding cbindgen to the generic crate to derive the C header from the Rust struct is a new dependency and needs the owner's yes; otherwise the header is generator-emitted or hand-kept with an assertion test.
- Q4.12: if the owner wants generated crates formatted for review, Option B (rustfmt at generation) replaces Option A.
