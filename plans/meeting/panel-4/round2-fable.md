# Panel 4, round 2, Fable (outside compiler architect)

Lens: how production selectors solve the same problems, weighed against decision 11's two engineers.
Positions changed in this round: Q4.4 (to Option D), Q4.10 (guards before phases), Q4.11 (to Option A with the header derived from the Rust struct).

### Q4.1

All three take Option B with cost first and declaration index as the last key, so this question is settled.
Opus's refinement is the right mechanics: store the declaration index on the ISA e-node and compare it last, rather than relying on e-node order in the class.
Sonnet's generation-time lint for exact duplicates is harmless and I accept it as a warning, since duplicates add e-nodes and no choice.
Final position: Option B, with the declaration index stored on the e-node and an exact-duplicate warning.
Confidence: high.
Divides the panel: nothing.

### Q4.2

All three take Option B; the only difference is whether the deterministic mode is the sole CI mode or a flag with a one-time equivalence check.
Strongest opposing argument (Sonnet, Opus): the default and deterministic modes must agree once on every example, and goldens are re-recorded once if they differ, in a commit of their own before any other step.
Concession: that one-time check costs minutes and catches hidden hash-order nondeterminism, which Opus names as the main risk, so it belongs in B29.
My remaining point stands: after that check, tests run only the deterministic mode, since two gating modes double what two engineers debug.
Evidence: act-backend/act_backend/generic/src/pipeline.rs:487-505 builds the `Runner` with `with_time_limit(TIME_LIMIT)` and a hook that calls `search.check_termination()`, so both the stop and the extraction cadence are wall-clock bound today; tests/conftest.py:56 and 197 already generate and simulate against the oracle, so the differential run is a new pytest entry over existing machinery.
Final position: Option B, with Option E's one-time equivalence check inside B29 and the deterministic mode as the only test mode afterwards.
Confidence: medium-high.
Divides the panel: only whether the wall-clock mode stays a user default; that is a value judgment, not an evidence question.

### Q4.3

All three take Option A's order with Option D's class-data constants and Option F's `Option` shape.
Strongest opposing argument (Opus, Sonnet): attributes as child e-nodes (Option C) let rewrites union attribute classes, enlarge every match, and change the language arity and every rewrite string, so it is not a small step.
Concession: I withdraw Option C even in scoped form for v3; with typed payloads the conditions read fields directly, and no agenda item needs an attribute to be a class.
Evidence: act-backend/act_backend/generic/src/ir/egraph.rs:163-165 compares only `std::mem::discriminant`, so payloads never participate in matching today and typed payloads (B31) change nothing egg sees.
Final position: Option A order, Option D constants, Option F `Option` shape, no Option C.
Confidence: high.
Divides the panel: nothing.

### Q4.4

Strongest opposing argument (Opus, with Sonnet): TableGen and ISLE compile rules to code to build a matcher automaton; here egg does the e-matching, so generated Rust buys no matching speed, and egg's own `Pattern` is data with conditions attached through the `Applier` trait, so data plus one applier is the idiomatic shape.
Rebuttal to my own round-1 claim: I argued that a data applier needs a second language with its own evaluator and errors.
The condition language already exists and is closed: a TAIDL constraint is conjunctions and disjunctions of comparisons over additive expressions with `+ - * / %` on attributes, shape dimensions, and literals.
Evidence: act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:415-500 (`ConditionWriter.condition`, `conjunction`, `comparison`, `generate_constraint_checks`) walks exactly that grammar and emits Rust `let` chains with checked i64 arithmetic and a divide-by-zero refusal; act-backend/act_backend/expression_writer.py:70-100 (`CHECKED_OPERATORS`, `FLOORED_OPERATORS`, `CheckedWriter`) is the whole operator set.
A Rust evaluator over that AST is one typed module of about the same size as the Python that lays the Rust out today, and it carries the i64 and division norms once rather than per emitted rule.
Result attributes are the same grammar over `lhs_metadata[position].shape` (isel_ir2isa_rewrites_rs_generator.py:240, 350), so they fit the same tree.
What generated code still buys is clippy and the type checker on each rule; a closed grammar evaluated by one tested function gets the same guarantee once.
On timing, Opus is right that deferring to a Q2 experiment makes B6 and B12 twice; Sonnet's two-rule spike is the first two rules of B6, not a separate experiment, so both views fit one schedule.
Final position: Option D, decided now, with `add_in_place` and one dot rule as the first B6 rules and the generic applier binding variables by name.
Confidence: medium.
Divides the panel: nothing on form; Sonnet still conditions the decision on the spike, which the first two B6 rules supply.

### Q4.5

Strongest opposing argument (Sonnet): the merge into one k-best extractor is behavioral, so it must wait for the deterministic gate (B29) and the clone removal (B9), while merging the two files and deleting the duplicate post-saturation pass are structural and can land at once.
Concession: the sequencing is right and costs nothing; one k-best extractor remains the end state, and Sonnet reaches Option A or B after the gate as well.
Opus's point that the enumerating extractor can stay as a test-only cross-check on small e-graphs is a cheap guard against a proxy mis-rank and I accept it.
Evidence: act-backend/act_backend/generic/src/pipeline.rs:487-505, where `limit` grows by one per hook call and `search.pass` runs every `N` iterations, so one pass shape already drives both extractors; cpp/cost/count.cc:11-17 counts `api.` lines, so the proxy equals the default cost only by construction.
Final position: Option B plus E as the end state, reached in Sonnet's order: file merge and duplicate-pass removal first, k-best after B29 and B9, the enumerator kept test-only.
Confidence: medium.
Divides the panel: the value of k, which all three agree is measured on the examples before it is fixed.

### Q4.6

Strongest opposing argument (Sonnet): equality (B30) first with a candidate dropped with a stated reason until B11 lands is a defined result under decision 16, so the window I called a violation is not one, and B30 is one local change in constraint.cc verified from the `--log` offsets.
Concession on the norm: a drop with a reason is defined behavior, so Option E does not violate decision 16; my round-1 step 3 overstated this.
Rebuttal on the outcome: Opus's point holds that the interim drops candidates today's `fix_inplace` accepts, so Option E is a visible regression on kernels with a live marked input until B11 lands.
Evidence: act-backend/act_backend/generic/src/isel/extractor/inplace_fix.rs:9-18 builds the copy after extraction today; act-backend/tests/test_extraction.py:507-520 (`pinned`) has an in-place load with no store, the shape of kernel where no copy through HBM exists and a drop would be the only answer.
A regression between two commits of one release is acceptable only if B11 fits the quarter; nobody has sized it, which is what the Option E fallback is for.
Final position: Option B as one item gated on the shared-overwritten-input oracle test; Option E with the drop reason is the fallback if B11 is sized above one quarter.
Confidence: medium-high.
Divides the panel: whether the interim regression is worth landing B30 early; that is a scheduling judgment for the owner.

### Q4.7

All three take Option B with Option E's loud rejections, and all three reject Option D as a new search dimension two engineers cannot absorb.
Opus adds the structural reason I lacked: an e-graph holds pure terms and egg has no effect ordering, so a state update needs a scheduling pass outside saturation, which confirms that updates cannot be a v3 item.
Final position: Option B plus E, with the silent drop at generator.py:119-122 turned into a message or rejection as the first step.
Confidence: medium-high.
Divides the panel: nothing.

### Q4.8

Strongest opposing argument (Opus): B7 needs shape rules in Rust anyway for the nodes rewrites create (slice, concat), so the backend copy is not only a check; and one generated table (Option D) is sized L or more and is not needed to meet decision 16, so the backend should reject in the v3 batch now (Option E) and delete its copy later.
Concession: the Rust `make` rules for rewrite-created nodes exist under any option, so "two copies" is unavoidable for that subset; Option D's value is bounded to keeping the two subsets from drifting.
Rebuttal on Option E: a rejection the backend writes now needs the same rule table (B21 to B23) the front end will write, so the duplicated work is the whole item, not its message; decision 10 makes the two rejections one release for the user, but not one item for the two engineers.
Sonnet's order avoids that: the backend asserts typed input now (Option B), keeps its literal count check (Option A), and deletes it in the batch that lands the front-end rule.
Evidence: act-backend/act_backend/spec_checks.py:394-418 (`check_regions`) rejects only a literal count above the region's and skips any non-literal, so it is a floor with no table behind it; act-backend/act_backend/generic/src/ir/egraph.rs:396-401 computes no shape in `make`, so the assertion of agreement between annotation and inference is new code under B7 anyway.
Final position: Option B as the contract now and Option A for the literal check; Option D only for the subset B7 needs, and only if the language track owns the table; Option E only for a rejection that needs no new table.
Confidence: medium.
Divides the panel: whether a backend-side rejection in v3 is worth its duplicated table; Opus says yes for decision 16, Sonnet and I say the assertion meets decision 16 cheaper.

### Q4.9

All three take Option B for R4 and R10: the backend never emits a violating region or constraint, and the allocator enforces the bound.
On unwritten storage, Opus and I converge on a read-before-write check on the candidate with a defined drop reason, and Sonnet accepts reject as the honest backend rule, so zeroing is out for all three.
Strongest opposing argument (Sonnet) on integers: exact in the oracle and i64-checked in the backend differ only when an intermediate leaves i64 and the final value returns inside, which is a defined, safe difference to document; i128 would not remove the limit because the C++ allocator holds 64 bits.
Partial concession: the difference is defined and one-sided, so it does not breach decision 16, and the allocator's 64-bit storage is a real end-to-end limit.
What I keep: panel 1 owns the norm, and the backend's obligation is to state its own as H4's i64 per intermediate with a fault, which the writer already implements.
Evidence: act-backend/act_backend/expression_writer.py:70-75 (`CHECKED_OPERATORS`, `FLOORED_OPERATORS`) and 92-96 (`operation` returning `None` past i64 or by zero); plans/semantics/holes.md H4.
Final position: Option B for bounds; Option C's reject branch as a read-before-write check with a drop reason; i64 per intermediate in the backend, documented as one-sided if panel 1 rules exact arithmetic for the oracle.
Confidence: medium.
Divides the panel: nothing within the backend's obligation; the integer norm is panel 1's.

### Q4.10

Strongest opposing argument (Opus): `Runner::default()` already uses egg's `BackoffScheduler`, so Sonnet's stopgap is in place and does not fix the bare-variable rules; guarding those rules on the ISA having an instruction that consumes the produced operator bounds growth at the source; phases (B32) should wait until B29 measures a gain.
Verification: the egg `Runner` documentation states "BackoffScheduler is the default scheduler" and `with_scheduler` says "The default one is BackoffScheduler"; act-backend/act_backend/generic/src/pipeline.rs:491-495 calls `Runner::default()` with node, iteration, and time limits and no `with_scheduler`, so the default scheduler is in force today.
Concession: Sonnet's Option C stopgap is already running and the node-limit stops persist, so backoff is not a fix; and Opus's guards are smaller than phases and measurable under B29.
Rebuttal kept: phases remain the standard shape for expansion then selection, and egg's `Runner` has no rulesets, so separate `run` calls are the only way to express them; they are a second step if the guards leave node-limit stops.
Evidence: act-backend/act_backend/generic/src/isel/rewrites/ir2ir_rewrites.txt:3-8 (bare-variable left-hand sides); lib.rs:20 (`NODE_LIMIT` 10000); ir2ir_rewrites.rs:20-27, 100 (tile 16 and `Dtype::I32` in the generic crate).
Final position: Option E first (rule 0's constants from the spec or the rule dropped), then conditional guards on the bare-variable rules, then Option B's phases only if B29 still records node-limit stops; no egglog.
Confidence: high.
Divides the panel: whether phases are a planned step or a conditional one; the B29 measurement decides it.

### Q4.11

Strongest opposing argument (Opus, Sonnet): one struct written in Rust with its C header derived from it gives one schema without coupling the FFI to B13, and the candidate facts the struct carries are not instruction-model facts.
Concession: the struct carries candidate lines, buffers, offsets, and kernel facts from `metadata.json`, none of which the B13 instruction model owns, so emitting it from B13 (Option B) couples two unrelated schemas; the Rust struct is the single source and the C header is derived from it.
Deriving the header needs either cbindgen or a small generator; cbindgen is a dependency not in act_backend/generic/Cargo.toml today, so it is asked for, not assumed.
Evidence: act-backend/act_backend/generic/cpp/malloc/src/act_malloc.cc:45-66 reads the candidate line by line through `parser::parse_line` and `parser::create_instruction`, then `parser::hold_tensors`, and any `why` drops the candidate, which is the lossy round trip all three want gone; `hold_tensors` is the allocator's own bookkeeping and stays.
Final position: Option A, with the C header derived from the Rust struct; Option E for B13's model source; Option F redefined after B7 as "reader checks annotations, analysis infers".
Confidence: medium-high.
Divides the panel: nothing of substance; the header tool is an ask to the owner.

### Q4.12

Strongest opposing argument (Sonnet): rustfmt is already a project dependency and the generated crate is Rust, so running it at generation removes the width logic with no new tool, while one template per fragment still leaves layout to the templates.
Rebuttal: rustfmt is a CI dependency on committed files, not a dependency of a user's generation path; a toolchain installed with the minimal rustup profile lacks it, so generation would gain a run-time requirement for no reader.
New evidence: act-backend/tests/test_generators.py:166-185 (`rustfmt_complaints`, `test_wide_compared_values_are_laid_out_as_rustfmt_lays_them`) already runs rustfmt in tests to prove the twins match its layout, so the twins do have a gate today; it is a gate on the twins' own output and goes with them.
act-backend/act_backend/template_loader.py:8-14 pins `LINE_WIDTH` 100, `CALL_WIDTH` 60, and five `TWINS` to rustfmt.toml and .clang-format, which is the formatter re-implemented in Python that Option A deletes.
Under Q4.4's Option D most IR-to-ISA templates disappear, so the question is small and Option E scopes it.
Final position: Option A under Option E; `cargo fmt` on the output is one command for a reader who wants it.
Confidence: high.
Divides the panel: Sonnet's fallback to rustfmt at generation, a dependency question for the owner if generated crates are read in review.
