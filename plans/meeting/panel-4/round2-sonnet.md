# Panel 4, round 2: Sonnet (incremental maintainer)

Lens: every step leaves act-backend generating, building, and compiling the examples.
Paths below are under /home/user/act-backend unless marked otherwise.
G = act_backend/generic.

## Summary of changes

| Question | Round 1 | Round 2 | Changed |
| -------- | ------- | ------- | ------- |
| Q4.1 | B plus D lint | B plus D lint | no |
| Q4.2 | B plus E check | B, deterministic as the only CI mode, one-time equivalence check | refined |
| Q4.3 | A order, D, F | A order, D, F | no |
| Q4.4 | D after spike | D after spike | no |
| Q4.5 | C now, then A/B | C now, then B plus E | refined |
| Q4.6 | B30 first, drop between | B30 with fix_inplace kept, B11 as cost step | refined |
| Q4.7 | B | B | no |
| Q4.8 | A, D long-run | B now, D end state | yes |
| Q4.9 | B, C, E third variant | B, C reject, i64-checked, norm deferred to panel 1 | refined |
| Q4.10 | E plus B, C stopgap | A plus E, guards, phases after measurement | yes |
| Q4.11 | A plus E | A plus E, schema from generator, no new dependency | refined |
| Q4.12 | E, else B | A under E | yes |

### Q4.1
All three agree on Option B with declaration order as the stated tie-break; Opus and Fable add that the key is stored on the ISA e-node.
Strongest opposing argument (Opus): ISLE's overlap checker rejects equal-priority overlap, so "ISLE allows overlap" is wrong; the e-graph case is the multi-constructor case that the checker exempts, so allowing overlap is justified for a different reason than I gave.
Concession: I fetched the wasmtime language reference; it says equal-priority overlap is allowed and "the tie-breaking heuristic is not part of the specified language semantics".
Opus cites the separate overlap checker (docs.rs cranelift-isle overlap.rs), which I did not re-fetch; the two sources describe different layers, and my step 3 was too broad.
The conclusion does not depend on either: tests/test_extraction.py:405-414 builds `add_in_place` and `add_d1` on one sum, and fast.rs:24-26 ties by e-node order.
Rebuttal to rejecting duplicates as a warning only (Opus): a duplicate is dead code the author did not intend, so a generation-time lint costs nothing; it need not be an error.
Final position: Option B; key is (cost, declaration index) stored on the ISA e-node; exact duplicates are a generation lint.
Confidence: high (was medium).
Still divides the panel: nothing material; only whether the duplicate lint is a warning or an error.

### Q4.2
Strongest opposing argument (Fable): Option E keeps two modes that must agree, which doubles the debugging surface for two engineers; make deterministic the only CI mode and keep wall-clock as the user default.
Concession: the CI and every gate run the deterministic mode only (B29).
The one-time equivalence run stays, as a one-off check before the first step, not a standing gate.
Evidence that the time dependence is real: G/src/pipeline.rs:157-170 stops on `current_time > timestamp * SATURATION_FACTOR` and G/src/lib.rs:18 sets `TIME_LIMIT` to 1000 ms.
Opus's point 4 is right and sharpens mine: a behavioral step changes selection by design, so identical assembly gates only structural steps, and the oracle run gates behavioral ones.
Final position: Option B; deterministic mode first and the only CI mode; chosen assembly plus `--log` files gate structural steps; the oracle run over test_extraction.py and test_matching.py gates behavioral steps; goldens re-recorded once if the modes differ.
Confidence: high.
Still divides the panel: nothing; Opus, Fable, and I differ only in wording about E.

### Q4.3
Where all three already agree: Option A order, `Option` shape, constants in class data.
Residual difference (Fable): attributes as child e-nodes (Option C) for the attributes patterns must match, such as slice offsets.
Rebuttal: ir2ir_rewrites.txt:0 already writes `slice_?0` as a payload in the operator, and `Language::matches` compares discriminants only (egraph.rs:163-165); once payloads are typed (B31), `matches` can compare the typed payload and no child node is needed.
Making attributes children changes the arity of every operator and every rewrite string, which is not a small step.
Revisit Option C only if a rule needs to union two attribute classes, and no rule in ir2ir_rewrites.txt does.
Final position: B31, then B7 with `Option` shape, then B9 with the constant in class data, then B6.
Confidence: high (was medium).
Still divides the panel: how far Option C goes; Fable says "only where patterns must match", which no current rule requires.

### Q4.4
Strongest opposing argument (Fable): every mature selector compiles rules to code; a data applier needs its own expression evaluator and type checker, which is a second language; and generated code binds names from the `Subst` and deletes `find_matches`.
Rebuttal on the last claim, which I verified: `Subst` binds variables to e-class ids only, and the conditions need the matched e-nodes' payloads.
G/src/isel/rewrites/applier.rs:106-113 shows `find_matches` recovering `(eclass, Option<TensorOp>)` pairs, walking `egraph[eclass].nodes` at applier.rs:182-205.
A generated rule would still have to enumerate e-nodes of the root class to read payloads, so `find_matches` survives in both forms; only the data form has one copy of it.
Concession to Fable: the second-language cost is real, and I have no measurement that a data applier is smaller.
The whole applier is 215 lines (applier.rs), which bounds what the data form replaces; the data form is justified only if the expression trees stay small.
Fable's own stop condition (a prototype under 300 lines with typed errors) is a spike criterion I accept.
Final position: Option D, decided by a two-rule spike (`add_in_place` plus one dot rule) against the Q4.2 gate; if the spike needs more than integer arithmetic and comparison over attributes, Option C.
Confidence: low.
Still divides the panel: Opus decides D now; Fable chooses B; I decide after a days-long spike.

### Q4.5
Strongest opposing argument (Fable and Opus): one k-best extractor on (class, buffer) is the standard shape; allocation feasibility is a filter, not a ranking, so k-best then filter replaces both extractors, and the critic's "cost known only after allocation" argument supports a k greater than one rather than two strategies.
Concession: the end state is Option B plus E, as they say; the critic's point is an argument for k and for a re-rank, not for keeping the slow extractor.
Concession: the slow extractor can stay as a test-only cross-check (Opus).
Where I still differ is sequencing, which is the maintainer's concern.
The merge changes `--log` file order and drops candidates (a behavioral step), so it waits for the Q4.2 gate and B9.
Before that, module merge and removal of the duplicated post-saturation pass (pipeline.rs:544-547) are structural and pass the structural gate.
A ranking through the cost ABI before allocation is not possible for the default model: cpp/cost/count.cc counts `api.` lines of assembly, which exist only after allocation; so the proxy ranks and the ABI re-ranks.
Final position: structural merge now; then one k-best extractor ranked by a node-count proxy, re-ranked by the cost ABI after allocation, with content dedup including buffers; k measured on the examples.
Confidence: medium.
Still divides the panel: whether the ABI can rank before allocation (Fable's "where it is cheap"); the default model cannot.

### Q4.6
Strongest opposing argument (Opus and Fable): shipping the equality before the copy leaves a window where a live marked input is overwritten, which violates decision 16; and the copy's cost must be visible to the extractor, so B30 and B11 land as one item.
Rebuttal, from code I read this session: the window does not exist.
`fix_inplace` (G/src/isel/extractor/inplace_fix.rs:9-18) runs inside both extractors (fast.rs:196, slow.rs:281) before allocation.
It gives a copy to every overwritten child that has more than one use (inplace_fix.rs:64-72, `count_uses(pii, child) > 1` then `copy_node`), and returns Err for a child pinned in HBM (inplace_fix.rs:66-68), which drops the candidate.
So a candidate reaching the allocator never has a shared overwritten input, and an equality constraint on offsets (B30) is safe on top of the current fix-up.
My round-1 step 4 ("a candidate whose input is live is dropped in between") overstated the gap: the drop already exists, and the copy is already made.
What B11 adds is real but different: the copy is made after the extractor chose the tree, so its cost does not take part in selection (Opus point 3 and 4).
That is a cost-quality improvement, not a correctness prerequisite, so it can follow B30 without a window.
Concession: shipping the two together is harmless if B11 is small; Fable's fallback (Option E with a drop reason) is what exists today.
Final position: B30 now with `fix_inplace` kept; B11 moves the copy into candidate building as a separate step, gated on the shared-overwritten-input oracle test at tests/test_extraction.py:507.
Confidence: medium.
Still divides the panel: one item (Opus, Fable) against two steps (me); the code shows the intermediate state is correct, so the difference is sequencing and risk.

### Q4.7
Where all three already agree: Option B (read-only `@s.` folded at compile time in v3; updates and state-only emission after; every other use a named rejection).
One difference: Opus adds a named rejection for an instruction with regions and no semantics, and a "not emitted" message for a state-only one; Fable's D12 warning (the writer would compile `@s.n` as attribute `n`) argues the rejection stays loud until each use is supported.
I accept both; the silent drop at act_backend/generator.py:119-122 must become a message first, as a small step needing no state design.
Final position: Option B with the silent drop replaced by a rejection or message now.
Confidence: medium.
Still divides the panel: nothing; it depends on panel 1 for R7 and RA8 before updates.

### Q4.8
Strongest opposing argument (Fable): two hand-written copies of shape rules in Python and Rust are the anti-pattern ODS exists to end; generate one table, and do not ship Option E, which makes two breaking changes for one error.
Opus: B7 needs shape rules in Rust anyway for nodes the rewrites create (slice, concat), so the backend copy is not only a check; `make` can assert that its inferred shape equals the annotated one.
Concession to Opus: that is a stronger form of my Option B assertion, and it needs no table; the assertion is an internal error, not a user check.
Concession to Fable: the end state is one generated table; I moved Option D from "long-run" to "end state".
Rebuttal on Option E (Fable): decision 10 batches breaking changes into one major release, so a backend rejection followed by the same front-end rejection inside that batch is one user-visible break, not two.
Even so, E is worth it only for rules that need no new table, as I said in round 1; the literal check at spec_checks.py:394-418 is such a rule.
Final position: Option B now (the backend asserts typed input, and `make` asserts agreement with annotations); region-count checks move to the front end (Option A); Option D is the end state if the language track owns a table.
Confidence: medium.
Still divides the panel: Opus rejects now and deletes later (E); Fable and I stop at the assertion.

### Q4.9
Where all three already agree: Option B for R4 and R10, and reject a kernel that reads unwritten on-chip storage (Option C's reject branch), not zeroing.
Opus's refinement: the read-before-write check works on the candidate, with a defined drop reason; every read has a producing node, so the case arises only from partial writes and in-place reuse.
Accepted, and it is cheaper than my allocator-side framing.
Strongest opposing argument on integers (Fable): the oracle-differential gate of Q4.2 compares two executors, so both must use one number system (H4, i64 per intermediate); my "exact in the oracle, i64-checked in the backend" produces a case where the oracle succeeds and the backend errors.
Concession: that case is a false failure in the gate unless the gate tolerates a defined backend error, and tolerance weakens the gate.
Evidence the backend is i64-checked: act_backend/expression_writer.py:72,94 use `checked_add` and `checked_mul`.
The choice belongs to panel 1 (Opus: the backend does not force it); the backend's obligation is only that an overflow is a defined error, which it already is.
Final position: Option B for bounds; reject read-before-write on the candidate; i64 per intermediate in the backend with the norm owned by panel 1, and the gate comparing only runs where both succeed or both error.
Confidence: medium.
Still divides the panel: Fable wants i64 in the oracle too; I leave that to panel 1.

### Q4.10
Strongest opposing argument (Opus): `Runner::default()` already uses egg's `BackoffScheduler`, so my Option C stopgap is already in place and does not fix the node-limit stop; the fix is to guard the bare-variable rules at the source.
Verification: G/src/pipeline.rs:491 builds `Runner::default()` and adds no `with_scheduler`, so the default applies; egg documents the default scheduler as `BackoffScheduler`.
Concession: my round-1 steps 2 and 3 (backoff as a one-line stopgap) are void; it is already in use.
Rules 2, 3, 4, and 7 in ir2ir_rewrites.txt (`(?x) => (dot_? ?x eye_?)`, `reshape` pair, `bitcvt` pair, `copy`) have a bare-variable left-hand side and match every class.
Opposing argument (Fable): phases remove those rules from the selection phase, and extraction once per phase end makes `N` = 1 (lib.rs:24) moot.
Rebuttal: phases are a behavioral step that can lose a candidate which interleaved firing finds (B32 caveat), so they need the B29 measurement; guards on the rules are a smaller step and may remove the need.
Agreement: Fable and I both fix rule 0's constants (tile 16, `I32` at ir2ir_rewrites.rs:20-27,100), which is target correctness in a generic crate.
Final position: Option A plus E; guard the bare-variable rules on the ISA having a consumer of the produced operator; fix rule 0 from the spec; phases (B32) only if B29 shows node-limit stops remain; no egglog.
Confidence: medium (was medium; the content changed).
Still divides the panel: phases now (Fable) or after measurement (Opus, me).

### Q4.11
Strongest opposing argument (Fable): emit one typed struct from the B13 model for both Rust and C++ so a single schema exists; Opus: derive the C header from the Rust struct, as cbindgen does.
Evidence: the generic crate has no cbindgen (G/Cargo.toml lists egg, itertools, libc, serde, serde_json), and engineering.md says not to add a dependency without asking.
The boundary is already `extern "C"` on both sides (G/src/malloc/cpp.rs:15; G/cpp/malloc/include/act_malloc.h:14), and the generator already emits a C++ header (act_backend/malloc_instructions_h_generator.py).
Concession to Fable: emitting both sides from the generator needs no new dependency and fits Option B; I accept it if the call's fields are generic (nodes, children, shapes, attributes) and not per-accelerator.
Rebuttal to Option C-only (hardening the text): B25 (`assert(false)` to error returns) should land anyway, as the cheap first step before the struct.
Final position: Option A plus E; one typed `#[repr(C)]` call with its C++ declaration emitted from the generator or hand-kept with a size and offset assertion test; no cbindgen without owner approval; B13 built from `instruction.py` and re-pointed when R42 lands; the Rust printer stays for `--log`.
Confidence: medium.
Still divides the panel: how the schema is shared (generator, cbindgen, or hand-kept); Opus and Fable want the schema derived.

### Q4.12
Strongest opposing argument (Fable and Opus): generated output is not committed, so no engineering rule requires it formatted; TableGen and ISLE emit unformatted code; running rustfmt at generation adds a tool to the user's generation path for no reader; width twins re-implement rustfmt badly.
Verification: README.md:110 shows the CI format check covers `git ls-files` only, and tests/lint.py:19-27 runs `cargo clippy ... -D warnings` on the generated crate, which checks meaning and not layout.
So nothing consumes the generated layout.
Concession: my round-1 argument that rustfmt is already a dependency is true for the repo's CI but not for a user who only generates; a rustfmt call at generation adds a hard dependency to the generation path.
Option A removes the twin logic (`TWINS`, `fits`, `LINE_WIDTH` in act_backend/template_loader.py:10-20) with no new tool, which is the larger saving.
A reader who wants the crate formatted runs `cargo fmt` on the output.
The Q4.2 gate (identical assembly) proves twin deletion, since only layout changes.
Final position: Option A applied under Option E, only to emitters that survive the Q4.4 decision.
Confidence: medium (was medium).
Still divides the panel: nothing; Opus's and Fable's condition (an owner preference to read generated crates in review) would reopen it.
