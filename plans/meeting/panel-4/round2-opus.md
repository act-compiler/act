# Panel 4 round 2: Opus (e-graph and instruction selection lens)

### Q4.1
All three hold Option B: overlap allowed, cost chooses, ties broken by declaration order.
- Sonnet adds a generation lint for exact duplicates (D); I accept it as a warning, since a duplicate adds e-nodes and no choice.
- One factual split on ISLE: Sonnet cites the language reference, which says equal-priority overlap is allowed.
- The ISLE implementation is stricter: `check_pair` in `overlap.rs` reports "rules are overlapping" when two rules share a priority.
- The same file exempts multi-constructors: "the ctor returns every match ... so overlap does not actually affect the results".
- An e-graph rewrite set is the multi-constructor case, so both readings support B.
Final position: Option B plus D as a warning; the tie key is a rule declaration index stored on the ISA e-node.
Confidence: high.
Still divides: nothing of substance.

### Q4.2
All three hold Option B.
- Strongest variation (Fable): make the deterministic mode the only CI mode rather than keeping two modes that must agree (E).
- I concede: the agreement check is a one-time step before goldens are re-recorded, and the wall-clock mode is never gated after that.
- Sonnet's one-time re-record in its own commit is the right mechanics.
Final position: Option B; deterministic mode is the CI mode; one equivalence run, then goldens re-recorded once if needed.
Confidence: medium-high.
Still divides: the cost of the oracle run in CI, which nobody has measured.

### Q4.3
All three hold Option A order with F (`Option` shape) and D (constants in class data, node emitted at candidate building).
- Fable would admit C (attributes as child e-nodes) for attributes patterns must match, such as slice offsets.
- Rebuttal: egg's `Subst` binds pattern variables to class `Id`s, so an attribute child becomes a class that rewrites can union with unrelated integers.
- A typed payload compared in a condition gives the same match without that hazard.
Final position: Option A with F and D; C rejected.
Confidence: high.
Still divides: Fable's narrow C scope only.

### Q4.4
Strongest opposing argument (Fable): every mature selector compiles rules to code; generated Rust gets rustc and clippy for free, while a data applier needs its own evaluator, checks, and errors.
- Sonnet holds D but only after a two-rule spike, with C as the fallback.
- Evidence that moves me: the generator already compiles conditions to Rust, with conjunction and comparison visitors (`isel_ir2isa_rewrites_rs_generator.py:442-528`) and checked i64 arithmetic in `expression_writer.py`.
- So B or C is the smaller delta: replace the hand-indexed `precond_N`, `metadata_N`, `set_shapes_N` tables with one generated value per rule binding variables by name.
- D instead writes a data format, an evaluator, and its error paths, and deletes working emitters; decision 11 weighs against that.
- egg's own idiom also attaches conditions as Rust functions on a `ConditionalApplier`, so generated code is not foreign to egg.
- My round-1 point stands that code generation buys no matching speed here, but it no longer decides the question.
- Rebuttal to Fable step 3: a generated rule cannot read matched e-nodes from `Subst`, which holds class `Id`s only.
- The re-match in `applier.rs:106-113` (`find_matches`, `applier.rs:168`) therefore stays as one shared helper in either form.
- B8 must name IR-to-IR rules first, so IR-to-IR stays hand-written named Rust.
Final position: Option C (changed from D): IR-to-IR hand-written named Rust; IR-to-ISA generated Rust bound by name, with one shared re-match helper.
Confidence: medium.
Still divides: Sonnet's spike for D; whether `find_matches` survives (Fable says no, I say yes).

### Q4.5
Strongest opposing argument (Sonnet): the merge changes selection and `--log` order, so it is behavioral and must wait for B9 and the deterministic gate; merge the files first (C) and land k-best later.
- I concede the sequencing: k-best lands after B29 and B9, gated by Q4.2's oracle run.
- I also concede Sonnet's early step: dropping the re-extraction of an unchanged e-graph after saturation (`pipeline.rs:544-547`) is behavior-preserving and can land at once.
- The end state is unchanged: Fable's point that allocation is a filter and not a ranking makes k-best then filter the standard shape.
Final position: Option B plus E as the end state, sequenced after the gate; the duplicate pass goes first.
Confidence: medium.
Still divides: whether C as an intermediate state is worth a commit, and how k is chosen (Sonnet: measured on examples).

### Q4.6
Strongest opposing argument (Sonnet): B30 can land first, keeping `fix_inplace` until B11, with a drop reason for what it cannot fix.
- Fable argued the opposite: equality before the copy leaves a window where a live marked input is overwritten.
- The code decides this: `fix_inplace` already gives an overwritten shared operand its own copy (`inplace_fix.rs:6-17`).
- It returns `Err` where the operand is pinned in HBM or views shared storage, so those cases are already defined drops.
- Both extractors call it before allocation (`fast.rs:196`, `slow.rs:281`), so with B30 the solver sees the copy and no live input is overwritten.
- My round-1 claim that a split drops accepted candidates was wrong for the same reason; equality only adds infeasible allocations, each a solver drop with a reason.
- B11 is still needed so the copy's cost is visible to the extractor, but that is a cost question, not a correctness one.
Final position: Option E (changed from B): B30 first with `fix_inplace` kept, B11 later moves the copy to candidate building.
Confidence: medium-high.
Still divides: Fable holds that the two must land together.

### Q4.7
All three hold Option B: fold read-only `@s.` at compile time in v3; updates and state-only emission after v3.
- All three also make every unsupported use a named rejection, and turn the silent drop at `generator.py:119-122` into a message now.
- Fable's D12 point (`@s.n` compiled as attribute `n` on the `@x.` path) is a reason to keep the rejection loud; I adopt it.
Final position: Option B with E's rejection half.
Confidence: medium.
Still divides: nothing; timing depends on panel 1 ruling R7 and RA8.

### Q4.8
Strongest opposing argument (Fable): reject-now-then-delete (E) ships one error twice, and one generated table (D) is what ODS does.
- Sonnet and Fable both make D the end state and B (assert typed input) the contract now.
- Rebuttal: decision 16 makes today's silent never-selected semantics a defect now, and the front-end table is not scheduled.
- A backend rejection and a later front-end rejection of the same spec give the user the same outcome, so E is not a second breaking change.
- I concede D as the end state if the language track owns it, since B7 shape rules and front-end rules are one function.
Final position: B now with E as the fallback until the front end lands its check; D as the end state.
Confidence: medium.
Still divides: whether the backend rejects in the v3 batch (me) or only asserts (Fable).

### Q4.9
All three hold Option B for R4 and R10.
- Strongest variation on unwritten storage (Sonnet, Fable): reject a read of unwritten on-chip storage rather than zero it; that matches my candidate check.
- Integers split: Sonnet documents exact-in-oracle versus i64-in-backend; Fable wants i64 in both.
- Fable's argument wins on my lens: the Q4.2 differential gate compares the two executors, so one norm in both avoids a known false alarm.
- Sonnet's step 8 adds that the C++ allocator holds 64 bits, so i128 in generated Rust would not remove the limit end to end.
Final position: B; reject read-before-write on the candidate; i64 per intermediate (H4) in both executors, unless panel 1 rules otherwise.
Confidence: medium.
Still divides: panel 1 owns the integer norm.

### Q4.10
Strongest opposing argument (Sonnet): `BackoffScheduler` is a one-line runner setting and the stopgap before phases.
- Rebuttal: egg's documentation states "BackoffScheduler is the default scheduler", and `pipeline.rs:491` builds `Runner::default()`.
- So backoff is already on, and saturation still ends at `NODE_LIMIT`; C's scheduler half is not a remedy.
- Fable and Sonnet favor phases (B32); I accept phases as the end state, since a selection phase without the bare-variable rules removes them by construction.
- Guarding the bare-variable rules on a consuming ISA instruction stays the cheaper first step, measured under B29.
- All three agree on E for rule 0 and on keeping egg.
Final position: A and E, guards first, then B32 phases once B29 measures them; no egglog.
Confidence: high.
Still divides: whether guards alone suffice (me) or phases are needed regardless (Fable).

### Q4.11
Strongest opposing argument (Fable): emit one schema for Rust and C++ from the B13 model (B), as cbindgen and ODS do.
- Rebuttal: B couples the FFI to B13, which is itself waiting on R42.
- A struct written in Rust with its header produced by cbindgen gives one schema with no B13 dependency, which is Fable's own cited tool.
- Sonnet's round-trip test for ABI drift is cheap and I adopt it.
- All three agree on E (B13 from today's `instruction.py`) and on keeping `hold` and `hold_tensors` in C++.
Final position: A with E and F (B5 redefined after B7); one schema from the Rust struct.
Confidence: medium.
Still divides: where the schema originates (Rust struct versus B13 model).

### Q4.12
Strongest opposing argument (Sonnet): delete twins and run rustfmt at generation (B), since rustfmt is already present.
- Rebuttal: generated output is not committed and no gate checks its layout (`tests/lint.py:19,27`), so a formatter run has no consumer.
- Fable agrees on A; Sonnet makes B the default only if Q4.4 leaves many emitters.
- Under my Q4.4 change to generated Rust, more emitters survive, which strengthens the case for deleting width logic rather than keeping it.
Final position: Option A under E, unchanged.
Confidence: medium.
Still divides: Sonnet's rustfmt run at generation.
