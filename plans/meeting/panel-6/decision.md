# Panel 6 decision: test suite
Moderator: Sonnet.
Rulings rest on code read this session and on Test262 INTERPRETING.md and the WebAssembly interpreter README, both fetched this session.
Owner decisions 16, 17, 22, 23, 24, and 25 bind every ruling; decision 23 means no test design names JAX.

| Question | Decision (one line) | Confidence | Owner? |
| -------- | ------------------- | ---------- | ------ |
| Q6.1 | Option A: `program.toml`, executor-neutral, matched on clause id | high | no |
| Q6.2 | Option A with the mark kept; cross-buffer seeds become rejected cases; storage reuse checked on assembly | medium | no |
| Q6.3 | Options A and C; outcome named `fault`; backend never faults | high | no |
| Q6.4 | Options A and D by S1.3 id; B for contested rows; a pending row blocks release | medium | no |
| Q6.5 | Options A, B, D, E, F; option C rejected | high | no |
| Q6.6 | Stored data from NumPy or closed forms; tolerance D, F, G; interpreter only a cross-check | medium | yes, interpreter |
| Q6.7 | Options A, C (strict), D | high | no |
| Q6.8 | Options A and C plus a mapped-deletion check | high | no |
| Q6.9 | Options B, C, D; E for corpus cases only | medium | no |
| Q6.10 | Hypothesis per PR, nightly fuzzer, per-release mutation pilot, all behind approval | medium | yes |
| Q6.11 | Options A, C, E, F, G | high; medium on C | no |
| Q6.12 | Options A and C; D only while the oracle lowers to XLA; E rejected | medium | no |
## Q6.1 Run artifact for instruction-level cases
Decision: option A.
- A case needing attribute values, addresses, or state carries `program.toml`: initial buffers and state, instruction calls, and expected buffers and state or a fault clause id.
- `kernel.hlo` and `inputs.npz` stay for the backend and T13.
- The runner uses an oracle interface, never JAX, and matches clause id and outcome kind, never the message.
Reasoning:
1. Decision 16 gives every violation a verdict the corpus must state, and an HLO kernel cannot set attributes, addresses, or state, so it cannot reach S1.3.1 to S1.3.10.
2. The seeds are call programs (act-oracle/tests/test_kernel.py:590-598, :1229-1240); option B leaves them where a second oracle cannot run them.
3. The seeds compare the whole message with `==` (:598, :1240) and raise at `k('fsim-compile')`, a JAX-path step, so a case names the outcome and clause, not when the oracle detects it.
Evidence: items above; plans/tests/corpus/format.md:8, :11; Wasm README (action plus `assert_trap`).
Dissent: none.
Confidence: high.
## Q6.2 In-place and aliasing cases
Decision: option A, mark kept as the declaration under test, backend half on assembly.
- Decision 22 keeps `set_inplace`; the corpus keys on the mark.
- A seed marking an input in another buffer than the output becomes a `rejected` case (decision 24).
- Unmarked-overlap and partial-overlap cases wait for a verdict (Q6.4).
Reasoning:
1. HLO is value-semantic, so T13's "read the marked input after the instruction" (plans/tests/generators.md:50) is not a kernel.
2. Spec-draft S5.5 requires the input in output 0's buffer with equal sizes (spec-draft/conformance.md:30) and equal starts at issue (S1.3.6, overview.md:42).
3. The seed act-backend/tests/test_extraction.py:507 reads `d0` and writes `d2`, marked; decision 24 makes it a `rejected` case.
4. Same-buffer seeds stay: `exp_in_place` (test_extraction.py:389-395) and `add_in_place` (:413).
5. The mark is not redundant for the backend: `inplace_fix.rs:59-62` reads `inplace_children()` and copies an overwritten operand that has another reader.
6. Equal input and output regions are one region and a call reads every input before it writes (semantics/rulings.md:8-13), so a read after the call gives the output marked or not; the debaters' oracle case cannot tell them apart.
7. Oracle cases are therefore: marked equals unmarked result; S1.3.6 faults with unchanged state; the S9 rows at conformance.md:29-31 are `rejected`.
8. Decision 17's storage reuse is observable on the backend: a check line asserts output address equals marked input address; an HLO kernel with the input live (test_extraction.py:461-468) covers clobbering.
9. The e-graph mark fact (test_generators.py:243) stays a per-PR `impl` test.
Dissent: Fable and Opus round 1 dropped the mark or used the derived form; both withdrew.
Confidence: medium; the oracle has no aliasing model and the address check needs deterministic assembly (Q6.9).
## Q6.3 Fault cases
Decision: options A and C together; the outcome is named `fault`.
- A fault case is an oracle-only program with `expect = "fault"`, a clause id, and unchanged buffers and state.
- T13 gains one invariant: simulating a supported kernel's assembly never faults; its "silent result is a failure" rule is dropped.
Reasoning:
1. R55 defines a fault as writing no region and no state, so a raise-only check tests half of it.
2. Audit rows R4 and R10 say an executor faults and a producer never emits, so T13's rule (plans/tests/generators.md:49) fails a conforming backend.
3. A faulting call is not expressible in HLO, so option B keeps a rule no runner executes.
4. The snapshot compare follows riscv-arch-test's whole-signature compare; `assert_trap` alone is weaker.
Dissent: none.
Confidence: high.
## Q6.4 Norm catalogue
Decision: option A keyed by S1.3 ids, option D, and option B for contested rows only.
- The norm table is the S1.3 table (spec-draft/overview.md:37-51) plus a `cases` column and one expectation per generator.
- Cases now: S1.3.1 to S1.3.5, S1.3.7 to S1.3.14, and S1.3.6 (settled by decisions 22 and 24).
- A `pending` row with the ruling id and no case: unmarked and partial overlap, R7, R8, R12, R56, S1.3.15.
- A `pending` row fails the v3 release gate, since decision 16 requires a verdict for every violation.
- An `implementation-defined` case carries an allowed set or tolerance; the seeds at test_kernel.py:712-720 and :749-757 move to oracle `cli` tests.
Reasoning:
1. S1.3 already numbers every fault, so a ruling-keyed catalogue would drift from it.
2. The audit marks R5, R7, R8, R12, and R56 questionable or wrong, and expectation rows key on case ids (generators.md:80), so early cases risk false reds and pin bumps under decision 11's capacity.
3. The release gate answers Opus's worry that contested clauses regress unseen.
4. A case with no expectation tests nothing (QA17); Test262 flags multi-path tests `non-deterministic` and Wasm uses `nan:arithmetic`.
Dissent: Opus round 1 wanted cases now for contested rows; round 2 accepted option B for them.
Confidence: medium.
## Q6.5 Rejection verdict
Decision: options A, B, D, E, F; option C rejected.
- Required: `phase` (`lex`, `parse`, `build`, `generate`, `run`) and `clause`; `message` is an optional regex.
- `error_type` is required for front-end phases only; a `run` fault matches on clause id, since a C/C++ oracle has no Python class (decision 23).
- `clause = "pending"` is allowed until L numbers S9; CI fails on it once the list ships.
- An optional `expect` table keeps parsed-value facts (test_grammar.py:352-357); lex and parse collapse if they cannot be separated.
Reasoning:
1. Test262 fails a negative test with a different phase or constructor name and states no message match.
2. The Wasm interpreter checks the failure string only as a prefix and calls it documentation.
3. T6 lists rewording as its fail point (corpus/format.md:20-29), and R44 puts the clause id in the message.
4. `value.NUMBER().getText() == text` (test_grammar.py:357) is not a verdict, so T7 loses it without option D.
Dissent: none.
Confidence: high.
## Q6.6 Reference, tolerance, comparison
Decision: expected values are stored data in each case.
- v3 regenerates them from NumPy or closed forms, named with a pinned version in `case.toml`; JAX is not a reference.
- Tolerance per case: `expect = "exact"` or `"almost"` with `ulps`, the bound a function of type and K (D); the spec owns the bound, `taidl.testing` implements it, the conftest imports it (F).
- NaN, signed zero, and infinity comparison is stated once in `taidl.testing`, with optional `nan = "any"` (G).
Reasoning:
1. Stored values keep any reference tool off the per-PR path; audit R29 finds the interpreter not on PyPI, needing CHLO legalization, and possibly off R6's pin.
2. A new dependency needs owner approval (R29, R45); NumPy references exist (act-oracle/tests/test_reduce.py:87-129).
3. A JAX-computed reference shares the oracle's lowering, and decision 23 moves the oracle off JAX.
4. `ULPS` is one value per type (act-oracle/tests/conftest.py:226-227), scoped to a transcendental word, so it cannot bound a K-term sum (audit R8); decision 23 judges limits against fsim-lib, so the bound moves to the spec (RA4).
Dissent: Sonnet and Fable each withdrew a round-1 position.
Confidence: medium.
Owner (interpreter role only):
- Cross-check only: it catches a wrong NumPy reference without a per-PR dependency or R6 pin clash.
- Normative regenerator: it is the executable StableHLO reference, so values follow the spec by construction.
- Moderator view: cross-check first, promote after a measured match on every S8 word.
## Q6.7 Coverage and row kinds
Decision: options A, C, and D.
- `taidl/conformance/coverage.md` is rendered per S id, S9 clause, and norm row; CI fails on a zero row outside an allowlist (S1.1, S1.2, S10, open rows).
- `xfail = "<id>"` runs the case and fails on XPASS; rows exist only for front-end-accepted cases.
Reasoning:
1. A per-section count hides a clause with no case (corpus/sections.md:33), and R9 fits neither `supported` nor `rejected` (generators.md:19-25).
2. LLVM lit XFAIL fails when it starts passing, while UNSUPPORTED skips silently, so a skip never reports a fix.
3. T12 never runs a rejected case, so its row is dead data; the `pending` clause of Q6.5 keeps the allowlist short.
Dissent: none.
Confidence: high.
## Q6.8 Migration order and deletion safety
Decision: options A and C, plus a mapped-deletion check.
- A test is deleted only in the step whose runner covers its fact at the same cadence; the S8 word and type cases land before T11; deletions are named by function.
- A script fails a PR in which a test id disappears without a mapped successor.
- The arity check at act-backend/tests/test_refusals.py:164-176 stays an `impl` test until the T12 slow tier runs per PR.
- Tests of removed constructs become `rejected` cases naming the replacement and clause id (RA11): `set_apart` marks (test_refusals.py:383-396) and the linalg words (act-oracle/tests/test_linalg.py:1).
- `set_inplace` tests convert to the v3 API (decision 22); only cross-buffer pairings become `rejected` (decision 24).
- `LOWERED` is replaced by a corpus runner over S8 words, not a JAX lowering table (decision 23).
Reasoning:
1. Step 6 deletes `LOWERED` before the step 8 runner, and act-oracle/tests/test_refusals.py:44-45 asserts on it.
2. T4's range (`test_generators.py:175-393`) holds the one test showing a mark reaches the e-graph (:243).
3. Deleting tests of removed constructs loses the fact that they are refused; decision 10 allows the break without a converter, so the error is the migration aid.
Dissent: none.
Confidence: high.
## Q6.9 CI tiers, budgets, determinism
Decision: options B, C, D, and E for corpus cases only.
- A per-PR slow tier in act-backend runs one build and one oracle-simulated kernel per S8 word, beside the nightly differential tier.
- Budgets are proposals fixed after a week of measurement; oracle-only tiers are re-budgeted against fsim-lib (decision 23).
- T13 to T15 run in R17's deterministic mode; flaky tests are quarantined in one PR with an owner and a removal date, never rerun until green.
- The R17 gate stays byte identity of assembly and candidate logs, via a same-job double run per architecture; corpus-side assembly facts are check lines.

| Tier | Trigger | Budget | Overrun action |
| ---- | ------- | ------ | -------------- |
| taidl | pull request | 5 min | split the job |
| generator fast | pull request | 10 min | move cases to slow |
| generator slow | pull request, changed paths | 20 min | quarantine, then nightly |
| act differential | nightly | 90 min | ticket with owner |
| release | tag | 3 h | block the tag |
Reasoning:
1. R49 makes differential tests nightly, so a deleted per-PR test needs a per-PR replacement (QA7); T12's slow tier has no trigger or limit.
2. The search stops on a 1000 ms wall-clock `TIME_LIMIT` (act-backend/act_backend/generic/src/lib.rs:18), so goldens are unstable before R17's mode; goldens are exact (test_mini_trn.py:39-42) and amd64 versus arm64 equality is unproven.
3. Check lines detect less than byte identity, so they cannot be R17's gate; byte goldens break on benign renames, so the corpus uses check lines.
Dissent: Opus round 1 rejected option E and conceded for stored goldens.
Confidence: medium; no budget is measured.
## Q6.10 Fuzzing, mutation, property tests
Decision (owner): recommend A per PR, B nightly, C as a per-release pilot, all behind approval of the new tools.
- A: Hypothesis, `derandomize` with a fixed seed per PR; shrunk cases promoted with `origin = property:<seed>`.
- B: nightly grammar fuzzer, property "accepted, or rejected with a clause id"; it starts after the Q6.5 runner exists.
- C: mutation over the front-end and oracle fault checks, each surviving mutant getting a case or a written reason; never a PR gate.
Reasoning:
1. T9 uses well-formed strategies only (corpus/extensions.md:8), and S9 says a conforming front end rejects exactly the listed descriptions (spec-draft/conformance.md:6), which is the fuzz property.
2. Sufficiency is a claim without mutation; a run executes the suite per mutant, so scope stays narrow.
3. Hypothesis and a mutation tool are new dependencies (R45).
Dissent: Fable wanted B at once; the runner dependency delays it.
Confidence: medium.
Owner options:
- Approve all: the fuzzer tests the S9 contract on unwritten inputs, and mutation is the only sufficiency measure.
- Hypothesis only: decision 11 caps capacity, and a mutation pilot costs time plus a dependency.
- None (option D): no new dependency; sufficiency stays a claim.
## Q6.11 Corpus home, format, layout
Decision: options A, C, E, F, and G.
- The corpus lives in the taidl wheel, read through the pin with T16's override; the kit splits per R43 (loader and tolerances in `taidl.testing`; goldens and the oracle import in `act/testkit`).
- Cases stay `spec.py` plus text `case.toml`, `semantics.txt`, `kernel.hlo`, `program.toml`, readable by a non-Python oracle (decision 23).
- Directories are slugs grouped by outline file; the S id lives in `case.toml`.
- `CHANGES.md` lists added and removed case ids, and `check_versions.py` prints the delta on a pin bump.
Reasoning:
1. One wheel gives language and tests one version, which Q6.4's one-PR rule needs; a fifth repo adds a pin.
2. Goldens depend on the oracle and the taidl wheel must not (R43); `Accelerator` forces `spec.py` (taidl/tests/test_instruction.py:10).
3. Expectation rows key on directory names (generators.md:80), so renumbering must not rename them; a silent case-set change alters what "conforming" means.
Dissent: none; Fable's trigger for a corpus repo (a second implementation outside the four repos) is unmet.
Confidence: high on A, F, G; medium on C.
## Q6.12 Real ISA descriptions, goldens, GPU
Decision: options A and C; option D conditional; option E rejected.
- R46, R47, R48 stand; T14 skips and reports a kernel that an expectations row marks `rejected`, with T10's row naming the S8 words and S5 features it needs.
- Golden updates follow QA15: a PR that names the semantic change and shows the record diff, on CPU only.
- `precision_config` HIGHEST on every `dot` (D) holds only while the oracle lowers to XLA; the durable rule is the spec's dot bound (S1.3.15) driving the GPU tolerance.
- GPU is no bitwise golden target; hosted runners have no GPU, so GPU checks are manual.
Reasoning:
1. The smoke test is the only per-PR signal that a generator builds its example (act-backend/tests/test_mini_trn.py:26-30); the GPU skip exists (act-oracle/tests/test_mini_trn.py:22).
2. Audit RA5: dropping the matmul-precision flag may let an f32 dot run as TF32; the `dot_general` `precision_config` values were not verified (the fetched StableHLO page was truncated), so D carries that task.
3. Under decision 23 the oracle's dot is fsim-lib's, so an XLA flag is transitional.
Dissent: none.
Confidence: medium.
## Escalation to the owner
Copied from the debaters:
- Sonnet round 2: decisions 16 and 22 together require a stated norm for an unmarked overlap of an input and an output at issue.
  - The suite can encode a fault (semantics/rulings.md:8-13) or an allowed set of results, never "unspecified".
  - The row stays `pending` until stated, and a `pending` row blocks the v3 release.
- Opus round 2: none.
- Fable round 2: none for the owner; a point for panel 1 (R5 removes `set_inplace` while decision 22 keeps the mark); decision 24 settles the cross-buffer seed at test_extraction.py:507.
Added by the moderator:
- New dependencies need approval: `hypothesis` (R45), a mutation tool (Q6.10), the StableHLO interpreter (Q6.6, R29); options sit in those sections.
- For equal regions the mark changes no oracle-visible value (Q6.2 step 6), so decision 17's promise is testable only on backend assembly.
  - This bears on decision 22's "unless truly redundant"; it is an input, not a counter to a decision.
