# Panel 6 agenda: test suite

Scope: conformance corpus, expectations, differential tests, migration.
Blockers and majors come first; findings that need no debate close the file.
Sources are cited by ID: T items in plans/tests/, QA and IN findings in plans/reviews/tests/, R and RA rows in plans/minutes/rulings.md and plans/reviews/rulings/audit.md, owner decisions by number.
A draft answer is one option among others.

## Questions

| Q | Topic | Severity of the source findings |
| - | ----- | ------------------------------- |
| Q6.1 | Run artifact for instruction-level cases | blocker |
| Q6.2 | In-place and aliasing cases | blocker |
| Q6.3 | Fault cases: contract and differential role | major |
| Q6.4 | Norm catalogue and its dependence on open behavior rulings | major |
| Q6.5 | Rejection verdict: phase and match contract | major |
| Q6.6 | Reference library, tolerance, and comparison rules | major |
| Q6.7 | Coverage enforcement and expectation row kinds | major |
| Q6.8 | Migration order and deletion safety | major |
| Q6.9 | CI tiers, budgets, cadence, and determinism | major |
| Q6.10 | Fuzzing, mutation testing, and property tests | major |
| Q6.11 | Corpus home, case format, and layout | major |
| Q6.12 | Real ISA descriptions, goldens, and the GPU | major |

### Q6.1: What artifact carries an expected run outcome for cases that need instruction calls, attribute values, addresses, or state?

- Context:
  - T5 attaches run outcomes only to `kernel.hlo` plus `inputs.npz`.
  - An HLO kernel cannot set attribute values, addresses, or state, so it cannot trigger an out-of-bounds region, a violated constraint, an addressing overflow, or a backtick literal fault.
  - The seeds T8 names are instruction-call programs, not HLO.
- Options:
  - A: add a second run artifact to T5 (for example `program.toml`) holding initial buffers and state, a list of instruction calls with attribute values, and the expected buffers and state or the fault clause id; the oracle runs it directly and `kernel.hlo` stays for the backend and T13.
  - B: keep `kernel.hlo` and `inputs.npz` as the only run artifact; run-time norm cases that HLO cannot express stay as per-generator tests outside the corpus.
- Evidence pointers:
  - QA1; QA4; T5 (corpus/format.md:8, :11); T8 (corpus/seeding.md:25).
  - act-oracle/tests/test_kernel.py:590-598, :1229-1240.
- Binding constraints:
  - Owner decision 16: every run-time violation is a defined error or a defined behavior, so each has a verdict the corpus can state.

### Q6.2: How are in-place and aliasing behaviors tested in the oracle and in the backend?

- Context:
  - HLO is value-semantic, so a kernel cannot read a marked input after the instruction and see the output, which is what T13 and T8 describe.
  - The named seeds use `set_inplace` (removed by R5) and `set_apart` or `set_no_partial_overlap` (removed by R58).
  - R5 derives aliasing from identical region texts; the audit marks R5 questionable for equal regions spelled differently and for partial overlap (RA2).
- Options:
  - A: oracle side through Q6.1 programs (identical region texts alias, a read after the call sees the output, a partial overlap gets its defined verdict); backend side as HLO kernels where the aliased input is still live afterwards, checked by simulating the backend's assembly with aliasing modelled.
  - B: keep T13's wording (both generators show the output in the marked input's storage, read back after the instruction) and restate it with the R5 derivation in place of the mark.
  - C: test the oracle side only in the corpus; the backend's in-place behavior stays in `impl` tests such as the e-graph and allocator tests.
- Evidence pointers:
  - QA2; QA8; T13 (generators.md:50); T8 (corpus/seeding.md:25); corpus/sections.md:22 (S5.5).
  - taidl/tests/test_instruction.py:273-291; act-backend/tests/test_refusals.py:383-396; act-backend/tests/test_extraction.py:441-563, :507.
  - Audit rows R5, R21 (B11 copies), R22; RA2, RA3.
- Binding constraints:
  - Owner decision 17: the output reuses the marked input's storage.
  - Owner decision 16: an overlap that is not an alias needs a defined error or a defined behavior.

### Q6.3: What does a fault case assert, and what role does it play in the differential tests?

- Context:
  - T13 fails an `error` case on a "silent result on either side", while R4 and R10 let the backend assume the violation never occurs; the audit marks that wording questionable.
  - R55 defines a fault as writing no region and no state, yet T5 and T8 check only that the defined error is raised.
  - A faulting call is not expressible in HLO.
- Options:
  - A: fault cases are oracle-only; add one universal T13 invariant that simulating any supported kernel's backend assembly on the oracle never faults.
  - B: keep T13 as written: an `error` case passes when the oracle faults on the backend's assembly or the backend rejects the kernel with the same S id.
  - C: a fault case states `expect = "fault"` and the runner also asserts buffers and state equal their pre-call values, in the oracle and, through T13, on the backend's assembly where it does not reject.
  - A and C can be taken together; B excludes both.
- Evidence pointers:
  - QA3; QA4; IN7; T13 (generators.md:49); T5 (corpus/format.md:11).
  - R4, R10, R55; audit rows R4, R10 (producer never emits, executor faults); RA2, RA3.
- Binding constraints:
  - Owner decision 16: no undefined behavior in v3.

### Q6.4: Which behavior norms does the corpus encode, given that several rulings are contested?

- Context:
  - QA4 finds no catalogue from each fault and behavior to its cases (R3, R7, R10, R11, R12, R55, R56, R57, and the S1.3 rows).
  - The audit marks R5, R7, R8, R12, and R56 as questionable or wrong against the semantics track (H1, H3, H4, H6), so a case written now may encode a norm that changes.
  - T5 lets an `implementation-defined` case carry no expectation, so it tests nothing (QA17).
- Options:
  - A: add a norm table to corpus/sections.md with one row per ruling or S1.3 row (case ids, kind `result` or fault clause, each generator's expectation); every fault case also asserts unchanged buffers and state.
  - B: write cases only for norms whose rulings the audit leaves standing; cases for contested norms wait for the owning panel's decision.
  - C: write cases against the draft rulings as they stand and revise the cases when the rulings change.
  - D: an `implementation-defined` case carries an allowed set or a tolerance that both generators must meet.
  - D combines with A, B, or C.
- Evidence pointers:
  - QA4; QA17; corpus/sections.md (S1.3 row); T8 (corpus/seeding.md:24).
  - R3, R5, R7, R8, R10, R11, R12, R55, R56, R57; audit rows R5, R7, R8, R12, R56; RA2, RA3, RA7, RA8.
  - plans/semantics/rulings.md:21; plans/semantics/holes.md (H1-H6, H12, H14).
  - act-oracle/tests/test_kernel.py:712-720, :749-757 (kernel layout errors, not TAIDL faults).
- Binding constraints:
  - Owner decision 16: a single simple norm that is easy to explain.

### Q6.5: What does a rejected case record and match: the rejecting phase, a clause id, an error class, or a message regex?

- Context:
  - T5 has one `rejected` verdict with a message regex, and T6 only records the layer that rejects.
  - R44 puts a clause id suffix such as `[S9.4]` in error messages; T5 still makes the regex the contract.
  - T7 deletes grammar blocks whose assertions are parse-tree facts that an accept or reject runner does not check.
- Options:
  - A: a required `phase` field (`lex`, `parse`, `build`, `generate`, `run`) and a runner that fails a case rejected in another phase.
  - B: match on `clause` plus `error_type`, with `message` an optional regex in a few cases.
  - C: keep T5 as written: verdict, S9 clause id, and a message regex, with the layer recorded.
  - D: add an optional `expect` table (parsed value, region shape, element type) that the runner checks; otherwise T7 deletes only blocks whose assertion is a verdict.
  - E: allow `clause = "pending"` so cases land before L numbers the S9 rows, with CI failing on a pending case once the clause list ships.
  - F: rename the run-time `error` outcome to `fault`.
  - A, B, D, E, and F combine with each other; C excludes A and B.
- Evidence pointers:
  - IN1; IN2; IN6; QA9; T5 (corpus/format.md:8, :22); T6; T22 question 2 (R44).
  - taidl/tests/test_grammar.py:352-357; taidl/docs/conformance.md:9-19.
- Binding constraints:
  - Owner decision 10: breaking changes are allowed, batched into one major release (TAIDL v3 and matching generator releases).
  - Owner decision 16: every run-time violation has a defined verdict, so a run-time case states its outcome.

### Q6.6: Which reference library, tolerance form, and comparison rules do S8 cases use?

- Context:
  - T8 takes references from NumPy or closed forms and forbids JAX; R29 rules the StableHLO reference interpreter, with NumPy only for words it lacks.
  - The audit marks R29 questionable (interpreter not on PyPI, needs CHLO legalization, version may differ from R6's pin) and R8 questionable (accumulation clause versus tolerance clause).
  - The plan keeps one `ULPS` per type; RA4 asks who owns the normative tolerance.
- Options:
  - A: the StableHLO reference interpreter for every S8 word it implements and NumPy for the rest, with `reference` and its pinned version in `case.toml`.
  - B: NumPy or closed forms for every case, as T8 states.
  - C: the interpreter pinned to the R6 StableHLO commit, with NumPy as the fallback for the whole reference role.
  - D: tolerance per case, `expect = "exact"` (default) or `expect = "almost"` with `ulps`, with the bound stated in the spec as a function of K (audit R8).
  - E: tolerance as one `ULPS` per element type, moved into the case file.
  - F: tolerance owner is the spec for the bound, `taidl.testing` for the implementation, and the conftest for an import only (RA4).
  - G: state how NaN, signed zero, and infinities compare once in `taidl.testing`, with an optional `nan = "any"` in a case.
  - A through C choose the reference; D and E choose the tolerance form; F and G apply to either.
- Evidence pointers:
  - QA14; IN4; IN8; IN9; T8 (corpus/seeding.md:21, :28); R6, R8, R29, R43.
  - Audit rows R8, R29; RA4, RA5.
  - act-oracle/tests/conftest.py:226-227 (`ULPS`); act-oracle/tests/test_reduce.py:87-129.
- Binding constraints:
  - Owner decision 14: terminology follows the owner's papers first, then StableHLO and MLIR.
  - Rulings.md R29 and R45 record that a new dependency needs owner approval first.

### Q6.7: How is spec coverage enforced, and what row kinds do the expectation tables allow?

- Context:
  - T5 and corpus/sections.md report coverage per section; nothing fails on a zero-case clause.
  - T11 allows only `supported` and `rejected` rows, so a known defect (R9, `ROOT` on a non-last line) fits neither.
  - T11 requires a row for every corpus case id, including cases the front end rejects, which T12 never runs.
- Options:
  - A: render `taidl/conformance/coverage.md` from the corpus per S id, S9 clause, and norm row; CI fails on a zero row unless it is on an allowlist (S1.1, S1.2, S10).
  - B: keep per-section counts as reporting only.
  - C: add an `xfail = "<issue or ruling id>"` row kind; the runner runs the case, fails on XPASS, and the renderer lists it as a known gap.
  - D: derive a per-section status (full, partial, none) from the rows, with rows only for front-end-accepted cases.
  - E: keep `supported` and `rejected` as the only row kinds, with a known defect carried as a skipped case or a rejected row.
  - A and B are alternatives; C and D combine with either; E excludes C.
- Evidence pointers:
  - QA5; QA18; IN3; IN5; T11 (generators.md:19-25); T21; T6.
  - R9; corpus/sections.md:33; inventory/cleanup.md:22.
- Binding constraints:
  - None from the owner decisions beyond decision 16 (every run-time case has a verdict).

### Q6.8: In what order do tests migrate, and when may a test be deleted?

- Context:
  - Migration step 6 deletes `LOWERED` before the T12 runner exists at step 8, and step 8 deletes the word and type loops whose per-case replacements land at step 12 (T8).
  - T4 deletes by line range, and one test in the range is the only unit test showing a mark reaches the e-graph.
  - R49 moves backend coverage from per-PR to a nightly run, and R5, R58, and R60 remove constructs that existing tests and spec pages use.
- Options:
  - A: delete a test only in the step whose runner covers its fact and at the same cadence; land the S8 word and type cases (a slice of T8) before T11; name deleted tests by function; keep the arity check as an `impl` test until the T12 slow tier runs per PR.
  - B: keep the migration steps as T16 to T18 order them.
  - C: tests that use removed constructs are converted to corpus cases of the replacement or `rejected` cases (for example the linalg words as S8.5 `rejected` cases), and a rejected v3 construct reports its replacement with the clause id and no converter (RA11).
  - D: tests that use removed constructs are deleted with the construct.
  - A and B are alternatives; C and D are alternatives and combine with either.
- Evidence pointers:
  - QA6; QA7; QA8; QA16; QA21; RA11; migration.md steps 5-8, 15, 17; overhaul.md steps 5-8; T4, T18, T20.
  - act-oracle/tests/test_refusals.py:44-45; act-backend/tests/test_refusals.py:146-197, :164-176, :383-396; act-backend/tests/test_generators.py:175-393, :243, :393; act-backend/tests/test_mini_trn.py:78.
  - Audit row R58 (tests and spec pages that call the removed marks); R5, R58, R60.
- Binding constraints:
  - Owner decision 10: breaking changes are allowed, batched into one coordinated major release.
  - Owner decision 11: sized for 3-4 engineers over two quarters.

### Q6.9: What are the CI tiers, their budgets, the differential cadence, and the determinism and flakiness rules?

- Context:
  - R49 runs differential tests nightly and on release; QA7 asks that a deleted per-PR test have a per-PR replacement.
  - T12's slow tier builds one cargo crate per section with no trigger or limit.
  - Goldens are exact, but T13 to T15 do not name R17's deterministic search mode; the only nondeterminism rule is the Hypothesis seed.
- Options:
  - A: nightly and on release for the differential tier, as R49 states.
  - B: a per-PR slow tier in act-backend (one build and one oracle-simulated kernel per S8 word) beside the nightly differential tier.
  - C: a table with one row per tier (taidl PR, generator fast tier, generator slow tier, act nightly, release) giving trigger, wall-clock budget, and overrun action.
  - D: T13 to T15 run the backend in R17 deterministic mode, plus a flaky-test rule: quarantine in one PR with an owner and a removal date, and no rerun-until-green.
  - E: assembly goldens as check lines over the assembly text (instruction word and operand captures) for R17 gating, instead of byte goldens.
  - A and B are alternatives; C, D, and E combine with either.
- Evidence pointers:
  - QA7; QA12; QA13; IN13; R17, R49; T12 (generators.md:34); T13; T15.
  - act-backend/tests/test_mini_trn.py:39-42 (`max_diff == 0`); audit row R22 (tie-breaking, deterministic mode).
- Binding constraints:
  - Owner decision 11: capacity of 3-4 engineers over two quarters.

### Q6.10: Which additional techniques (front-end fuzzing, mutation testing, property tests) does the plan include?

- Context:
  - T9 drives the API with well-formed strategies; nothing checks that arbitrary semantics text is accepted or rejected with a clause id.
  - No mutation testing exists, so corpus sufficiency is a claim, not a measurement.
  - `hypothesis` is a new test-only dependency (R45), and a mutation tool would be another.
- Options:
  - A: T9 as written: Hypothesis strategies for ranks, dimensions, element types, and names, with a fixed seed in CI and shrunk counterexamples promoted to corpus cases.
  - B: add a nightly grammar fuzzer to T9 (Hypothesis over the `.g4` tokens plus mutated corpus texts) whose property is "accepted, or rejected with a clause id".
  - C: add a Q2 item for a periodic mutation run over the front-end checks and the oracle's fault checks, each surviving mutant getting a case or a written reason.
  - D: no new technique beyond the hand corpus and T9.
  - B and C each add to A or D.
- Evidence pointers:
  - QA10; QA11; R45; T9 (corpus/extensions.md:8); T22 question 3.
- Binding constraints:
  - None from the owner decisions; a new dependency needs owner approval per rulings.md R45 and engineering.md.

### Q6.11: Where does the corpus live, what format are cases in, and how are directories named and versioned?

- Context:
  - T19 puts `taidl/conformance/` in the taidl wheel, read through the pin; R43 splits the kit between `taidl.testing` and `act/testkit`.
  - Cases are Python `spec.py` scripts because `Accelerator` is the only entry point; reference suites use text formats any implementation can parse.
  - T5 names directories `S<n>.<m>-<slug>`; the plan has no statement of how a pin bump reports added and removed case ids.
- Options:
  - A: corpus in the taidl wheel read through the pin (T19), with the override path of T16.
  - B: the corpus in a fifth repo.
  - C: kit split as R43 states (loader and tolerances in `taidl.testing`; goldens and the oracle import in `act/testkit`).
  - D: the whole kit in one place, either taidl or act.
  - E: cases stay Python `spec.py` plus text `case.toml`, `semantics.txt`, and `kernel.hlo`, with only `spec.py` waiting on L's text format.
  - F: directory names by slug alone, grouped by outline file, with the S id in `case.toml` only.
  - G: a `taidl/conformance/CHANGES.md` rendered per release with added and removed case ids, and `check_versions.py` printing the delta on a pin bump.
  - A and B are alternatives; C and D are alternatives; E, F, and G are independent.
- Evidence pointers:
  - IN10; IN15; IN16; T3; T5; T16; T19; T22 question 1; R42, R43.
  - act/tests/check_versions.py; taidl/tests/test_instruction.py:10.
- Binding constraints:
  - None from the owner decisions; owner decision 10 covers the release batching a pin bump follows.

### Q6.12: Which real ISA descriptions, golden policy, and GPU treatment does the end-to-end suite use?

- Context:
  - R46 keeps one `mini_trn` smoke test per generator; R47 adds one GPU tensor-core description (mma/wgmma and TMA); R48 uses the reference tolerance only for GPU goldens.
  - The audit notes that dropping the matmul-precision flag (R32) may let an f32 dot run as TF32 on GPU (RA5).
  - A listed description that one generator rejects is only noticed through expectation rows (T10).
- Options:
  - A: R46, R47, and R48 as stated.
  - B: no per-generator smoke test; act alone runs the real descriptions.
  - C: T14 skips a kernel whose features an expectations row marks `rejected` and reports it as skipped, with T10's row naming the S8 words and S5 features it needs.
  - D: the oracle emits `precision_config` HIGHEST on every `dot`, tested on GPU.
  - E: GPU is a CI target for bitwise goldens.
  - A and B are alternatives; A and E are alternatives on the GPU golden question; C and D are independent.
- Evidence pointers:
  - IN14; RA5; R32, R46, R47, R48; T10 (corpus/extensions.md:11); T14; T15; T22 questions 4-6.
  - act-oracle/tests/test_mini_trn.py:22; act-backend/tests/test_mini_trn.py:26-30.
- Binding constraints:
  - Owner decision 15: tensor accelerators plus GPU tensor-core instructions (mma, wgmma, TMA); no SIMT thread model.

## For the authors

- QA15: golden updates need a stated policy (regenerate only in a PR that names the semantic change and shows the record diff, on CPU only per R48, and not in the same PR as a generator change unless the reference check passes).
- QA16: `act-oracle/tests/test_linalg.py:1` tests the three words R60 removes; make them S8.5 `rejected` cases.
- QA19: recount the inventory totals (act-backend `merge 16` omits `test_constructs.py`; act-oracle `rewrite 31` counts `test_templates.py`) and use one verdict word set across review.md and the inventory.
- QA20: drop TB5; `test_long_operand_list_laid_out` asserts through `check_oracle` and is a layout test for T4 only.
- QA21: add the missing inventory rows (`act-backend/tests/lint.py`, `lint.sh`, `regenerate.sh`, `cost/*.c`, `act/tests/conftest.py`, the four `pytest.sh` runners) and add `plain_dir` (`act-backend/tests/conftest.py:480-484`) to T2.
- QA22: delete the debate rounds and session note under tests/decisions/, keep one rulings table, and fold T19 to T21 into the items they bind.
- IN11: add an `origin` field to `case.toml` (`hand`, `seeded:<old test path>`, `property:<seed>`, `accelerator:<name>`) so T18's verification greps it.
- IN12: require a one-line `description` in `case.toml` and print it beside the case id in coverage.md.
- Drop the :712 and :749 seeds from T8 or file them as oracle `cli` tests (QA4).
- Add the missing S9 row ids in taidl/docs/conformance.md:9-19 before the first corpus PR (T5 dependency on L).
