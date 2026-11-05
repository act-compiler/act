# Panel 6 round 1: Opus (conformance purist)

Lens: every spec clause has cases, and every generator's behavior on every case is declared and checked.
Assumption: the spec draft (plans/language/spec-draft/) is the clause list the corpus answers to, with S1.3 rows already numbered (overview.md:37-51).
Assumption: rulings are options; where a ruling and the spec draft disagree, a case follows the clause text that ships.
External sources: Test262 INTERPRETING.md (github.com/tc39/test262), the WebAssembly reference interpreter README (github.com/WebAssembly/spec, interpreter/README.md), riscv-arch-test (github.com/riscv-non-isa/riscv-arch-test), StableHLO interpreter_status.md (github.com/openxla/stablehlo), Hypothesis API reference (hypothesis.readthedocs.io).

### Q6.1
Position: option A, a `program.toml` run artifact of instruction calls with attribute values, initial buffers and state, and the expected buffers and state or the S1.3 row id.
1. Ten of the fifteen S1.3 rows are faults triggered by attribute values or state at issue (S1.3.1 to S1.3.10), which an HLO kernel cannot set, so option B leaves most of S1.3 with no corpus case.
2. The oracle already runs exactly this shape: a kernel of instruction calls with attribute values, refused at `fsim-compile` with a defined message.
3. Conformance suites for ISAs attach the outcome to an action on a module, not to a high-level program: Wasm's `assert_trap <action> <failure>` and riscv-arch-test's per-instruction signatures.
4. Option B moves norm cases into per-generator tests, which no other generator can run, so the norm stops being a corpus fact.
5. `kernel.hlo` stays for the backend and T13, because the backend's input is HLO.
- Evidence: spec-draft/overview.md:37-46; act-oracle/tests/test_kernel.py:590-598, :1229-1240; plans/tests/corpus/format.md:8, :11; Wasm interpreter README (assert_trap); riscv-arch-test README (signature from the Sail model).
- Confidence: high.
- Would change my mind: a demonstration that every S1.3 fault row can be triggered from an HLO kernel through some fixed attribute binding.

### Q6.2
Position: option A, with RA2 cases written once their verdict exists; option C is rejected.
1. Owner decision 17 makes storage reuse a spec promise, so it is a corpus clause and not an `impl` detail; option C leaves the backend's half unchecked by any shared case.
2. HLO is value-semantic, so "read the marked input after the instruction" (T13 wording, option B) is not expressible in a kernel; the observation must happen at program level (oracle) or on the simulated assembly (backend).
3. The oracle side is a Q6.1 program: call, then read the aliased region, and expect the output.
4. The backend side is an HLO kernel where the aliased input is still live afterwards; the check is that the oracle-simulated assembly gives the reference result, which fails if the backend clobbers a live value.
5. S1.3.6 already defines one aliasing fault (in-place starts differ at issue), so it gets a program case now; equal-but-differently-spelled regions and partial overlap get cases only after RA2 or panel 1 rules.
6. The existing seeds call `set_inplace`, which R5 removes, so they convert to the derived form or to `rejected` cases.
- Evidence: owner decision 17; spec-draft/overview.md:42; plans/tests/generators.md:50; taidl/tests/test_instruction.py:273-291; act-backend/tests/test_refusals.py:383-396; audit RA2.
- Confidence: medium.
- Would change my mind: evidence that the oracle simulation cannot model the backend's storage reuse, so a clobber would go unnoticed.

### Q6.3
Position: options A and C together.
1. The spec draft says a backend emits programs that never fault, so a backend never has a fault verdict to agree with; option B's "backend rejects with the same S id" conflates compile-time rejection with a run-time fault.
2. R4 and R10 let the backend assume no violation, so a "silent result" from the backend on a violating input is conforming; T13 as written fails a conforming backend.
3. A fault is defined as writing no region and no state, so a case that checks only the raised error tests half the definition; option C asserts unchanged buffers and state.
4. Test262 fails a negative test that throws in the wrong phase or with the wrong type; Wasm `assert_trap` is a run-time assertion separate from `assert_invalid`; both keep run-time faults distinct from rejection.
5. Option A's universal invariant (no simulated backend assembly ever faults) is the backend's only fault obligation, and it is checkable on every differential case for free.
- Evidence: spec-draft/execution.md:4, :34; spec-draft/glossary.md:57; rulings R4, R10, R55; plans/tests/generators.md:49; Test262 INTERPRETING.md (negative.phase); Wasm interpreter README.
- Confidence: high.
- Would change my mind: an owner decision that the backend must detect and report violations itself.

### Q6.4
Position: option A keyed by the S1.3 row ids, plus option D; contested rows follow the spec text, not the draft ruling.
1. S1.3 already numbers every fault and defined behavior, so the norm table is that table with a `cases` column per row and an expectation per generator; no parallel catalogue is needed.
2. Option B leaves contested norms with zero cases, which hides exactly the clauses most likely to regress; option C ties cases to rulings that the audit marks questionable.
3. A case encodes a clause of the spec version it ships with; when panel 1 changes a clause, the clause edit and the case edit land in one PR, so the corpus never disagrees with the spec text.
4. An open row (S1.3.15) has no verdict yet, so it sits on the coverage allowlist (Q6.7) rather than carrying a placeholder case.
5. An `implementation-defined` case with no expectation tests nothing; Wasm states such outcomes as `(either ...)` or `nan:arithmetic`, which is option D's allowed set.
6. The :712 and :749 seeds are kernel-layout errors of the oracle API, not S1.3 faults, so they are oracle `cli` tests, not corpus cases.
- Evidence: spec-draft/overview.md:37-51; plans/tests/corpus/format.md:9; act-oracle/tests/test_kernel.py:712-720, :749-757; audit rows R5, R7, R8, R12, R56; Wasm interpreter README (either, nan:canonical).
- Confidence: medium.
- Would change my mind: a rule that the spec draft will not be co-versioned with the corpus, which would make clause-keyed cases drift.

### Q6.5
Position: options A, B, D, E, and F together; C is rejected.
1. Test262's negative metadata records `phase` and error `type` and fails a test that errors in another phase; that is options A and B.
2. Wasm matches the failure string by prefix and treats the rest as documentation, so a message regex as the contract (option C) is stricter than the reference suites and breaks on rewording, which T6 itself lists as a fail point.
3. R44 puts the clause id in the message, so the clause is the stable key and the message is optional.
4. The spec draft's S9 rows already name an error class per row (`ValueError`, `TypeError`, `SyntaxError`) but no row ids, so option E is needed for cases to land before L numbers them, and CI must fail on `pending` once the ids ship.
5. T7 deletes grammar tests whose facts are parse results; without option D those facts vanish from every suite.
6. One word per concept: R55 names the run-time error "fault", so option F aligns the outcome name with the glossary.
- Evidence: Test262 INTERPRETING.md (negative: phase, type); Wasm interpreter README (prefix match); spec-draft/conformance.md:10-30; spec-draft/glossary.md:57; plans/tests/corpus/format.md:9; taidl/tests/test_grammar.py:352-357.
- Confidence: high.
- Would change my mind: evidence that the front end cannot attribute a rejection to one phase deterministically.

### Q6.6
Position: store the expected outputs as data in the case; generate them offline with NumPy or closed forms (option B), name the generator in `case.toml`, and adopt D, F, and G.
1. riscv-arch-test embeds expected values from the Sail model into the tests, and StableHLO interpreter tests carry expected constants in the test file; the reference tool is a generation-time concern, not a run-time dependency.
2. With stored outputs, R29's dependency problems (not on PyPI, CHLO legalization, version skew against R6) move off the CI path; the interpreter can be a later regeneration source if the owner approves it.
3. The independence requirement stands: a reference computed through JAX shares the oracle's lowering and proves nothing, so `reference` must name a non-JAX source.
4. A per-type `ULPS` cannot bound an accumulation over K terms (audit R8), so the bound belongs to the spec as a function of type and K, stated per case (option D), implemented once in `taidl.testing` (option F).
5. NaN and signed-zero comparison must be stated once, as Wasm does with `nan:canonical` and `nan:arithmetic` (option G).
- Evidence: riscv-arch-test README; StableHLO docs/interpreter_status.md; plans/tests/corpus/seeding.md:21, :28; act-oracle/tests/conftest.py:226-227; audit rows R8, R29, RA4; Wasm interpreter README.
- Confidence: medium.
- Would change my mind: an S8 word whose NumPy result is known to disagree with the StableHLO spec, making NumPy an unsound source for it.

### Q6.7
Position: options A, C, and D; option E is rejected.
1. A clause with zero cases is an untested promise, so coverage must fail CI, not only report; the allowlist names clauses that are not testable by construction (S1.1, S1.2, S10, and open rows such as S1.3.15).
2. riscv-arch-test measures coverpoints against normative rules, and StableHLO tracks interpreter status per specced op; both treat coverage as a per-clause fact.
3. A known defect (R9) is neither `supported` nor `rejected`; carrying it as `rejected` is a false declaration, and skipping it hides when it is fixed; an `xfail` row that fails on XPASS keeps the declaration true.
4. T12 runs only front-end-accepted cases, so rows for rejected cases declare a behavior nobody checks; option D restricts rows to accepted cases.
- Evidence: plans/tests/generators.md:19-25, :31; rulings R9; riscv-arch-test README (coverpoints); StableHLO docs/interpreter_status.md (91 of 96 ops).
- Confidence: high.
- Would change my mind: an allowlist that grows past a handful of rows, which would show the clause list is not testable as written.

### Q6.8
Position: options A and C.
1. A deleted test is a lost check unless its replacement runs at the same cadence; T16 to T18 delete `LOWERED` before the T12 runner exists and per-PR checks before their nightly replacements land.
2. Naming deleted tests by function, not line range, makes the claim "this fact is preserved by case X" verifiable.
3. A removed construct is still an S9 or S8.5 clause in v3 (the front end must reject it), so its tests become `rejected` cases (option C); deleting them (option D) leaves the removal untested.
4. A rejection that names the replacement and the clause id is the migration aid owner decision 10 allows without a converter.
- Evidence: plans/tests/migration.md steps 5-8; act-oracle/tests/test_linalg.py:1; act-backend/tests/test_refusals.py:146-197; audit row R58; owner decision 10.
- Confidence: high.
- Would change my mind: a capacity plan showing option A's ordering does not fit owner decision 11.

### Q6.9
Position: options B, C, and D; option E is not needed.
1. Under R49 a backend regression in a supported word is noticed a day late; option B keeps one built and simulated kernel per S8 word per PR, so the per-PR tests T4 deletes have a per-PR replacement.
2. A tier with no budget or trigger (T12 slow tier) has no failure condition, so option C's table is what makes each tier a check.
3. Exact goldens are only meaningful if generation is deterministic; R17 defines that mode, so T13 to T15 must run in it; Hypothesis likewise derandomizes on CI.
4. Rerun-until-green turns a nondeterministic failure into a pass, so a flake is quarantined with an owner and a date.
5. In deterministic mode byte goldens are stable, and check lines detect strictly less, so option E weakens the gate R17 asks for.
- Evidence: rulings R17, R49; plans/tests/generators.md:34; act-backend/tests/test_mini_trn.py:39-42; Hypothesis API reference (derandomize, default True on CI).
- Confidence: medium.
- Would change my mind: measured per-PR cost of option B above the budget the table sets.

### Q6.10
Position: options A, B, and C, with B and C nightly or periodic and behind owner approval of the dependencies.
1. S9 says a conforming front end rejects exactly the listed descriptions and no other, so the property "accepted, or rejected with a clause id" is the S9 contract itself; option B tests it on inputs nobody wrote.
2. Corpus sufficiency is otherwise a claim; a surviving mutant in a front-end check or an oracle fault check is a clause with no killing case, which is the coverage question measured.
3. Hypothesis replays saved failures and supports `@example`, so promoted counterexamples stay as fixed cases (option A).
4. riscv-arch-test makes coverpoint coverage optional but available; a periodic mutation run is the same kind of optional measurement.
- Evidence: spec-draft/conformance.md:6; plans/tests/corpus/extensions.md:8; rulings R45; Hypothesis API reference (database, example); riscv-arch-test README.
- Confidence: medium.
- Would change my mind: owner refusal of the mutation tool, which reduces C to a manual audit.

### Q6.11
Position: options A, C, E, F, and G.
1. A corpus versioned with the front end it describes keeps clause text and cases in one release, which Q6.4 needs; a fifth repo adds a pin to keep in step.
2. R43 places each kit part with what it depends on; the loader and tolerances need only taidl, goldens need the oracle.
3. Python `spec.py` is forced while `Accelerator` is the only entry point; the other files are text any implementation can read, and Test262 and Wasm suites are text for that reason.
4. The S id belongs in `case.toml`, because a renumbered section would otherwise rename directories and break every expectation row.
5. A pin bump that silently adds or drops cases changes what "conforming" means; option G makes that delta visible.
- Evidence: plans/tests/corpus/format.md:8, :13; rulings R43; taidl/tests/test_instruction.py:10; act/tests/check_versions.py; Test262 INTERPRETING.md.
- Confidence: medium.
- Would change my mind: a decision by L for a text ISA format, which would change E's timing but not the rest.

### Q6.12
Position: options A, C, and D; option E is rejected.
1. A real description is an integration test, not a clause test, so one smoke test per generator (R46) is enough local signal and the corpus carries the clauses.
2. riscv-arch-test selects tests by the device's declared capabilities; option C does the same with expectation rows, so a skipped kernel is a declared skip, not a silent gap.
3. If the oracle lets an f32 dot run as TF32 on GPU, the GPU result checks a different reference than the spec names; option D pins `precision_config` HIGHEST and tests it.
4. Bitwise GPU goldens are not portable across devices (R48), so a GPU golden uses the reference tolerance.
- Evidence: rulings R32, R46, R47, R48; audit RA5; plans/tests/corpus/extensions.md:11, :20-26; act-backend/tests/test_mini_trn.py:26-30; riscv-arch-test README (capability selection).
- Confidence: medium.
- Would change my mind: evidence that HIGHEST precision is unavailable or not honored for a GPU dot shape in the pinned XLA.
