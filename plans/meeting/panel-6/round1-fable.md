# Panel 6, round 1, Fable: industry conformance suites

Lens: how WebAssembly spec tests, Test262, LLVM lit and FileCheck, the StableHLO interpreter suite, and riscv-arch-test with RISCOF are built and kept green.
Assumption throughout: the corpus is a conformance suite in the sense of those references, so a case is data any implementation can run, and a verdict is checked, never merely recorded.

### Q6.1

Position: option A, a second run artifact (`program.toml`) with initial buffers and state, instruction calls with attribute values, and expected buffers and state or a fault clause.

1. Every mature suite separates the module under test from the script that drives it: a `.wast` file holds a module plus `invoke` actions and `assert_return` or `assert_trap` lines, and a riscv-arch-test case is an assembly program whose signature region is dumped and compared.
2. The seeds the plan names are already such scripts: `test_window_outside_buffer_refused` calls `api.store_rm(n=64, addr_in=0, addr_out=28672)` inside a `kernel` decorator, and `test_division_by_zero_refused` passes `n=0, src=8`.
3. HLO is value-semantic and attribute-free at this level, so an HLO kernel cannot name an address, an attribute value, or a state register; option B therefore leaves every run-time norm untestable in the corpus, which contradicts owner decision 16 making each one a stated verdict.
4. The oracle already executes instruction calls directly through the kernel API, so running `program.toml` needs a loader, not a new execution path.

Evidence: WebAssembly spec interpreter README (script assertions `assert_return`, `assert_trap`, `assert_malformed`); RISCOF test format page (signature region, `RVTEST_SIGUPD`); act-oracle/tests/test_kernel.py:590-598, :1229-1240; plans/tests/corpus/format.md T5 bullets 1 and 4; owner decision 16.
Confidence: high.
Would change my mind: a demonstration that the oracle can bind attribute values and addresses from an HLO custom-call or frontend attribute so that `kernel.hlo` alone reaches every fault clause.

### Q6.2

Position: option A, oracle aliasing through Q6.1 programs and backend aliasing checked by simulating the assembly with aliasing modelled; drop the mark wording.

1. The WebAssembly memory tests assert aliasing by storing through one view and loading through another after the action, which is a program-level check, not a module-level one; the TAIDL analogue is a program that writes through the output region and reads the input region afterwards.
2. Both named seeds use APIs v3 removes: `set_inplace` at test_instruction.py:273-291 (R5) and `set_apart` or `set_no_partial_overlap` at test_refusals.py:383-396 (R58), so option B restates cases that will not parse in v3.
3. HLO has no post-instruction read of an input's storage, so the backend half of T13's wording cannot be written; the observable backend fact is that the assembly, simulated with aliasing modelled, still matches the reference while the aliased input is live.
4. Option C leaves the R5 derivation (identical region texts) and RA2's open cases (equal regions spelled differently, partial overlap) with no cross-generator check, which is exactly where the audit found the ruling thin.

Evidence: WebAssembly spec interpreter README (action then assertion); taidl/tests/test_instruction.py:273-291; act-backend/tests/test_refusals.py:383-396; act-backend/tests/test_extraction.py:507 (`set_inplace([True])` on a `d2` input); plans/reviews/rulings/audit.md rows R5 and RA2; owner decision 17.
Confidence: medium.
Would change my mind: a backend simulation path that cannot model aliasing without the removed marks, which would make C the only honest option.

### Q6.3

Position: options A and C together: fault cases are oracle-only programs asserting `expect = "fault"` plus unchanged buffers and state, and T13 adds the invariant that a supported kernel's assembly never faults on the oracle.

1. `assert_trap` checks the trap alone and the WebAssembly README says the failure string is documentation checked only as a prefix; that is the weaker precedent and it works because a trap in Wasm has no partial write by construction.
2. R55 defines a fault as writing no region and no state, so the TAIDL norm has a memory post-condition; riscv-arch-test compares the whole signature region after the test, which is the precedent for asserting state after a fault.
3. R4 and R10 let the backend assume the violation never occurs, so "silent result on either side" in T13 asks the backend to detect what the ruling says it may ignore; the consistent invariant is that the backend never produces a faulting assembly for a supported kernel.
4. A faulting call is not expressible in HLO, so T13 cannot carry fault cases at all; option B keeps text that no runner can execute.

Evidence: WebAssembly spec interpreter README (`assert_trap`, failure string as prefix); RISCOF test format page (signature region preloaded with `0xdeadbeef`, compared after the run); plans/minutes/rulings.md R55; plans/tests/generators.md T13 `error` bullet; act-oracle/tests/test_kernel.py:590-598; owner decision 16.
Confidence: high.
Would change my mind: a ruling that the backend must emit bounds checks, which would make backend faults observable and give option B a runner.

### Q6.4

Position: option A plus D, with B applied to the contested rulings only: a norm table with one row per ruling, cases now for rulings the audit leaves standing, cases for R5, R7, R8, R12, and R56 after their panel decides, and every `implementation-defined` case carrying an allowed set or tolerance.

1. The StableHLO interpreter test guidelines say to skip implementation-defined and undefined behavior and to give each op its own suite; a conformance case encodes a settled norm, not a draft.
2. Test262 writes tests for a proposal only at stage 3 and tags them with `features`, so a suite can hold tests for in-flight semantics without binding implementations to them; the TAIDL equivalent is a norm table row with no case id until the ruling lands.
3. A case with no expectation is not a test; the one implementation-defined row the semantics track keeps (float reduction order within the H6 tolerance) is expressible as a tolerance, and WebAssembly's `nan:canonical` and `nan:arithmetic` patterns show the allowed-set form.
4. Option C writes cases that the audit already predicts will change, which doubles the migration work under owner decision 11's capacity.

Evidence: StableHLO reference page (testing guidelines, `check.expect_almost_eq`); Test262 CONTRIBUTING.md (features tag, stage 3 rule); WebAssembly spec interpreter README (NaN patterns); plans/reviews/rulings/audit.md rows R5, R7, R8, R12, R56; plans/semantics/rulings.md:21; plans/semantics/holes.md H1, H3, H4, H6.
Confidence: medium.
Would change my mind: a panel 1 decision within the quarter on H1 to H6, which would remove the contested set and make plain A plus D sufficient.

### Q6.5

Position: options A, B, E, and F together, D as an optional table, not C: a required `phase`, matching on `clause` plus `error_type`, `message` optional, `clause = "pending"` until L numbers S9, and `fault` as the run-time outcome.

1. Test262 fails a negative test that throws in a phase other than the one listed or with a constructor other than `type`; WebAssembly separates `assert_malformed` from `assert_invalid`; both references treat the phase as part of the contract.
2. Both references treat the message as non-normative: Test262 never matches text and the WebAssembly interpreter checks the failure string only as a prefix for documentation; a regex as the contract (T5) makes every rewording a corpus change, which T6 itself lists as its fail point.
3. R44 puts the clause id in the message, so `clause` is machine-checkable at zero extra cost, and `error_type` maps onto the two layers T6 records (`SyntaxError` at load, `ValueError` in the API).
4. A parse-tree fact such as the literal text at test_grammar.py:352-357 is a value expectation, so an optional `expect` table (D) keeps it; without D, T7 must keep those blocks as unit tests, as LLVM keeps unit tests apart from lit regression tests.
5. R55 names the run-time error a fault, so F is a vocabulary fix with no cost.

Evidence: Test262 INTERPRETING.md (`negative.phase`, `negative.type`, wrong phase fails); WebAssembly spec interpreter README (`assert_malformed` vs `assert_invalid`, prefix check); LLVM Testing Guide (unit tests vs regression tests); plans/minutes/rulings.md R44, R55; plans/tests/corpus/format.md T5 and T6; taidl/tests/test_grammar.py:352-357; owner decision 10.
Confidence: high.
Would change my mind: evidence that the front end cannot distinguish lex from parse errors without ANTLR internals, which would collapse `phase` to `parse`, `build`, `generate`, `run`.

### Q6.6

Position: reference B for v3 (NumPy or closed form, named and pinned in `case.toml`) with the StableHLO interpreter as a nightly cross-check only; tolerance D, with F and G.

1. Sail is the RISC-V golden reference because it is an executable spec that every target compares against; the StableHLO interpreter plays that role for StableHLO, but it is built from source with Bazel or CMake, has no PyPI release, and its own page states a one-size tolerance of 0.0001 with transcendental tolerances still being worked out.
2. A reference whose version can differ from the R6 jax pin checks a different spec than the oracle lowers to, which the audit flags; a NumPy or closed-form reference is pinned by the same wheel set every repo already uses.
3. R29 and R45 require owner approval for a new dependency, so A or C cannot be the day-one reference; B is executable now from `test_reduce.py:87-129` and `conftest.py:226-227`.
4. One `ULPS` per float type (E) cannot express the R8 bound, which the audit says grows with K; a per-case `expect = "exact"` or `"almost"` with `ulps` is the StableHLO `expect_eq` versus `expect_almost_eq` split, and the spec owns the bound while `taidl.testing` implements it (F).
5. WebAssembly's `nan:canonical` and `nan:arithmetic` and `test_reduce_identity`'s byte-level signed-zero compare show that NaN and signed-zero rules must be stated once (G), otherwise every case re-decides them.

Evidence: StableHLO reference page (interpreter via `stablehlo-translate --interpret`, `check.expect_eq`, `check.expect_almost_eq`, 0.0001 tolerance); RISCOF test format page (golden reference); WebAssembly spec interpreter README (NaN patterns); plans/reviews/rulings/audit.md rows R8, R29, RA4; plans/minutes/rulings.md R6, R8, R29; act-oracle/tests/conftest.py:226-227; act-oracle/tests/test_reduce.py:87-129; owner decision 14.
Confidence: medium.
Would change my mind: an owner-approved build route that pins the interpreter to the R6 StableHLO commit and a measured match with NumPy on every S8 word; then A becomes the normative reference.

### Q6.7

Position: options A, C, and D; not B or E: a rendered `coverage.md` per S id and norm row with CI failing on a zero row outside an allowlist, an `xfail` row kind that runs and fails on XPASS, and a derived per-section status.

1. The LLVM Testing Guide states that an XFAIL test is successful if it fails and a failure if it succeeds, and that UNSUPPORTED skips silently so nobody learns when it starts working; a skipped case (E) is the UNSUPPORTED trap for a known defect such as R9.
2. The StableHLO guideline that each op has its own test suite is a coverage rule enforced by review; a renderer with a failing zero row is the same rule enforced by CI, which is cheaper for 3-4 engineers.
3. A row for a case the front end rejects (T11) is dead data, as T12 never runs it; riscv-arch-test selects tests from the device YAML and reports unsupported extensions rather than listing a row per excluded test.
4. Reporting alone (B) is what T5 does today and what QA5 found drifting.

Evidence: LLVM Testing Guide (XFAIL, XPASS, UNSUPPORTED); StableHLO reference page (per-op suites); RISCOF test format page (test database from `RVTEST_CASE` conditions and device YAML); plans/tests/generators.md T11; plans/tests/corpus/sections.md:33; plans/minutes/rulings.md R9.
Confidence: high for C and D, medium for A.
Would change my mind: a count showing most S9 clauses are untestable until L's rows ship, which would make the allowlist larger than the gate.

### Q6.8

Position: options A and C: delete a test only in the step whose runner covers its fact at the same cadence, and convert tests of removed constructs into `rejected` cases that name the replacement with the clause id.

1. The LLVM guide's rule is that regression tests always pass and run before every commit; a suite never removes a check before the replacement check runs, and the migration table breaks this twice (step 6 deletes `LOWERED` before step 8's runner, step 8 deletes word loops whose cases land at step 12).
2. Deleting by line range (T4) is fragile; lit and Test262 name tests by file and feature, and the one unit test proving a mark reaches the e-graph lives inside the range.
3. Test262 kept tests for removed features as negative tests rather than deleting them, and R44 already puts the clause id in the message, so a `rejected` case for `set_inplace`, `set_apart`, and the linalg words costs one directory each and gives users of the removed construct an error that names the replacement (RA11).
4. The arity check at test_refusals.py:164-176 has no replacement until the T12 slow tier runs per PR, so it stays as an `impl` test.

Evidence: LLVM Testing Guide (regression tests run before every commit); Test262 CONTRIBUTING.md (file naming); plans/tests/migration.md steps 5-8 and 12; act-backend/tests/test_refusals.py:164-176, :383-396; act-oracle/tests/test_refusals.py:44-45; plans/reviews/rulings/audit.md rows R58 and RA11; owner decisions 10 and 11.
Confidence: high.
Would change my mind: a CI run showing the T12 slow tier fits a per-PR budget from step 8, which would let step 6 and step 8 keep their order.

### Q6.9

Position: options B, C, D, and E: a per-PR slow tier in act-backend beside the nightly differential tier, a tier table with budgets and overrun actions, R17 deterministic mode plus a quarantine rule, and FileCheck-style check lines over assembly text instead of byte goldens.

1. LLVM runs its regression tests on every commit and keeps whole-program tests in a separate test-suite repo, which is the per-PR and nightly split; a differential tier that is nightly only (A) leaves a PR with no signal on selection facts after step 15 deletes them.
2. FileCheck matches patterns with variable captures so a test asserts the instruction word and operand relations and tolerates incidental text; a byte golden fails on any renaming and is why R17 asks for a deterministic mode in the first place.
3. `test_compiled_kernel_matches_golden` asserts `max_diff == 0`, so goldens are exact today; exact goldens plus a search with wall-clock limits are flaky by construction, so the deterministic mode must be named in T13 to T15.
4. Hypothesis documents `derandomize` on CI with longer non-deterministic nightly runs, which is the same tier split for property tests.
5. Owner decision 11 caps capacity, so the tier table (C) is the budget instrument; without it the slow tier has no trigger or limit (T12).

Evidence: LLVM Testing Guide (regression tests vs test-suite); LLVM FileCheck guide (`[[VAR:regex]]`, `CHECK-NEXT`, `CHECK-DAG`); Hypothesis settings reference (`derandomize`, `database`); act-backend/tests/test_mini_trn.py:39-42; plans/minutes/rulings.md R17, R49; plans/tests/generators.md T12; owner decision 11.
Confidence: high for E and D, medium for B.
Would change my mind: a measured per-PR build time for one crate per S8 word exceeding the budget the owner accepts, which would push B to a nightly tier and leave A.

### Q6.10

Position: A plus B now, C as a bounded Q2 pilot on the front-end checks only, not D.

1. Test262 and WebAssembly both grew generated and fuzzed corpora beside hand cases once the hand corpus plateaued; the property "accepted, or rejected with a clause id" is the Wasm `assert_malformed` or `assert_invalid` invariant applied to arbitrary text, and it needs no reference value.
2. Hypothesis ships CI support the plan relies on: `derandomize` for repeatable runs, `print_blob` and `@reproduce_failure` for promotion of a shrunk counterexample to a corpus case.
3. Mutation testing is the only measurement of corpus sufficiency; mutmut needs no source change, caches completed work, and runs only the tests relevant to a mutant, so a periodic run over the front-end checks is small.
4. A second new dependency needs owner approval under R45, so C stays a pilot with a written surviving-mutant reason, not a gate.

Evidence: WebAssembly spec interpreter README (`assert_malformed`, `assert_invalid`); Hypothesis settings reference (`derandomize`, `print_blob`, `@reproduce_failure`); mutmut documentation (no source changes, resume, relevant tests only); plans/tests/corpus/extensions.md T9; plans/minutes/rulings.md R45.
Confidence: medium.
Would change my mind: a mutmut pilot showing a mutation score already above 90 percent from the hand corpus, which would make C not worth a dependency.

### Q6.11

Position: options A, C, E, F, and G: corpus in the taidl wheel read through the pin, kit split per R43, Python `spec.py` plus text files for now, slug-only directory names grouped by outline file, and a rendered `CHANGES.md` with a pin-bump delta.

1. The WebAssembly spec tests live in the spec repository and engines vendor them at a version, and Test262 is consumed by engines at a pinned commit; the language owner publishes the corpus and implementations pin it, which is A, not a fifth repo.
2. Test262 deprecated `es5id` and `es6id` after spec renumbering and names files by feature; `S<n>.<m>-<slug>` repeats that mistake on day one, and the S id belongs in `case.toml` where a renumbering is one field edit.
3. Reference suites are text any implementation can parse, and `spec.py` is the one artifact waiting on L; `case.toml`, `semantics.txt`, and `kernel.hlo` are already text, so E keeps the text share maximal without blocking on L.
4. `check_versions.py` already checks that every place naming a version agrees, so printing added and removed case ids on a pin bump is an extension of an existing script, not a new tool.
5. R43's split puts loader and tolerances with the spec and goldens with the oracle import, which matches where each dependency lives.

Evidence: Test262 CONTRIBUTING.md (deprecation of `es5id` and `es6id`, naming by feature); WebAssembly spec repository layout (test/core vendored by engines); act/tests/check_versions.py:1-30; taidl/tests/test_instruction.py:10; plans/tests/migration.md T16; plans/minutes/rulings.md R42, R43.
Confidence: high for A and F, medium for C.
Would change my mind: a second implementation outside these four repos, which would justify a corpus repo with its own release cadence (B).

### Q6.12

Position: options A, C, and D: keep R46 to R48 as stated, skip and report a kernel whose features an expectations row marks `rejected`, and emit `precision_config` HIGHEST on every oracle `dot`; not E.

1. riscv-arch-test compares a signature against the reference for the target's configuration and selects tests from the device YAML, so an unsupported feature is a reported skip, not a failure; T14 should do the same from T10's rows (C).
2. GPU results are not bitwise portable across drivers and architectures, and R48's reference tolerance is the RISCOF stance of comparing to a reference rather than to another target's bytes; E would make CI depend on one GPU model.
3. The StableHLO `dot_general` carries `precision_config` with DEFAULT, HIGH, and HIGHEST, and DEFAULT permits reduced-precision f32 on GPU (TF32); once R32 drops the matmul-precision flag the oracle must state HIGHEST in the module or the reference tolerance absorbs a precision bug (RA5).
4. A per-generator smoke test (R46) is the LLVM pattern of a fast regression check per commit beside the whole-program suite; B removes the only per-PR signal that `mini_trn` still generates.
5. The attention test is skipped on GPU because the goldens are exact on the CPU alone, which confirms R48 from the existing code.

Evidence: RISCOF test format page (device YAML selection, golden reference comparison); StableHLO specification, `dot_general` `precision_config` (cited from the spec's op table; not re-read this session); LLVM Testing Guide (regression tests per commit); act-oracle/tests/test_mini_trn.py:22; act-backend/tests/test_mini_trn.py:26-30, :39-42; plans/reviews/rulings/audit.md row RA5; plans/minutes/rulings.md R32, R46, R47, R48; owner decision 15.
Confidence: medium.
Would change my mind: a GPU run showing f32 `dot` under DEFAULT precision already matches HIGHEST within the R8 bound on the supported hardware, which would make D unnecessary.
