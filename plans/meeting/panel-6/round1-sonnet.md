# Panel 6, round 1: Sonnet (CI and test engineer)

Lens: what runs on every pull request, how long it takes, whether it gives the same answer twice, and whether a migration step can be reverted.
Assumptions used throughout:

- A1: GitHub-hosted runners, no GPU, and the amd64 plus arm64 matrix (act-backend/.github/workflows/test.yml:11-17, act/.github/workflows/test.yml:11-17).
- A2: A pull request check that takes over about 15 minutes is skipped or batched by contributors; this is practice, not a measured figure here.
- A3: The 503 `def test_` counts across the four test trees are mostly fast Python; the slow cost is cargo builds and e-graph search.
- A4: A new dependency needs owner approval first (rulings R29, R45).

## Q6.1

Position: option A, a second run artifact (`program.toml`) that the oracle runs in process; `kernel.hlo` stays for the backend.

1. Decision 16 gives every run-time violation a verdict, and an HLO kernel cannot set attributes, addresses, or state (corpus/format.md:11, qa-lead QA1).
2. The seeds are instruction-call programs: `test_window_outside_buffer_refused` calls `api.<name>(n=64, addr_in=..)` (act-oracle/tests/test_kernel.py:590-598) and the divide case calls `divide_api.load_quotient(n=0, src=8)` (:1229-1240).
3. Option B leaves those norms in per-generator tests, so the corpus has no row for them and a second generator cannot run them; that defeats the shared corpus.
4. For CI the program runs in the oracle without cargo, so it belongs in the per-PR fast tier; WebAssembly keeps `assert_trap` and `assert_return` actions in the same script as the module for the same reason (https://github.com/WebAssembly/spec/blob/main/interpreter/README.md).
5. The runner asserts the fault clause id and error class, not the message; test_kernel.py:598 compares the whole message with `==`, which is the brittle form to avoid.

Confidence: high.
Would change my mind: a showing that every violation in the norm catalogue is reachable from a compiled HLO kernel plus `inputs.npz`.

## Q6.2

Position: option A, with option C's split as the floor; the oracle aliasing cases are programs, the backend side is a small set of HLO kernels, and e-graph in-place facts stay `impl`.

1. HLO is value-semantic, so T13's "read the marked input after the instruction" cannot be written as a kernel (qa-lead QA2; generators.md:111).
2. R5 derives aliasing from identical region texts, so the oracle case is a program with two equal region texts and a read after the call; partial overlap gets the RA2 fault (audit RA2).
3. The seeds use removed APIs: `set_inplace` at taidl/tests/test_instruction.py:273-291 and act-backend/tests/test_extraction.py:507, and `set_apart` marks at act-backend/tests/test_refusals.py:383-396; these tests must be rewritten, not moved.
4. Backend side: an HLO kernel with an input live after the aliasing instruction, run through the oracle with aliasing modelled, is a result check that needs no new artifact; it is nightly work (R49) because it needs a backend build.
5. The e-graph reachability fact (test_generators.py:243) and the dropped-candidate check (act-backend/tests/test_mini_trn.py:78) are search facts that no result check sees, so they stay per-PR `impl` tests (T20).

Confidence: medium; the model of aliasing in the oracle simulation is unbuilt.
Would change my mind: evidence that the oracle simulation cannot model aliasing without a new runtime, which moves me to option C.

## Q6.3

Position: options A and C together; fault cases are oracle-only and assert unchanged buffers and state; one universal invariant covers the backend.

1. R4 and R10 let the backend assume no violation, and the audit marks "executor faults, producer never emits" (audit R4, R10), so T13's "silent result is a failure" (generators.md:110) would fail a correct backend.
2. The universal invariant "simulating a supported kernel's backend assembly never faults" is one assertion inside every T13 case, so it costs no extra build.
3. R55 defines a fault as writing no region and no state; an assertion that checks only the raise misses a half-written output, which is the likelier bug in an oracle (IN7; riscv-arch-test compares the whole signature region, https://riscof.readthedocs.io/en/stable/testformat.html).
4. WebAssembly `assert_trap` checks the trap alone, the weaker precedent; the pre-call snapshot compare is a few lines in the runner and deterministic.
5. Rename the outcome `error` to `fault` so the case field matches R55.

Confidence: high.
Would change my mind: an oracle design where a fault cannot leave a partial write by construction, making the snapshot assertion vacuous.

## Q6.4

Position: option A (norm table) gated by option B (cases only for norms whose rulings stand), with option D; a contested norm has a table row marked pending and no case.

1. A case that encodes a norm the owner later changes is a false red for both generators; CI cost is two repos blocked plus a pin bump to repair.
2. The audit marks R5, R7, R8, R12, and R56 questionable or wrong (audit rows R5, R7, R8, R12, R56), so five norm rows are unstable today.
3. Option C (write now, revise later) is cheap per case but churns case ids and expectation rows keyed on them (generators.md:80).
4. The table row still exists for a pending norm, so QA4's gap (no catalogue) closes now and the coverage renderer lists it as a known gap (Q6.7).
5. An `implementation-defined` case with no expectation tests nothing (QA17); an allowed set or a tolerance is the form Test262 uses for `non-deterministic` tests (https://github.com/tc39/test262/blob/main/INTERPRETING.md).
6. Fault cases already follow Q6.3, so the table needs only `result` or `fault` kinds.

Confidence: medium; depends on how fast panels 1 and 5 settle the contested rulings.
Would change my mind: a ruling date from the owner that settles all five before the first corpus PR, which makes option C the cheapest.

## Q6.5

Position: options A, B, E, and F combined, with D for the parse-tree assertions; C is rejected.

1. The match contract must survive message rewording; T6 itself names rewording as its fail point (corpus/format.md:20-29).
2. Test262 fails a negative test whose phase differs and matches the error constructor name, not the message (https://github.com/tc39/test262/blob/main/INTERPRETING.md).
3. WebAssembly checks the failure string only as a prefix kept for documentation (https://github.com/WebAssembly/spec/blob/main/interpreter/README.md).
4. So the required match is `phase`, `clause`, `error_type`; `message` is an optional regex in a few cases.
5. Today rejection happens at several layers (corpus/format.md:24, taidl/tests/test_instruction.py:53-66), and a case rejected by the wrong layer passes silently under C; `phase` removes that.
6. T7 would delete assertions that a verdict cannot express (taidl/tests/test_grammar.py:352-357), so an optional `expect` table is the cheaper fix than keeping those tests beside the corpus.
7. `clause = "pending"` (E) unblocks the first corpus PR before L numbers S9 (corpus/format.md:15); CI fails on pending once the list ships.

Confidence: high on phase and clause; medium on D versus deleting only verdict blocks.
Would change my mind: a front end that cannot report a stable error class across the lexer, parser, and API (then `phase` plus `clause` alone).

## Q6.6

Position: expected values are stored data in the case (computed once by a regenerate script), the per-PR gate uses no reference library, and the pinned interpreter is the preferred source of those values once the owner approves it; tolerance is per case (D) with F and G.

1. A per-PR check that imports a reference library inherits its install and version risk; the audit says the interpreter is not on PyPI, needs CHLO legalization, and may differ from R6's pin (audit R29).
2. Stored expected arrays make the gate deterministic and fast: WebAssembly `assert_return` and the StableHLO interpreter tests store results beside the program (https://github.com/openxla/stablehlo/blob/main/docs/reference.md).
3. The `reference` name and pinned version in `case.toml` (IN4) record provenance, so a reviewer sees what produced the data; T8's "not JAX" rule (corpus/seeding.md:57) stays.
4. Until the owner approves the build route, NumPy or closed forms fill the role (option B), which needs no approval (A4) and is what the oracle tests already use (act-oracle/tests/test_reduce.py:87-129).
5. One `ULPS` per type (act-oracle/tests/conftest.py:226-227) is too coarse; R8's accumulation clause and tolerance clause conflict at K=64 (audit R8), so the bound is stated per case as a function of K.
6. The tolerance owner is the spec for the bound and `taidl.testing` for the code (RA4, F); G (NaN, signed zero, infinity) is stated once there, since WebAssembly needs `nan:canonical` patterns for the same reason.

Confidence: medium.
Would change my mind: the owner approving a prebuilt interpreter wheel with a pinned hash, which makes option A the per-PR reference at low cost.

## Q6.7

Position: options A, C, and D; not B, not E.

1. A section count hides a clause with no case (corpus/sections.md:33); a failing zero row with a short allowlist is mechanical (QA5, IN3).
2. Test262 ties each test to one spec anchor and reports per clause (https://github.com/tc39/test262/blob/main/CONTRIBUTING.md).
3. A known defect (R9) fits neither `supported` nor `rejected` (generators.md:80); a skip hides it and a red test blocks the merge queue.
4. lit `XFAIL` runs the test, passes while it fails, and fails when it starts passing; `UNSUPPORTED` skips (https://llvm.org/docs/TestingGuide.html).
5. The `xfail = "<id>"` row must be strict, as pytest `xfail(strict=True)` is, so a fixed bug is noticed in the next run; this is the same mechanism Q6.4 uses for pending norms.
6. Rows only for front-end-accepted cases (D) because T12 never runs a rejected case (generators.md:25, QA18).

Confidence: high.
Would change my mind: a count of known-defect rows under three for the whole plan, which makes E tolerable.

## Q6.8

Position: option A with option C; a deletion needs a named replacement test that has run green at the same cadence.

1. Migration step 6 deletes `LOWERED` before the T12 runner at step 8, and `act-oracle/tests/test_refusals.py:44-45` still reads it (migration.md:24-26; QA6).
2. The overhaul order has the same shape: step 5 deletes `LOWERED` and step 6 adds the runner (overhaul.md steps 5-6).
3. Rule: each PR that deletes tests lists their function names, and the replacement's test id; a script compares test ids before and after, so no id disappears without a mapped successor.
4. Rule: before deleting a guard, break the guarded code locally and confirm the replacement goes red; this is a one-off manual mutation check, cheap per PR.
5. Cadence: the per-PR arity check at act-backend/tests/test_refusals.py:164-176 stays until the T12 slow tier runs per PR (QA7); R49 moves coverage to nightly (rulings R49).
6. Removed constructs become `rejected` corpus cases with the replacement named (option C, RA11); option D loses the fact that the construct is now refused, which a user upgrading sees first.
7. T4 deletes by line range (`test_generators.py:175-393`); names by function avoid deleting the one test at :243 (QA8).

Confidence: high.
Would change my mind: a reorder of the steps so each runner precedes its deletion, which makes the explicit rule redundant but harmless.

## Q6.9

Position: options B, C, D, and E; A alone is rejected.

1. The CI today runs the whole suite once per push and pull request on two architectures (act-backend/.github/workflows/test.yml:3-17, tests/pytest.sh); there is no tier concept yet.
2. A nightly-only differential tier (A) moves the e-graph invariants out of the pull request loop (QA7); one build plus one oracle-simulated kernel per S8 word is the minimum per-PR slow tier (B).
3. Proposed tier table (budgets are proposals under A2, to be measured in the first week):

| Tier | Trigger | Budget | Overrun action |
| ---- | ------- | ------ | -------------- |
| taidl | pull request | 5 min | split the job |
| generator fast | pull request | 10 min | move cases to slow |
| generator slow | pull request, changed paths | 20 min | quarantine rule, then move to nightly |
| act differential | nightly | 90 min | open a ticket with owner |
| release | tag | 3 h | block the tag |

4. The backend search stops on wall-clock `TIME_LIMIT` of 1000 ms (act-backend/act_backend/generic/src/lib.rs:18, pipeline.rs:157), so a loaded runner can pick different assembly; byte goldens and R17 gates are flaky until R17's deterministic mode exists (rulings R17, QA13).
5. Exact goldens (`max_diff == 0`, act-backend/tests/test_mini_trn.py:39-42) run on amd64 and arm64 (A1); exact float equality across the two is unproven, so the first step is one run on both and per-architecture records if they differ.
6. Hypothesis's `ci` profile sets `derandomize=True`, `deadline=None`, `database=None`, and its docs suggest quick deterministic runs per commit plus a longer random nightly (https://hypothesis.readthedocs.io/en/latest/reference/api.html).
7. Flaky rule (D): one PR quarantines a test with an owner and a removal date; no rerun-until-green, since a rerun hides the nondeterminism that goldens are meant to expose.
8. Check lines over assembly text (E) survive renames that byte goldens do not, as FileCheck does (https://llvm.org/docs/CommandGuide/FileCheck.html); they still need R17 mode to be stable.

Confidence: medium; budgets are unmeasured.
Would change my mind: measured timings showing the slow tier under 5 minutes (then it joins the fast tier) or R17 mode unavailable (then assembly checks stay nightly).

## Q6.10

Position: option A in the pull request; option B as a nightly job with a time cap; option C once, per release, on a narrow scope; owner approval for the new tools.

1. T9 uses well-formed strategies only, so nothing checks that arbitrary text is accepted or rejected with a clause id (corpus/extensions.md:8, QA10).
2. Per pull request Hypothesis runs derandomized with a fixed seed so a red run reproduces; nightly runs random with the failure database saved as a CI artifact and shrunk cases promoted to corpus cases with `origin = property:<seed>` (IN11; https://hypothesis.readthedocs.io/en/latest/reference/api.html).
3. The fuzz property "accepted, or rejected with a clause id" has a cheap oracle: the runner of Q6.5, so no new infrastructure.
4. Mutation testing measures corpus sufficiency, but a full run executes the suite once per mutant; mutmut documents that incidentally tested functions make runs slow and offers path filters (https://github.com/boxed/mutmut).
5. So C runs on the front-end checks and the fault checks only (QA11), per release, and each surviving mutant gets a case or a written reason; it never gates a pull request.
6. Both B and C add test-only dependencies (R45), so approval comes first (A4).

Confidence: medium.
Would change my mind: a first mutation run showing a score so low that C should move earlier, or a nightly fuzzer that finds nothing for three months, which retires it.

## Q6.11

Position: options A, C, E, F, and G.

1. The corpus in the taidl wheel gives language and tests one version, and the generator CI reads it through the pin (T16, migration.md:174-183); a fifth repo adds a fourth pin to keep in sync with `act/tests/check_versions.py`.
2. A single stored location for the loader and tolerances (`taidl.testing`) and another for goldens and the oracle import (`act/testkit`) follows R43, since goldens depend on the oracle and the taidl wheel must not (rulings R43).
3. Directory names by slug (F): the test id and every expectation row key on the directory name, so a renumbering that renames directories invalidates quarantine lists and xfail rows; Test262 deprecated `es5id` and `es6id` for this reason (https://github.com/tc39/test262/blob/main/CONTRIBUTING.md).
4. `CHANGES.md` and a printed delta on a pin bump (G) turn "collection failed on missing rows" into a readable list, which is the moment a pin bump costs the most.
5. Python `spec.py` plus text sidecars (E) is the only form the front end reads today (taidl/tests/test_instruction.py:10); WebAssembly mirrors its text testsuite into consumer repos (https://github.com/WebAssembly/testsuite/blob/main/README.md), which is the model once L decides a text format.

Confidence: high on A, F, G; medium on C.
Would change my mind: a non-Python generator appearing, which moves the corpus to text-only cases.

## Q6.12

Position: options A, C, and D; E is rejected; keep the per-generator smoke test.

1. The smoke test in each generator (`act-backend/tests/test_mini_trn.py:26-30`) is the only per-PR check that a generator builds its own example; dropping it (B) moves failure discovery to the act repo, one pin bump late.
2. Real descriptions through both generators are the end-to-end value of act (T14); `act/tests/test_end_to_end.py` runs on both architectures (A1).
3. A kernel one generator rejects is reported skipped from the expectations row, not failed (C, IN14; riscv-arch-test conditions tests on the DUT's configuration, https://riscof.readthedocs.io/en/stable/testformat.html).
4. A GPU job is not available on hosted runners (A1), so bitwise GPU goldens (E) cannot be a CI target; R48 already uses the reference tolerance for GPU, and the current skip is `act-oracle/tests/test_mini_trn.py:22`.
5. `precision_config` HIGHEST on every `dot` (D) is a one-line oracle change that removes a TF32 source of GPU differences (audit RA5); it is tested when a GPU run exists.
6. Golden policy (QA15): regenerate only in a PR that names the semantic change and shows the record diff, on CPU, never with a generator change unless the reference check passes (act-backend/tests/regenerate.py:35-52).

Confidence: medium; GPU treatment depends on a runner being available at all.
Would change my mind: a GPU runner available to the project, which moves the GPU tolerance job from manual to nightly.
