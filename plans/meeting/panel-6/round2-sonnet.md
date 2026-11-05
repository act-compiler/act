# Panel 6, round 2: Sonnet (CI and test engineer)

Lens: what runs on every pull request, how long it takes, whether it gives the same answer twice, and whether a migration step can be reverted.
Binding input used: the in-place mark stays, since overlap alone does not settle read-after-write (in `r0 = r1 + r0` an implementation may or may not reuse the source storage).
Binding input used: owner decisions 10, 11, 16, and 17.
Checked this round: plans/semantics/rulings.md:8-13, taidl/tests/test_instruction.py:273-291, act-backend/tests/test_extraction.py:500-507, act-oracle/tests/conftest.py:226-227, rulings R17 and R49, audit rows R5, R22, RA2.
Not verified this round: the `dot_general` `precision_config` values in the StableHLO spec (the fetched page was truncated before that section).

### Q6.1

All three debaters choose option A (a `program.toml` run artifact beside `kernel.hlo`); no opposing argument remains.
Final position: option A.
Confidence: high.
What still divides the panel: nothing.

### Q6.2

Strongest opposing argument (Fable, Opus): the observable half of in-place is a program-level read after the call, so the corpus case is a Q6.1 program, and the mark wording of T13 should be dropped or restated through the R5 derivation.
Fable says "drop the mark wording"; Opus accepts R5's derivation and defers the RA2 cases.
Rebuttal to Fable and to my own round-1 step 2, which treated aliasing as derived from identical region texts: the mark stays, so the corpus keeps it as the declaration under test.
Evidence: plans/semantics/rulings.md:8-13 keeps `set_inplace`, says the mark changes no computed value, and makes an overlap without the mark a rejection; the audit row R5 says R5 reverses that ruling without saying so.
Evidence: R22's audit row shows `add_in_place` and `add_d1` declared on the same sum on purpose (act-backend/tests/test_extraction.py:405-414), so the mark selects between variants and is not derivable from the region texts alone.
Concession to Opus: the partial-overlap and equal-but-differently-spelled cases (RA2) get cases only after their verdict is ruled; the marked case needs no wait.
The case set, all oracle programs under Q6.1:

| Case | Setup | Expectation |
| ---- | ----- | ----------- |
| marked | input k marked, region text equal to the output | a read of the input region after the call returns the output |
| unmarked, equal | no mark, region text equal | the verdict the owner's norm gives (see Escalation); until then a pending row |
| partial overlap | input and output overlap, not equal | the RA2 fault, buffers and state unchanged |
| mark on a different region | mark names an input whose region differs from the output | rejected, with a clause id |

The backend half stays an HLO kernel with the marked input live afterwards, checked on simulated assembly with aliasing modelled.
That backend half runs nightly under R49, because it needs a backend build.
The tests at taidl/tests/test_instruction.py:273-291 (`TestInplace`) check the mark API and are kept and converted, not deleted; they stay `impl` tests because they are Python API facts.
The e-graph fact that a mark reaches the extractor (act-backend/tests/test_generators.py:243) stays a per-PR `impl` test.
Final position: option A, with the mark kept as the tested declaration and the unmarked-overlap row pending on the owner's norm.
Changed: yes, from "aliasing derived from region texts" to "mark kept as the tested declaration".
Confidence: medium.
What still divides the panel: Fable would drop the mark wording; Opus and I keep the corpus keyed to a spec promise (decision 17); the RA2 verdict is not yet ruled.

### Q6.3

All three debaters choose options A and C: fault cases are oracle-only, assert `expect = "fault"` plus unchanged buffers and state, and T13 adds one invariant that simulated backend assembly for a supported kernel never faults.
Final position: options A and C.
Confidence: high.
What still divides the panel: nothing.

### Q6.4

Strongest opposing argument (Opus): a case encodes the clause text of the spec version it ships with, and a clause edit and its case edit land in one pull request, so contested rows need cases now and gating them hides the clauses most likely to regress.
Rebuttal: co-versioning works when the clause text is settled, and the five audited rows (R5, R7, R8, R12, R56) are not.
A case written against an unsettled clause is a false red in both generators and forces a pin bump to repair (decision 11 caps the capacity for that).
Concession to Opus: the table row exists now and carries the S1.3 key, so the catalogue gap (QA4) closes this week, and a contested row is `pending`, not absent.
Narrowing: the in-place mark stays, so the R5 row is no longer wholly contested.
The mark row (marked, equal region, output visible through the input) has a standing verdict and gets cases now.
Only the partial-overlap, equal-spelled-differently, and unmarked-overlap parts of R5 stay pending.
On `implementation-defined` rows all three agree on option D (an allowed set or a tolerance), and the unmarked read-after-write is the first user of it.
The :712 and :749 seeds are oracle API layout errors and move to oracle `cli` tests (Opus, Fable, and the agenda agree).
Final position: options A and D, with B applied to the contested parts only; rows keyed by S1.3 id.
Changed: no; the contested set shrank because of the mark.
Confidence: medium.
What still divides the panel: Opus writes contested cases now and revises with the clause; Fable and I wait for the owning panel's ruling.

### Q6.5

Strongest opposing argument (Opus, Fable, on the shared side): the required match is `phase`, `clause`, and `error_type`, not a message regex; nobody argues for option C.
The only spread is option D (an optional `expect` table): Opus and Fable keep it, as do I.
Final position: options A, B, D, E, and F; C is rejected.
Confidence: high.
What still divides the panel: nothing material.

### Q6.6

Strongest opposing argument (Fable): the reference is NumPy or closed form for v3, with the StableHLO interpreter only as a nightly cross-check, because the interpreter is built from source, has no PyPI release, and may differ from the R6 pin.
Opus gives the same source with stored outputs.
Rebuttal: I already agree on the source; my round-1 wording of "the interpreter is the preferred source once approved" went further than the evidence supports.
Concession: the interpreter is a cross-check at most, and needs the same owner approval (R29, R45) as any dependency.
Evidence for stored data: the per-PR gate then imports nothing, so install and version risk of any reference tool stay off the pull request path.
Tolerance: `ULPS` is one value per type (act-oracle/tests/conftest.py:226-227: bf16 2, f16 2, f32 8, f64 8) and cannot bound a K-term accumulation, so the bound is per case (D) and implemented once in `taidl.testing` (F), with NaN, signed zero, and infinity stated once there (G).
Final position: stored expected data generated offline from NumPy or closed forms, named in `case.toml`; tolerance D with F and G; the interpreter as an optional approved cross-check.
Changed: yes, the interpreter moves from "preferred source" to "optional cross-check".
Confidence: medium.
What still divides the panel: whether the interpreter cross-check runs nightly (Fable) or waits for approval and a measured match (Opus, I).

### Q6.7

All three debaters choose options A, C, and D: a rendered `coverage.md` that fails on a zero row outside a short allowlist, a strict `xfail` row kind, and rows only for front-end-accepted cases.
One refinement from Opus: the allowlist holds clauses untestable by construction (S1.1, S1.2, S10) plus open rows such as S1.3.15, and a growing list is itself the alarm.
Final position: options A, C, and D.
Confidence: high.
What still divides the panel: nothing.

### Q6.8

All three debaters choose options A and C: delete a test only in the step whose runner covers its fact at the same cadence, name deletions by function, and turn tests of removed constructs into `rejected` cases that name the replacement.
One addition from the mark decision: `set_inplace` is not a removed construct, so its tests are converted to the v3 API, not turned into `rejected` cases.
The `set_apart` and linalg-word tests (act-backend/tests/test_refusals.py:383-396, act-oracle/tests/test_linalg.py:1) are the ones that become `rejected` cases.
Final position: options A and C.
Changed: refined (the in-place tests convert).
Confidence: high.
What still divides the panel: nothing.

### Q6.9

Strongest opposing argument (Opus): option E weakens the gate, because R17 asks for identical chosen assembly and candidate logs, and in deterministic mode byte goldens are stable while check lines detect strictly less.
Rebuttal in part (Fable, and me): byte goldens fail on every benign rename or reordering, so a review learns to regenerate them without reading, which weakens the gate in practice.
Concession in part: R17 states the structural-step gate as identity of chosen assembly and candidate logs (rulings R17), so that gate is byte identity in deterministic mode, and check lines cannot replace it.
Resolution: byte-identity goldens, in R17's deterministic mode, gate structural refactors; check lines over the instruction word and operand captures carry the per-PR slow tier and the semantic goldens.
Evidence: exact goldens today are `max_diff == 0` (act-backend/tests/test_mini_trn.py:39-42), and the search stops on a wall-clock limit (act-backend/act_backend/generic/src/lib.rs:18), which is why neither form is stable until R17's mode exists.
The three agree on B (a per-PR slow tier), C (the tier table), and D (deterministic mode plus a quarantine rule with an owner and a removal date, no rerun-until-green).
Tier budgets stay proposals until the first week of measurement: taidl 5 minutes, generator fast 10, generator slow 20, act nightly 90, release 3 hours.
Final position: options B, C, D, and E, with E narrowed to the semantic per-PR checks and byte identity kept for the R17 gate.
Changed: yes, E no longer replaces byte goldens.
Confidence: medium.
What still divides the panel: Opus rejects E outright; Fable would use check lines for all assembly goldens.

### Q6.10

Strongest opposing argument (Opus): all three techniques are nightly or periodic and behind owner approval, so nothing new enters the pull request.
Fable would run the grammar fuzzer (B) now and treat mutation testing as a bounded pilot.
Rebuttal: the fuzzer property "accepted, or rejected with a clause id" needs the Q6.5 runner, so B cannot start before that runner exists; "now" means nightly from the runner's first week, not earlier.
Concession to Fable: mutation testing (C) is a pilot on the front-end checks and the oracle fault checks, with each surviving mutant getting a case or a written reason, and never a pull request gate.
Per pull request, Hypothesis runs derandomized with a fixed seed; nightly runs random with the failure database kept as a CI artifact, and shrunk cases are promoted to corpus cases with an `origin` field.
Both B and C add test-only dependencies, so owner approval comes first (R45).
Final position: option A per pull request, B nightly with a time cap once the Q6.5 runner exists, C as a bounded pilot per release.
Changed: no; C is worded as a pilot.
Confidence: medium.
What still divides the panel: whether B starts before the Q6.5 runner (Fable) or after it (Opus, I).

### Q6.11

All three debaters choose options A, C, E, F, and G: corpus in the taidl wheel read through the pin, the kit split per R43, `spec.py` plus text sidecars, slug-only directory names with the S id in `case.toml`, and a `CHANGES.md` delta printed on a pin bump.
Final position: options A, C, E, F, and G.
Confidence: high on A, F, and G; medium on C.
What still divides the panel: nothing; Fable's trigger for a separate corpus repo (a second implementation outside the four repos) is not met.

### Q6.12

All three debaters choose options A, C, and D and reject E.
Evidence caveat: the `precision_config` HIGHEST claim (D) rests on the StableHLO spec, which Fable cites from memory and my fetch this round did not reach the `dot_general` section, so D carries a verification task before the first GPU run.
The GPU job is unavailable on hosted runners (A1 of round 1), so GPU checks are manual or on an owner-provided runner whatever the golden form.
Final position: options A, C, and D; keep the per-generator smoke test.
Confidence: medium.
What still divides the panel: nothing.

## Escalation to the owner

There is no counter to a binding decision.
One dependency is recorded for the owner.
Decision 16 (no undefined behavior) and the in-place input (an implementation may or may not reuse the storage in an unmarked overlap) together require a stated norm for an unmarked overlap of an input and an output at issue.
The test suite can encode it either as a fault (the semantics track's choice at plans/semantics/rulings.md:8-13) or as an allowed set of results; it cannot encode it as "unspecified".
The corpus row for that case stays `pending` until the norm is stated.
