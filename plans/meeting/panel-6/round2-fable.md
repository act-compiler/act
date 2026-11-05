# Panel 6, round 2, Fable: industry conformance suites

Lens: how WebAssembly spec tests, Test262, LLVM lit and FileCheck, riscv-arch-test, and the StableHLO interpreter suite are built and kept green.
Owner input in force: the in-place mark stays unless shown truly redundant, and overlap alone does not settle read-after-write.

### Q6.1

All three take option A, a `program.toml` run artifact beside `kernel.hlo`, with the runner matching the fault clause id and error class rather than the message.
Final position: option A.
Confidence: high.
Nothing divides the panel.

### Q6.2

Strongest opposing argument: Sonnet takes option C as the floor, since the e-graph reachability fact at act-backend/tests/test_generators.py:243 and the dropped-candidate check at test_mini_trn.py:78 are search facts no result check sees; Opus writes the RA2 cases only after a verdict exists.
Rebuttal and concession, in three parts.
First, my round-1 wording "drop the mark" is withdrawn under the owner input; the case contract is the mark, not R5's identical-region-text derivation.
The evidence that the mark is not redundant is in the code: `inplace_fix.rs:59-74` reads `inplace_children()` to decide whether an overwritten operand with another reader must be copied, and `act-backend/tests/test_extraction.py:507` marks a load from `d0` into `d2` in place, where no region text is shared.
In `r0 = r1 + r0` the oracle and the backend can legitimately differ on storage reuse without the mark, so a corpus case that asserts read-after-write must carry the mark explicitly.
Second, I concede Sonnet's point about the e-graph facts: `test_input_marked_in_place_reaches_the_egraph` is a generator-internal fact and stays an `impl` test under T20.
Third, option A's backend half is already executable: `test_extraction.py:461-468` compiles `IN_PLACE`, where `exp_in_place` overwrites one load of x while `add_d1` reads the other, and checks the result through `simulated`, which is the read-after-write check on assembly that T13's HLO wording cannot express.
The oracle half is a Q6.1 program: a marked input, the call, then a read of the input region expecting the output.
Unmarked overlap, equal regions spelled differently, and partial overlap (H1, RA2) get cases only once panel 1 rules, as Opus says; S1.3.6 (marked input whose starts differ from output 0 at issue) is defined now and gets a fault case now.
Final position: option A with the mark as the declared aliasing in every case, S1.3.6 now, RA2 cases after the ruling, e-graph facts kept as `impl` tests.
Confidence: medium-high.
Still divides: whether R5's "set_inplace removed" survives the owner input at all; that is panel 1's question, and the corpus format must leave room for the mark either way.

### Q6.3

All three take options A and C together and rename the outcome `error` to `fault`.
Final position: options A and C, with `fault` as the outcome name.
Confidence: high.
Nothing divides the panel.

### Q6.4

Strongest opposing argument: Opus keys the norm table by the S1.3 row ids already numbered in spec-draft/overview.md:37-51 and rejects option B, because a contested row with zero cases hides the clause most likely to regress; the case follows the clause text that ships and changes in the same PR as the clause.
Rebuttal and concession.
I concede the key: S1.3 already enumerates fifteen rows with kind and section, so the norm table is that table plus a `cases` column and a per-generator expectation, and a parallel ruling-keyed catalogue would drift from it.
I do not concede that contested rows get cases now.
The owner input of this round shows why: S1.3.6 encodes the mark semantics of R5, and R5 is now in question, so a case written today against that row text may encode a norm that changes within the quarter.
Opus's remedy (edit clause and case in one PR) works only if the corpus and spec draft are co-versioned, which Q6.11 option A gives, but it still charges the migration twice under owner decision 11.
Test262's stage-3 rule is the precedent: a row exists, carries its status, and gets cases when the semantics settle.
So contested rows (those behind R5, R7, R8, R12, R56, and the open S1.3.15) sit on the Q6.7 allowlist with a pending reason, which keeps them visible, as Opus wants, without a case that may need rewriting.
Option D is uncontested: an `implementation-defined` case carries an allowed set or a tolerance.
Final position: option A keyed by S1.3 row ids, option B for the contested rows only, option D.
Confidence: medium.
Still divides: whether a contested row carries a case against the current text (Opus) or an allowlist entry until the ruling (Sonnet and I).

### Q6.5

All three take options A, B, E, and F and reject C; all three take D, Sonnet with medium confidence on D versus deleting only verdict blocks.
Final position: options A, B, D, E, and F.
Confidence: high.
Still divides: only the size of D, which is settled by counting the test_grammar.py blocks whose assertion is a parsed value.

### Q6.6

Strongest opposing argument: Sonnet and Opus store the expected outputs as data in the case, generated offline by a named reference, so the per-PR gate imports no reference library and R29's dependency problems leave the CI path; riscv-arch-test embeds Sail results and Wasm `assert_return` carries the value in the script.
Concession.
My round-1 position ran NumPy at test time, which makes every PR depend on NumPy's semantics for the word under test and re-decides NaN and signed-zero handling in each run.
Stored outputs are the form of every suite in my lens, they make the gate deterministic, and they turn the reference question into a regeneration-tool question that the owner can answer later without blocking the first corpus PR.
What I keep from round 1: the regenerator for v3 is NumPy or closed forms, named with its pinned version in `case.toml`, since the interpreter has no PyPI release and needs CHLO legalization (audit R29) and a new dependency needs approval (R29, R45); the interpreter becomes a regeneration source only after approval and a measured match on every S8 word.
Tolerance per case (D) with the spec owning the bound and `taidl.testing` the implementation (F), and NaN, signed zero, and infinities stated once with an optional `nan = "any"` (G), are shared by all three.
Final position: stored expected data per case; option B as the v3 regenerator named in `case.toml`; options D, F, and G.
Confidence: medium-high.
Still divides: nothing material; only whether the interpreter ever becomes the normative regenerator, which waits on the owner.

### Q6.7

All three take options A, C, and D and reject B and E, with `xfail` strict so an XPASS fails.
Final position: options A, C, and D.
Confidence: high.
Nothing divides the panel.

### Q6.8

All three take options A and C: delete a test only when its replacement has run green at the same cadence, name deleted tests by function, and convert tests of removed constructs into `rejected` cases naming the replacement and clause id.
Sonnet adds a test-id mapping script and a local break-the-guard check per deleting PR; I endorse both as the LLVM rule that a regression check never leaves before its successor runs.
Final position: options A and C.
Confidence: high.
Nothing divides the panel.

### Q6.9

Strongest opposing argument: Opus rejects option E, because in R17's deterministic mode byte goldens are stable and check lines detect strictly less, so E weakens the gate R17 asks for.
Partial concession.
Opus is right about strictness: R17 gates a structural step on identical chosen assembly and candidate logs, and a check line with captures cannot be stricter than byte identity.
So the R17 gate keeps its byte golden of the chosen assembly and the candidate log.
E stands for the corpus-side T13 to T15 cases, where the fact under test is which instruction word was chosen and how its operands relate, not the printer's spacing or a mnemonic rename; FileCheck exists because LLVM learned that byte-identical IR goldens break on every printer change.
Sonnet's evidence settles the rest: the search stops on a 1000 ms `TIME_LIMIT` (generic/src/lib.rs:18, pipeline.rs:157), so neither golden form is stable until R17 mode exists, and T13 to T15 must name that mode.
Sonnet's tier table (C) with measured budgets in the first week and the quarantine rule (D) are shared by all three; the per-PR slow tier (B) is shared, pending measurement.
Final position: options B, C, and D; E for corpus cases, byte goldens for R17's own gate.
Confidence: high for C and D, medium for B and the E split.
Still divides: whether corpus-side assembly checks are byte goldens (Opus) or check lines (Sonnet and I).

### Q6.10

Strongest opposing argument: Sonnet runs mutation testing (C) once per release on the front-end and fault checks only, never gating a PR; Opus runs it periodically behind owner approval.
No real opposition: my round-1 pilot is the same scope and cadence, with each surviving mutant getting a case or a written reason.
All three keep A per PR with `derandomize`, B as a nightly fuzzer with the property "accepted, or rejected with a clause id", and owner approval before either new tool lands.
Final position: options A and B now, C per release on the front-end and fault checks after owner approval.
Confidence: medium.
Nothing material divides the panel.

### Q6.11

All three take options A, C, E, F, and G.
Final position: options A, C, E, F, and G.
Confidence: high for A, F, and G, medium for C.
Nothing divides the panel.

### Q6.12

All three take options A, C, and D and reject E.
Sonnet's assumption A1 (hosted runners, no GPU) means D is tested only when a GPU run exists, so the `precision_config` HIGHEST line lands with a manual GPU check and the QA15 golden policy.
Final position: options A, C, and D.
Confidence: medium.
Still divides: nothing beyond the GPU runner's availability.

## Escalation to the owner

None.
The one open point touching the owner's in-place input is for panel 1, not the owner: R5 removes `set_inplace` while the input keeps the mark, and `test_extraction.py:507` marks a cross-buffer load in place, so S1.3.6's start-equality rule needs restating for a mark whose input and output live in different buffers.
