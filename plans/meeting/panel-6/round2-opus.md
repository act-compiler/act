# Panel 6 round 2: Opus (conformance purist)

Lens: every spec clause has cases, and every generator's behavior on every case is declared and checked.
Binding input: the in-place mark stays (owner decision 17), unless shown truly redundant; overlap alone does not settle read-after-write.
New evidence read this round: rulings.md R5, R17, R49; spec-draft/instructions.md:47-55; spec-draft/conformance.md:29-31; spec-draft/overview.md:42; spec-draft/glossary.md:46.

### Q6.1

All three agree on option A (`program.toml` run by the oracle, `kernel.hlo` kept for the backend); no rebuttal needed.
- Refinement taken from Sonnet: the runner matches the S1.3 row id and error class, never the full message, since test_kernel.py:598 compares the message with `==`.
- Final position: option A, matching on row id and error class. Unchanged.
- Confidence: high.
- What still divides the panel: nothing.

### Q6.2

Strongest opposing argument (Fable, and Sonnet's step 2): R5 derives aliasing from identical region texts, so the corpus should drop the mark wording and test the derived form; the `set_inplace` seeds will not parse in v3.
Rebuttal:
- The binding owner input keeps the mark, and the spec draft still defines it: S5.5 has `set_inplace(inplace)` with one `bool` per input region (spec-draft/instructions.md:47-55).
- The draft also checks it twice: at build, an in-place input whose buffer or sizes differ is a `ValueError` (spec-draft/conformance.md:30); at issue, differing starts are fault S1.3.6 (spec-draft/overview.md:42).
- Text identity cannot replace the mark: S5.5 compares region texts "by tokens ignoring whitespace", and S1.3.6 exists because two such regions can still have differing starts at issue.
- Overlap does not settle read-after-write: in `r0 = r1 + r0` an unmarked implementation may compute into fresh storage and copy, or write in place.
- Only the mark makes the reuse a promise both generators must keep (decision 17), so the mark is observable and not redundant.
- I withdraw my round-1 step 6, which converted seeds "to the derived form"; the `set_inplace` seeds stay as v3 mark cases, updated only for any syntax change L makes.
Corpus cases this gives:
- Oracle, marked: a program issues a marked call, then reads the marked input region and expects the output (storage reuse made observable).
- Oracle, marked with differing starts at issue: `fault` S1.3.6 with unchanged buffers and state (Q6.3).
- Oracle, build-time: the three S9 rows at conformance.md:29-31 as `rejected` cases.
- Oracle, unmarked overlap: a case only once panel 1 or RA2 gives the verdict; until then a pending norm row (Q6.4).
- Backend: an HLO kernel whose marked input is still live after the instruction; the oracle-simulated assembly must match the reference, so a backend that reuses storage while the value is live fails.
- Backend, reuse kept: the e-graph and allocator facts (test_generators.py:243, test_extraction.py:507) stay per-PR `impl` tests, as Sonnet argues; they are facts about search, not results.
- Final position: option A with the mark kept as written in S5.5; option C only for search facts. Changed (seed conversion target), not the option.
- Confidence: medium-high; the oracle's aliasing model is still unbuilt (Sonnet).
- What still divides the panel: Fable's "drop the mark wording", which the owner input overrides; whether the R5 derivation survives at all is panel 1's question.

### Q6.3

All three agree on options A and C together, plus renaming `error` to `fault`.
- Fable's limit is right: the snapshot assertion is the precedent of riscv-arch-test's full-signature compare, not of Wasm `assert_trap`.
- Final position: A and C. Unchanged.
- Confidence: high.
- What still divides the panel: nothing.

### Q6.4

Strongest opposing argument (Sonnet, with Fable): a case encoding a contested norm becomes a false red in two repos plus a pin bump when the ruling changes, and five rulings (R5, R7, R8, R12, R56) are marked questionable now.
Rebuttal and partial concession:
- Concede the cost: case ids are keys in expectation rows (generators.md:80), so churn is not one edit.
- Keep the gap visible: a contested norm gets its table row now, marked `pending` with the ruling id, and is listed by the coverage renderer.
- Add a release gate: v3 cannot ship with a pending row, because decision 16 requires every run-time violation to have a defined verdict, so pending is a work queue, not a resting state.
- Uncontested S1.3 rows (spec-draft/overview.md:37-51) get cases now, keyed by row id.
- Option D stays for any `implementation-defined` row (Wasm `nan:canonical` and `nan:arithmetic`; Fable's float reduction order within the H6 tolerance).
- Final position: A plus D, with B gating only the contested rows, and a release gate that fails on any pending row. Changed: I accept B for the contested rows.
- Confidence: medium.
- What still divides the panel: whether the release gate is mine alone; Sonnet and Fable leave pending rows ungated.

### Q6.5

All three agree on A, B, E, F, with D (Sonnet and Fable call D optional; I keep it as optional too); C is rejected.
- Fable's fallback (collapse `lex` into `parse` if ANTLR cannot separate them) is acceptable; phase stays required.
- Final position: A, B, D, E, F. Unchanged.
- Confidence: high.
- What still divides the panel: nothing material.

### Q6.6

Strongest opposing argument (Fable): the StableHLO interpreter is the executable spec, so it should run nightly as a cross-check of the NumPy references; Sonnet makes it the preferred source of stored values once approved.
Rebuttal or concession:
- Concede: a nightly cross-check catches a NumPy reference that disagrees with the StableHLO spec, which is the exact failure my round-1 "would change my mind" named.
- It stays behind owner approval (R29, R45), and off the per-PR path, since all three agree expected values are stored data in the case.
- Final position: stored expected values, generated by NumPy or closed forms (B), `reference` and version in `case.toml`, the interpreter as a nightly cross-check once approved; D, F, G. Changed: I add the nightly cross-check.
- Confidence: medium.
- What still divides the panel: whether an approved interpreter replaces NumPy as the generator (Sonnet) or only cross-checks it (Fable, me).

### Q6.7

All three agree on A, C, D; B and E rejected.
- Sonnet's strict `xfail` (pytest `strict=True`, lit XPASS fails) is the form I meant.
- Final position: A, C (strict), D. Unchanged.
- Confidence: high.
- What still divides the panel: allowlist size (Fable warns S9 may be mostly untestable before L numbers rows); option E of Q6.5 (`pending` clause) answers that without enlarging the allowlist.

### Q6.8

All three agree on A and C.
- Sonnet's extra rule is worth taking: a PR that deletes tests lists their function names and replacement ids, and a script fails if a test id disappears unmapped.
- Final position: A and C, plus the mapped-deletion check. Unchanged in substance.
- Confidence: high.
- What still divides the panel: nothing.

### Q6.9

Strongest opposing argument (Fable, Sonnet): byte goldens fail on any incidental renaming, while FileCheck-style check lines assert the instruction word and operand relations with captures; R17 needs determinism, and check lines express the selection facts directly.
Rebuttal and partial concession:
- R17 itself says the gate is "identical chosen assembly and candidate logs" in a deterministic search mode (rulings.md R17), so determinism is the property, and an identity check is its only exact test.
- But a stored byte golden tests two things at once: determinism, and that the chosen assembly is the expected one; only the first is a conformance fact.
- The result is already checked by oracle simulation against the reference; the assembly shape is a backend property.
- So I concede E for stored goldens: check lines for the selection facts a test means to pin.
- Determinism is then checked directly: run deterministic mode twice in the same job and require byte-identical assembly and candidate logs, which is R17's gate without a stored file.
- Sonnet's amd64 versus arm64 point applies: the identity check runs per architecture.
- Final position: B, C, D, and E for stored goldens, plus a same-job double-run identity check in R17 mode. Changed: I accept E.
- Confidence: medium.
- What still divides the panel: tier budgets are unmeasured (Sonnet's table is a proposal); whether R17 mode exists in time for T13 to T15.

### Q6.10

All three take A and B; they differ on C's timing.
- Strongest opposing argument (Sonnet): a full mutation run executes the suite once per mutant, so C must be narrow, per release, and never gate a PR.
- Concede the scope: front-end checks and the oracle's fault checks only (QA11), as Fable's pilot also limits it.
- Keep the outcome rule: each surviving mutant gets a case or a written reason, otherwise the run measures nothing.
- Final position: A per PR (derandomized), B nightly with a time cap, C per release on the narrow scope; all new tools behind owner approval. Changed: cadence fixed to per release.
- Confidence: medium.
- What still divides the panel: Fable's "Q2 pilot" versus Sonnet's "per release"; these are compatible if the pilot is the first release run.

### Q6.11

All three agree on A, C, E, F, G.
- Final position: A, C, E, F, G. Unchanged.
- Confidence: medium on C, high on the rest.
- What still divides the panel: nothing.

### Q6.12

All three agree on A, C, D; E rejected; per-generator smoke tests kept.
- Sonnet's golden policy (QA15) is acceptable as the author note: regenerate only in a PR naming the semantic change, on CPU.
- Final position: A, C, D. Unchanged.
- Confidence: medium.
- What still divides the panel: no GPU runner exists (Sonnet A1), so D's test is manual until one does.

## Escalation to the owner

None; the owner input on the in-place mark matches the spec draft's S5.5, and no evidence this round shows the mark redundant.
