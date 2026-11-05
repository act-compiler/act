# Panel 3 round 1: Sonnet, third-party generator author

## Assumptions

- The author builds a generator (simulator or RTL) outside the three repos and sees only the published front-end contract and registry.
- Python is the author's language for now; a later author may not be.
- Plan items are sized by the plan's own scale: S under a week, M one to three weeks, L over three weeks.
- Cited paths are relative to /home/user/ for code and to /home/user/act/plans/ for plans.

## External sources used

- MLIR diagnostics: https://mlir.llvm.org/docs/Diagnostics/ (every diagnostic has a location; a handler engine; `expected-error` tests match message text; no stable error codes).
- StableHLO compatibility: https://openxla.org/stablehlo/compatibility (guarantees only for portable artifacts; bytecode has none; 5 years backward, 2 years forward).
- MLIR ODS: https://mlir.llvm.org/docs/DefiningDialects/Operations/ (verifier phases; `InferTypeOpInterface` lets a tool ask an op for its derived result types).
- MLIR shape inference: https://mlir.llvm.org/docs/ShapeInference/ (unknown shapes need runtime checks; errors are reported only when guaranteed).
- Rust error index: https://doc.rust-lang.org/error_codes/error-index.html (stable per-code identifiers; gaps in numbering).

### Q3.1: Who owns the shape and per-word type rules, and how large is the item?

- Position: B, one item owned by S4, sized L, R31 checks listed in its What lines, with an internal order of move first, new checks second.
- Argument:
  1. The plan index already sizes S4 at L (shared/plan/README.md S4 row), and R62 merges four items into M with no stated saving.
  2. The audit finds the sizes L3 L, L12 L, O6 S, O7 L in the source plans, so the merged M undercounts by an order of magnitude.
  3. The move itself is mechanical: 500 lines in `act-oracle/taidl_to/generic/shapes.py:31-543` plus the word table at `oracle_visitor.py:19-97`.
  4. The new checks are not mechanical: R31 adds region values, constant ranges and compare-type fit, none of which exist as a shared copy today.
  5. A generator author needs one owner so the published rule set has a single version; split owners (option D) leave the cycle S4 -> L3 -> S4 unresolved.
  6. Verification of the move is cheap and independent: run old and new copy on the corpus and diff verdicts (S4 Fail points).
- Evidence: shared/plan/README.md S4 and S3 rows; shared/plan/S4.md What and Fail points; shared/bugs.md X1, X5; act-oracle/taidl_to/generic/shapes.py:58-62 (agree is a rule, not a table).
- Confidence: medium.
- Would change my mind: a line count of the R31 checks showing them under two weeks, or a spike showing the move needs rewrites.

### Q3.2: What do the backend items B21 to B23 become?

- Position: A, B21 to B23 become one S-sized item "call `validate()` first and delete `spec_checks.py:394-418`", removed from the S3, S4 and D4 dependency columns.
- Argument:
  1. The front end checks the same facts (S3 What lines 2 and 3), so a new backend copy is a duplicate with a known deletion date (option B).
  2. The existing backend check handles only all-literal regions (`act-backend/act_backend/spec_checks.py:394-418`, the `continue` at the `None in (tile, region)` test), so writing more is wasted.
  3. The dependency cycle (S3 says "before B21, B22" while listing them as dependencies) disappears if the backend item depends on S3 and not the reverse.
  4. An outside generator is told "call `validate()` first"; if the backend had its own interim checks the published contract would be unclear about which rejections are the front end's.
  5. Option C (drop all) leaves X3 and X4 open for the backend between S3 and the backend call, which is a state no one would ship.
- Evidence: shared/plan/S3.md Order and dependencies; shared/bugs.md X3 ("backend B22 is then a no-op"); act-backend/act_backend/spec_checks.py:394-418; draft R25 (backend keeps the literal check until L5 ships).
- Confidence: high.
- Would change my mind: a requirement that the backend run standalone on a description that was never validated.

### Q3.3: How are the Q1 items that depend on Q2 items resolved?

- Position: C, a mix: move L15 (the renames) to Q1 because S2 documents the contract under those names; decide L16 and L28 by size; S2 stays before B13 and the B13 dependency is dropped.
- Argument:
  1. A contract documented under old names and re-documented after the L15 renames means a third-party author reads two versions; S2 itself says it lands with the renames (S2 Caveats).
  2. Decision 19 and 20 fix the new vocabulary already, so L15 is a rename with owner-fixed targets and is small.
  3. Moving S2, S5, O8 and D10 to Q2 (option B) delays the contract and the corpus gate to the quarter that also holds S8; the release train would have no earlier contract test.
  4. The B13 dependency contradicts S2's own order line ("before B13"); one of them is wrong, and S2's order is the consistent one.
  5. Because decision 10 batches everything into one release, quarter placement only governs working order, not what ships; the cheapest fix is the one that removes arrows, not the one that moves the most items.
- Evidence: shared/plan/README.md (S2 depends on L15, B13); shared/plan/S2.md Caveats and Order; minutes/owner-decisions.md rows 10, 19, 20.
- Confidence: low; I have not read the L16 and L28 sizes.
- Would change my mind: L16 or L28 sized L, which would make moving them to Q1 infeasible under decision 11.

### Q3.4: What form does the front end to generator contract take?

- Position: D with A as the first step: a documented typed Python model plus a typed node model, trees marked unstable, `CONTRACT_VERSION`, and a stated trigger for a serialized form.
- Argument:
  1. Today an outside author must import ANTLR 4.9.2 to read the contract (taidl/pyproject.toml:15) and walk `IDLV2Parser.ModuleContext` (taidl/taidl/instruction.py:36-57).
  2. `instr_expression_asts` is a dict keyed by `TerminalNode`, so its meaning depends on the parse-tree identity, which no documentation can promise.
  3. Option B (document the trees) pins the contract to a parser version that the owner may regenerate; every grammar change becomes a contract break.
  4. A serialized form (option C) is a second compatibility promise; StableHLO guarantees compatibility only for its portable artifact and explicitly none for bytecode, and pays 5 and 2 year windows for it.
  5. A small typed node model (lines, operands, attributes, integer expressions) gives an outside author what a serialized form would, without a second promise.
  6. Plain typed dataclasses make a later JSON dump a mechanical add-on, so the trigger (second non-Python consumer) is cheap to honor.
- Evidence: taidl/taidl/__init__.py:9; taidl/taidl/instruction.py:34-57; shared/interface.md "Documentation of the contract"; https://openxla.org/stablehlo/compatibility; draft R42 and decision SD5 (shared/decisions.md).
- Confidence: medium.
- Would change my mind: an RTL generator team committed to a non-Python host language in v3.

### Q3.5: Which derived facts does `validate()` publish to generators?

- Position: A, `validate()` attaches result type and shape per line, folded sizes, the value-to-region map and the call site as documented typed members.
- Argument:
  1. After S4 the front end computes these facts; discarding them forces every generator to recompute, which is the duplication the track removes (interface.md "Where generators re-parse or re-derive").
  2. MLIR exposes derived result types through `InferTypeOpInterface` on the op itself, so a tool queries the op rather than re-running inference.
  3. Option B (separate result object) makes the pairing between an instruction and its facts a lookup the author must keep in sync.
  4. Option C (helper functions) leaves each generator calling `fold` and shape rules per use, with possible mismatched call order; it also exposes rule functions as API, a larger surface than facts.
  5. Option D leaves the author re-deriving shapes from `@c.` expressions, which the backend does today by regex (isel_ir2isa_rewrites_rs_generator.py:91, :134-137).
  6. The call site is needed for line 2 of a `U` message from an outside generator (scheme.md Message format), so it is a listed member (D2 stores it; S2 omits it).
- Evidence: shared/interface.md "Where generators re-parse or re-derive"; diagnostics/plan/D02-call-site-and-exceptions.md What; taidl/taidl/accelerator.py:25 (call-site mechanism via `inspect`); https://mlir.llvm.org/docs/DefiningDialects/Operations/.
- Confidence: medium.
- Would change my mind: evidence that derived facts differ per generator (for example element type promotion), which would make them generator-local.

### Q3.6: How is shape agreement decided when dims depend on `@c.` values?

- Position: C, keep the call-time check in the oracle and restate S4's benefit to the literal-dim cases; add no symbolic normalizer now.
- Argument:
  1. `agree` already treats a non-integer dim as "possibly equal" (act-oracle/taidl_to/generic/shapes.py:58-62), the same stance MLIR takes: errors are reported when guaranteed, otherwise a runtime check (ShapeInference doc).
  2. A symbolic normalizer (option A) needs an affine-equality decision procedure with its own corpus; no plan item sizes it.
  3. The backend check at selection (option B) needs a coded error in a path with no call; the backend's rewrite already emits shape checks (isel_ir2isa_rewrites_rs_generator.py:593-603), so a failed match is a non-selection, not an error.
  4. Decision 16 is satisfied because the oracle's call-time check raises a defined error (utils.py:771-786).
  5. S4 must say that the backend may accept a description whose `@c.` shapes disagree and select nothing; the unsupported table states that assumption.
  6. A cheap partial step is allowed: treat two dims that are the identical expression text as equal; no other rewriting.
- Evidence: act-oracle/taidl_to/generic/shapes.py:58-62; shared/plan/S3.md Caveats; shared/bugs.md X1, X10; https://mlir.llvm.org/docs/ShapeInference/.
- Confidence: medium.
- Would change my mind: a case study whose real accelerator shapes are all affine in `@c.` and need generation-time rejection.

### Q3.7: How does the generated oracle reach the shape rules?

- Position: A, import at run time at the pinned tag, and require `taidl.shapes` and `taidl.diagnostics` to import without the parser.
- Argument:
  1. A generated oracle that loads ANTLR to check two tuples is a heavy dependency for the person running a kernel.
  2. `import taidl.shapes` first runs `taidl/__init__.py:5-7`, which imports `Accelerator`, and `instruction.py:4` imports `antlr4`, so option B loads the parser (taidl/taidl/instruction.py:4).
  3. The shape rules take `(type, dims)` tuples, not trees (act-oracle/taidl_to/generic/shapes.py:45-62), so parser independence is feasible.
  4. The cost of A is a lazy package init (exports resolved on first use) or a split distribution; this is a one-time change in `__init__.py`.
  5. Vendoring (option C) copies the rules into every generated oracle, and a copy outside the pin is the stale-copy problem R39 exists to avoid.
  6. For a third-party generator the same rule holds: it imports the rules at the tag it pins, and it needs the same parser-free entry.
- Evidence: taidl/taidl/__init__.py:5-9; taidl/taidl/instruction.py:4; act-oracle/taidl_to/generic/utils.py:771-786; draft R39.
- Confidence: medium.
- Would change my mind: measured import time of ANTLR under 100 ms in the oracle's target runtime, which would make the parser-free demand a nicety.

### Q3.8: Where does the clause id appear in a message, and what does it name?

- Position: placement A (suffix `[S<id>]` on line 1, line 3 lists further sections only); namespace D with E as its row form (`S9.4` is a section id).
- Argument:
  1. Line 1 is the line that logs, CI summaries and `grep` keep; a clause id on line 3 is lost by any tool that shows only the first line.
  2. The stable key is the `code`; the clause id is the human link to the spec, so a single suffix is enough and line 3 would only repeat it (option C duplicates).
  3. MLIR has no codes and tests match message text; Rust uses a code per rejection; ACT's registry holds one `spec` field per code, so one clause id per message is the registry's shape.
  4. If L numbers the S9 rows as `S9.n`, namespaces D and E coincide: an S9 row is a spec section id.
  5. The `coverage.md` clause id (option F) is a generator's namespace and must not enter front-end messages, or the front end depends on generator documents.
  6. `TaidlError.spec` already carries the id as a field (scheme.md Exceptions), so an outside generator reads the field and ignores the text.
- Evidence: diagnostics/scheme.md Message format and Registry; diagnostics/registry-seed.md Spec column; tests/generators.md T11 to T13; https://mlir.llvm.org/docs/Diagnostics/; https://doc.rust-lang.org/error_codes/error-index.html.
- Confidence: medium.
- Would change my mind: a clause that is not a section (a row split across sections), which would need an id outside the S namespace.

### Q3.9: What do expectation rows and tests match, and who renders `coverage.md`?

- Position: D for the match key (code assertions everywhere, one text test per format rule, one golden three-line test per class); E for rendering (one renderer owned by T11).
- Argument:
  1. A code is the only identifier a generator author can match without parsing prose; the registry never renumbers a code (scheme.md Classes).
  2. Regex of message text breaks on any wording change (E16 records this); MLIR's `expected-error` regex matching accepts that cost because MLIR has no codes, ACT has them.
  3. Pure code assertions (option A) never test the wording rules (one sentence, no final period, 160 characters), so the format rules need their own tests.
  4. The `object` field is part of the contract only if its text form is fixed; golden tests per class pin it once.
  5. Two renderers (option F) produce two files that must agree on the same clause list; one renderer reading the registry for text and expectations for verdicts has one place to be wrong.
  6. A third-party generator publishes its own expectations table in the same format and gets its coverage page from the same renderer.
- Evidence: diagnostics/bugs.md E16; diagnostics/plan/D10-tests-and-lint.md; tests/generators.md T11, T12; tests/corpus/format.md:9, :22; https://mlir.llvm.org/docs/Diagnostics/.
- Confidence: medium.
- Would change my mind: a T11 renderer that needs generator-specific logic beyond the registry and expectations table.

### Q3.10: What is the run-time error class called, and what does it guarantee?

- Position: D, the class is Fault with the no-write rule stated in the norm and in D8's tests, and the class letter `V` is renamed to a letter other than `F`.
- Argument:
  1. A third-party author needs one name in prose, in the exception class and in the code letter; two names (Violation class, fault outcome) are a glossary entry the author has to learn (option B).
  2. "Fault" is the hardware-architecture word for a defined run-time error and fits decision 14's order (owner's papers, then StableHLO and MLIR) for an ISA language.
  3. The no-write guarantee is the property a simulator or RTL author must implement: an instruction decides every fault before any write, or stages writes; the scheme states no such guarantee today (scheme.md Run-time norm).
  4. The front end already uses `Fault = tuple[CommonToken, str]` for a parse-time finding (taidl/taidl/operations.py:167-168), so option C (Fault for every class) would make that name ambiguous with the run-time one.
  5. Keeping the front-end finding as "rejection" (class R) and only run-time cases as Fault removes the overlap (option A).
  6. Scope letter `F` (front end) next to a class letter `F` would give codes like `FF001`; the class letter changes (option D) so code reading stays unambiguous.
- Evidence: diagnostics/scheme.md Classes, Run-time norm, Exceptions; diagnostics/plan/D14-run-time-norm.md; taidl/taidl/operations.py:167-168; minutes/owner-decisions.md rows 14, 16; draft R55.
- Confidence: medium.
- Would change my mind: an owner paper that uses "violation" as the precise term (decision 14 puts papers first).

### Q3.11: Which scope letter do shared fault codes take?

- Position: A, fault codes take scope `F` in `taidl/diagnostics.json` and generators reuse them.
- Argument:
  1. A fault is a language rule; both generators report it and T13 compares across generators, so the code must name the rule, not the raiser.
  2. The R codes already use `F` so two generators reject one description with one code (scheme.md Classes); the same reasoning applies to faults.
  3. Per-raiser codes (options B, C) give the same rule two codes (`VO005`, `VB001` for the 64-bit range, per E13), and a third generator would add a third.
  4. A third-party generator can reuse `F` codes without registering anything; with per-raiser codes it needs its own scope letter and a mapping row for T13 (option C).
  5. The scheme's registry already puts "shared V" codes in `taidl/diagnostics.json` but has no scope for them; scope `F` closes that gap with the least change.
  6. A rule that only one generator can decide (for example `VB001`, decidable at generation) is still a `F` fault; the generator raises it earlier, not under a different code.
- Evidence: diagnostics/registry-seed.md:87-93; diagnostics/scheme.md Classes and Registry; diagnostics/bugs.md E13; tests/generators.md T13.
- Confidence: high.
- Would change my mind: a fault whose condition differs between generators in a way the rule text cannot express.

### Q3.12: How does a rejected v3 construct report its replacement, and when are its codes seeded?

- Position: C and E, one migration note in the S8 release and a generic rejection; seed codes only after the v3 API is fixed.
- Argument:
  1. The seed codes RF013 to RF017 describe `set_inplace` and `set_apart`, which v3 removes before first release; a code is never reused or renumbered, so seeding them creates retired codes on day one (scheme.md Classes).
  2. A removed Python method raises `AttributeError` without any code; a coded replacement message needs a stub method kept in the API, which engineering.md forbids as a "removed" placeholder and which an outside author must then read in the contract.
  3. A converter (option B) is not sized and capacity is fixed (decision 11); 3 to 4 engineers cannot absorb it.
  4. Text-level removals (the words `cholesky` and others, R60) are already rejected by the OPERATIONS lookup; the existing unknown-word message can carry the S8 pointer.
  5. For the third-party author the removed constructs never appear in the contract, so this question costs them nothing; the cost is the description writer's, and a release note serves them.
  6. Seeding after the API is fixed also settles the numbering once, since R2, R5, R58 and R60 each rename or remove names the codes cite.
- Evidence: diagnostics/registry-seed.md RF013 to RF017; diagnostics/plan/D04-move-checks-to-front-end.md Caveats; shared/plan/S8.md; draft R5, R58, R60; minutes/owner-decisions.md rows 10, 11; .claude/engineering.md "Things to avoid" (delete cleanly).
- Confidence: low; option A is defensible if the owner values an error that names the replacement more than a clean API.
- Would change my mind: a count of v2 descriptions in the wild calling the removed constructs, which would raise the value of a coded replacement message.
