# Panel 3, round 1, Fable: compiler infrastructure architect

Lens: the boundary between one front end and many back ends, as MLIR, StableHLO and LLVM draw it.
Paths are relative to /home/user; plan paths are relative to act/plans/.
External sources are named by host and path, plain text.
Standing assumption: both generators stay Python through v3 (shared/decisions.md SD5), and v3 is one coordinated release (owner decision 10).

### Q3.1

Position: option B, one item owned by S4, sized L, with the R31 checks named in its What lines, and the module move and the front-end checks as two PRs inside it.

1. The one-checker rule is the Sema rule: Clang's parser hands every construct to Sema, which alone "performs the actual semantic analysis", and StableHLO keeps every op constraint in one TypeInference.cpp with the spec id beside each check (`// reduce_c4`).
2. One owner is therefore right, and the cycles L3-O6, L12-O7, O7-B23 vanish when S4 is the only item and L3 is its spec text, written from the moved table rather than before it.
3. The size is not M: the move alone covers act-oracle/taidl_to/generic/shapes.py:31-543 and oracle_visitor.py:19-97, the front-end checks RF040 to RF050 in D4 add region values, constant ranges and compare-type fit (R31), and the oracle's `agree` accepts any non-integer dim, so the front end needs new symbolic work (Q3.6).
4. Two sequential items (C) would recreate the cross-item dependency the audit flags; two PRs under one owner give the same sequencing with one index row.
5. B23 is dropped and becomes the Q3.2 redefinition; D4 keeps only its wiring and code rows.

Evidence: clang.llvm.org/docs/InternalsManual.html (Sema section); raw.githubusercontent.com/openxla/stablehlo/main/stablehlo/dialect/TypeInference.cpp; reviews/shared-diagnostics/review.md SD1; reviews/rulings/audit.md:35; shared/plan/S4.md; diagnostics/plan/D04-move-checks-to-front-end.md; act-oracle/taidl_to/generic/shapes.py:58-62; language/plan/spec.md:12; oracle/plan/checks.md:10-11.
Confidence: medium.
Would change my mind: a line count of the moved module plus RF040 to RF050 showing under three engineer-weeks, which is the plan index's M bound (diagnostics/plan/README.md:3).

### Q3.2

Position: option A, with R25 kept as a no-op clause: B21 to B23 become "call `Accelerator.validate()` first and delete `spec_checks.py:394-418`", the existing literal check stays untouched until S3 ships, and no new backend check is written.

1. MLIR dialect conversion decides legality in one `ConversionTarget`; a pattern never re-decides what the target already ruled, and that is the role `validate()` takes for both generators.
2. The scheme already says a generator that meets an `R` condition after D5 raises `I`, so a backend copy written now is deleted by D5 within the same release.
3. `spec_checks.py:394-418` only compares literal element counts (`if None in (tile, region): continue`), so keeping it unchanged until S3 lands costs nothing and satisfies R25.
4. Writing B21 to B23 as interim checks (B) spends S plus S plus M of a two-quarter budget on code with a known delete date.
5. Dropping them entirely (C) loses the one backend task that remains: the call to `validate()` and the deletion, which must be scheduled and tested (`test_refusals.py:439` no-output rule).

Evidence: mlir.llvm.org/docs/DialectConversion/ (ConversionTarget legality); diagnostics/scheme.md "Which layer raises what"; act-backend/act_backend/spec_checks.py:394-418; backend/plan/defects.md:9-11; shared/plan/S3.md; minutes/rulings.md:49 (R25); owner decision 11.
Confidence: high.
Would change my mind: a backend-only user flow (act/act/generators/__init__.py:13-15) that must ship before S3 and needs the type check of X3 in that interval.

### Q3.3

Position: option C, per item: L15 moves to Q1 ahead of S2; L16 is already Q1 so S5 and O8 need no move; D10 drops its L28 dependency and keeps the word list; S2 precedes B13 and B13 leaves S2's dependency column.

1. A contract is documented once under its final names, as LLVM's policy treats a rename as a "potentially breaking change" announced once; S2 after L15 is the only order that documents each member once.
2. L15 is sized S (language/plan/frontend.md:15), so moving it to Q1 is cheap, while moving S2, S5, O8 and D10 to Q2 (B) pushes the contract and the literal check behind every generator item that reads them.
3. L16 sits in Q1 at language/plan/frontend.md:16, so the SD3 claim about L16 is a stale row elsewhere, not a cycle.
4. L28 is "after v3" (language/plan/terminology.md:13); D10's lint needs only the word list file, so D10 owns the list and L28 consumes it.
5. B13 builds its model from the front end's objects (backend/plan/generators.md:11), so it reads the contract and cannot precede it.

Evidence: llvm.org/docs/DeveloperPolicy.html (potentially breaking changes); language/plan/frontend.md:15-16; language/plan/terminology.md:13; backend/plan/generators.md:11; shared/plan/S2.md (Caveats, Order); shared/plan/README.md:8; diagnostics/plan/README.md:17; owner decisions 10 and 11.
Confidence: medium.
Would change my mind: an L25 dependency of L15 that itself cannot land in Q1.

### Q3.4

Position: option D with A as the now part: a typed node model for lines, operands, attributes and integer expressions, trees reachable but unstable, a `CONTRACT_VERSION` and a per-member deprecation rule, and the second non-Python consumer as the stated trigger for a serialized form.

1. MLIR's ODS exists because generic access such as `getOperand(3)` is "error prone" against a typed accessor like `getStride()`; a dict keyed by `TerminalNode` and `IDLV2Parser.ModuleContext` are the generic access, and the typed node model is the ODS layer.
2. The current contract pins every consumer to `antlr4-python3-runtime==4.9.2` and to grammar rule names, so a grammar refactor is a breaking change for every generator today.
3. StableHLO gives compatibility guarantees "only for portable artifacts", which is the cost of a serialized form: a versioned dialect, a bytecode and a compatibility test per release; that machinery pays off only with consumers that cannot share a process.
4. LLVM keeps textual IR unstable and bitcode stable; the analogue is trees unstable, typed model stable, with a version constant so a generator can assert the contract it was built against.
5. Option B pins what interface.md shows is undocumented today (`comp_attr`, `inplace`, `placed`, the magic `d0`), so documenting it as the contract freezes the wrong surface.

Evidence: mlir.llvm.org/docs/DefiningDialects/Operations/ (motivation); openxla.org/stablehlo/compatibility; llvm.org/docs/DeveloperPolicy.html (IR backwards compatibility); taidl/pyproject.toml:15; taidl/taidl/__init__.py:9; taidl/taidl/instruction.py:46-57; shared/interface.md "Documentation of the contract"; shared/decisions.md SD5; minutes/rulings.md:75 (R42).
Confidence: high.
Would change my mind: a committed non-Python generator (a Rust or C++ front end consumer) in the v3 window.

### Q3.5

Position: option A, with the SD3 helpers kept: `validate()` attaches per-line result type and shape, folded sizes, the value-to-region map and the call site as typed members, and `fold` and `literal_fits` stay exported functions.

1. StableHLO type inference writes the inferred result types onto the op; an MLIR consumer reads `getType()` on the value, not a side table, and the ODS adaptor exposes the same accessors.
2. A separate analysis object (B) forces every generator to carry two handles per instruction and to key one by the other, which is the dict-order fragility of X7 again.
3. Helpers alone (C) let each generator call them in a different order or with different inputs, which is how three folders and two literal checks diverged (X6, X8).
4. The call site is stored by D2 and needed for line 2 of every message a generator raises, so it is a read member and belongs in the S2 table (SD16).
5. Publishing nothing (D) keeps the backend re-deriving shapes by regex over raw text (isel_ir2isa_rewrites_rs_generator.py:91, :134-137), the duplication the track removes.

Evidence: raw.githubusercontent.com/openxla/stablehlo/main/stablehlo/dialect/TypeInference.cpp; mlir.llvm.org/docs/DefiningDialects/Operations/ (adaptors and accessors); shared/interface.md "Where generators re-parse or re-derive"; shared/bugs.md X6, X7, X8; diagnostics/plan/D02-call-site-and-exceptions.md; shared/decisions.md SD3; reviews/shared-diagnostics/review.md SD5, SD16.
Confidence: medium.
Would change my mind: a measurement that the attached members make `Accelerator` construction slow enough to hurt the oracle's import time, which would argue for lazy computation behind the same member names.

### Q3.6

Position: A and B together, with C's restatement: the front end folds and compares `@c.` dims in canonical form and rejects a provable disagreement, an unresolved pair is compatible at the front end, the oracle raises the fault at the call, and the backend checks agreement at selection where `@c.` values are bound and raises the same coded fault.

1. StableHLO's verifier treats a dynamic dim as compatible with anything (`isDynamicDimSize(dimSize1) || isDynamicDimSize(dimSize2) || dimSize1 == dimSize2`), and the shape dialect lowers an unresolved `cstr_eq` to a run-time assertion; the split static-then-runtime is the standard answer.
2. The oracle's `agree` is already the static half with no folding; S6's `fold` gives the canonical form for free, so `(@c.n - 1) / 4 + 1` and its rewrite compare equal or provably unequal.
3. The backend binds `@c.` values at selection: it emits per-rewrite shape checks from `shape_vec` with `lets` (isel_ir2isa_rewrites_rs_generator.py:593-603), so a disagreement is decidable there and must be a coded error, not a rewrite that never matches or a panic in `merge`.
4. Decision 16 forbids the silent outcome X1 describes for the backend, so C alone is not allowed; the fault takes the same code in both generators (Q3.11).
5. S4's benefit line then reads: closes X1 and X5 for literal and foldable dims at the front end, and for the rest at the oracle call and at backend selection with one code.

Evidence: raw.githubusercontent.com/openxla/stablehlo/main/stablehlo/dialect/TypeInference.cpp (verifyCompatibleDims); mlir.llvm.org/docs/Dialects/ShapeDialect/ (cstr_eq, witnesses); act-oracle/taidl_to/generic/shapes.py:58-62; act-oracle/taidl_to/generic/utils.py:771-786; act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:593-603; shared/bugs.md X1, X10; shared/plan/S6.md; owner decision 16.
Confidence: medium.
Would change my mind: a case-study shape whose dims are not affine in `@c.` values, which would make the canonical form a solver problem rather than a fold.

### Q3.7

Position: option A: import at run time at the pinned tag, with `taidl.shapes` and `taidl.diagnostics` importable without the parser, which also means the shape rules take the typed node model, not parse trees.

1. `taidl/taidl/__init__.py` imports `instruction`, and `instruction.py:4` imports the ANTLR runtime, so today `import taidl.shapes` would load the parser through the package init.
2. The oracle's `declared(result_type)` calls `.shape().dim()` on a parse tree, so a moved module that keeps that signature drags the tree types with it; the typed node model of Q3.4 removes the dependency at the source.
3. LLVM's rule for a stable consumer surface is a thin layer with its own compatibility test; a parser-free import of two modules is that layer, and a test that imports them with `antlr4` absent proves it.
4. Vendoring (C) creates the second copy of the rules that SD1 and R39 remove, and it can drift from the tag the generator pins.
5. Option B works but makes every generated oracle carry the parser pin at run time, so a runtime bump of ANTLR breaks shipped oracles that parse nothing.

Evidence: llvm.org/docs/DeveloperPolicy.html (C API stability, compatibility test); taidl/taidl/__init__.py:5-7; taidl/taidl/instruction.py:4, :10; act-oracle/taidl_to/generic/shapes.py:31-35; act-oracle/taidl_to/generic/utils.py:771-786; shared/overhaul.md:14-16; minutes/rulings.md:72 (R39); shared/decisions.md SD8.
Confidence: high.
Would change my mind: a generated oracle that must run where `taidl` cannot be installed at the pinned tag, which would make vendoring the only route.

### Q3.8

Position: placement A and namespace D: line 1 ends with `[S<id>]` where the id is a spec section id, S9 rows being numbered subsections of S9; line 3 lists further sections only; a generator's `coverage.md` clause is the `spec` field of a `U` code, never the suffix of an `R` or fault message.

1. rustc puts the code in brackets at the head of line 1 (`error[E0308]`) and Clang puts the controlling flag at the tail (`[-Wunused-variable]`); head for the code and tail for the rule is a read pattern users already know.
2. R44 rules the suffix, and the suffix on line 1 survives a one-line channel (a CI log, a `str(error)` first line), where line 3 does not.
3. One namespace keeps the registry's single `spec` field honest: an S9 row is a section id, so E is a case of D, and F describes a generator, not the language (shared/decisions.md SD7).
4. Line 3 as the only carrier (B) makes the test key depend on a line that `I` and `X` omit and that a tool prefix does not touch.
5. The word "clause" has no precise use in the owner's papers, so under decision 14 the id is named "section id" in docs and the suffix carries the id alone.

Evidence: doc.rust-lang.org/error_codes/error-index.html (code form); clang.llvm.org/docs/InternalsManual.html (warning flags); diagnostics/scheme.md "Message format", "Registry"; diagnostics/registry-seed.md:1-5; minutes/rulings.md:82 (R44); tests/corpus/format.md (T5 verdict field); owner decisions 14 and 16.
Confidence: medium.
Would change my mind: a reading of the TAIDL paper that uses "clause" for a numbered rule, which would fix the word under decision 14.

### Q3.9

Position: match key D with rows as in B: expectation and corpus rows carry `code` and section id, tests assert `code` and `object`, one text test per format rule and one golden three-line test per class; rendering E, one renderer owned by T11 for `coverage.md`, while D11 keeps rendering S9 in taidl.

1. MLIR and Clang verify tests match message text by substring, and both pay for it with mass test edits on a rewording; rustc separates the stable code from the text, and E16 records the same cost in this repo.
2. A code is never reused or renumbered (scheme), so it is the one key that survives every wording change, while a regex pins wording by construction.
3. The format rules are few (one sentence, no period, 160 characters, fixed phrases), so one text test per rule and one golden per class cover the text contract without a regex per case.
4. Two renderers of `coverage.md` (F) disagree the day one of them changes a column; the registry holds the text and the expectations hold the verdict, so one renderer reads both.
5. S9 (`taidl/docs/conformance.md`) is a different page in a different repo, so D11's S9 renderer is not the duplication; only the per-generator `coverage.md` is rendered twice today.

Evidence: mlir.llvm.org/docs/Diagnostics/ (expected-error substring); clang.llvm.org/docs/InternalsManual.html (Verifying Diagnostics); doc.rust-lang.org/error_codes/error-index.html; diagnostics/bugs.md E16; diagnostics/scheme.md "Classes and codes", "Wording rules"; tests/generators.md T11, T12; diagnostics/plan/D10-tests-and-lint.md; diagnostics/plan/D11-generated-docs.md.
Confidence: medium.
Would change my mind: a count showing most corpus cases need the found value in the assertion, which a regex gives and a code does not; the fix would then be a structured `found` field, still not a regex.

### Q3.10

Position: option A: the class is Fault, the norm and D8's tests state the no-write rule, the code letter `V` stays since codes are never renumbered and the letter names the condition, and front-end and kernel cases are "rejection" and "input error".

1. LLVM separates undefined behavior from a trap; decision 16 removes the first, so the remaining run-time outcome is a trap-like event, and a precise trap writes no architectural state, which is what "fault" means in the owner's hardware vocabulary.
2. The no-write rule is already true of the oracle's one existing check: `check_shapes` raises before any value is written (utils.py:771-786), so the norm states an invariant the code has.
3. Option C blurs the one distinction the class split earns: a front-end rejection has no run time to write, so "writes no region" is vacuous there, and the scheme's class table separates them for that reason.
4. Option B keeps "Violation" as the class name against R55, so docs and code would carry two names for one class.
5. Renaming the letter (D) is unnecessary: `V` does not collide with scope `F`, and the seed already issues `V` codes.

Evidence: llvm.org/docs/LangRef.html (undefined behavior, verifier well-formedness); act-oracle/taidl_to/generic/utils.py:771-786; diagnostics/scheme.md "Classes and codes", "Run-time norm"; minutes/rulings.md:27 (R55); taidl/taidl/operations.py:167-168; diagnostics/plan/D14-run-time-norm.md; owner decisions 14 and 16.
Confidence: medium.
Would change my mind: a passage of the ACT or TAIDL paper that uses "fault" for a description error, which under decision 14 would favor C.

### Q3.11

Position: option A: fault codes take scope `F` in `taidl/diagnostics.json`, like `R` codes; generators reuse them, and `VB001` is not a shared fault but a backend `U` row.

1. rustc error codes are one compiler-wide namespace, not one per backend; a code names the rule, and the raiser is already on line 2 and in the tool prefix.
2. Every seeded `V` rule cites a language section (S5.2, S5.3, S6.2, S7.2), so the rule belongs to the language registry, which the scheme already says holds "shared V codes".
3. T13 compares verdicts across generators on the supported intersection; one code per rule makes that comparison a string equality, while B and C need a mapping table that is a third copy of the rule list.
4. `VB001`, "decidable at generation", describes a backend limit, not a language rule, so it is a `U` row with the backend's assumption stated, as the norm requires.
5. Q3.6 puts the same shape fault in both generators, so the shared scope is needed on day one.

Evidence: doc.rust-lang.org/error_codes/error-index.html; diagnostics/registry-seed.md:87-93; diagnostics/scheme.md "Registry", "Run-time norm"; diagnostics/bugs.md E13; tests/generators.md:13 (T13); owner decision 16.
Confidence: high.
Would change my mind: a fault whose rule differs by generator in a way the spec sanctions, which would make per-raiser scope honest.

### Q3.12

Position: options A and E together: a rejected v3 construct reports its replacement and the section id with one seeded `RF` code per removed construct; no converter; codes for the removed constructs' own rules (RF013 to RF017) are not seeded, and the rest of the seed waits for the v3 API.

1. rustc keeps a retired code listed as no longer emitted and never reuses it; seeding RF013 to RF017 now retires five codes before their first release, which is the one case the never-reuse rule makes costly.
2. The removal set is small and known: audit R58 lists the tests and spec pages that call `set_apart`, and no example or tutorial does, so a converter (B) is more code than the migration.
3. A generic rejection (C) fails decision 16's rule that a rejection ties to a defined rule, and leaves the user to find the replacement in release notes.
4. One code per removed construct is the usual deprecation shape (LLVM's "Potentially Breaking Changes" notes name the replacement), and it is the only code about `set_apart` that outlives v3.
5. Seeding everything after the API is fixed keeps the registry monotone, while the `R` rows for surviving constructs can be seeded today since their rules do not move.

Evidence: doc.rust-lang.org/error_codes/error-index.html; llvm.org/docs/DeveloperPolicy.html (deprecation, release notes); diagnostics/registry-seed.md RF013 to RF017; reviews/rulings/audit.md:34, :51; diagnostics/plan/D04-move-checks-to-front-end.md (Caveats); diagnostics/scheme.md "Classes and codes"; minutes/rulings.md:15, :30 (R5, R58); owner decisions 10 and 11.
Confidence: medium.
Would change my mind: a user base with v2 descriptions outside the three repos, which would justify a converter.
