# Panel 3 round 2: Sonnet, third-party generator author

Paths are relative to /home/user/ for code and to /home/user/act/plans/ for plans.
The lens is an author outside the three repos who reads only the published contract and registry.

### Q3.1

- All three agree: S4 is the single owner, sized L, the R31 checks are named in its What lines, and B23 is dropped.
- Strongest opposing argument (Opus): two sequential items, because the move changes no verdict and the checks do, which matches the shared rule of structural prep first and the behavioral delta second.
- Rebuttal: the sequencing is kept, but as two PRs inside one S4 row (Fable's form), because a second index row recreates the cross-item arrow the audit flags (reviews/rulings/audit.md:35) and adds a row a generator author must read.
- The commit rule is met by PR order; it does not need a second plan item.
- Evidence checked: act-oracle/taidl_to/generic/shapes.py:58-62 shows `agree` is a rule, so the move is not a pure copy and the PR split is where the port work is visible.
- Final position: B, one S4 item sized L, move PR then checks PR.
- Confidence: medium.
- Still dividing: one row with two PRs (Fable, Sonnet) against two rows under S4 (Opus); both give size L in total.

### Q3.2

- All three agree on A: B21 to B23 become "call `validate()` first and delete `spec_checks.py:394-418`", and the R25 literal check stays untouched until `validate()` ships.
- Fable adds that dropping the items entirely (C) loses the scheduled call and the no-output test; I accept that point.
- Final position: A (unchanged).
- Confidence: high.
- Still dividing: nothing.

### Q3.3

- Strongest opposing argument (Opus and Fable): the review premise is stale for L16, which is already Q1, so only L15 and L28 need handling.
- Verification: language/plan/frontend.md:15-16 lists L15 as S, Q2, depending on L25, and L16 as S, Q1, depending on L3; language/plan/terminology.md lists L25 as S, Q1, and L28 as "after v3".
- Concession: my round-1 text had not read the L16 and L28 rows, and the rows decide it.
- L15 moves to Q1 (S, depends on L25 in Q1), L16 needs no move, and D10 drops its L28 dependency.
- D10 owns the word list and L28 consumes it after v3 (Fable); S2 precedes B13 and B13 leaves S2's dependency column.
- Final position: C, with those four per-item rulings.
- Confidence: medium-high (was low).
- Still dividing: nothing material; Fable and Opus agree with the refined position.

### Q3.4

- All three agree on D with A as the now part: typed model, typed node model, trees reachable but unstable, `CONTRACT_VERSION`, a per-member deprecation rule, and the second non-Python consumer as the trigger for a serialized form.
- Fable adds that S2 as written would document undocumented members (`comp_attr`, `inplace`, `placed`, the magic `d0`) and so freeze the wrong surface; I accept it, and it is the strongest argument against B.
- Opus adds that members read as absent (raise) before `validate()`; I accept it for Q3.5.
- Final position: D with A (unchanged).
- Confidence: high (was medium).
- Still dividing: nothing.

### Q3.5

- All three agree on A: result type and shape per line, folded sizes, the value-to-region map and the call site as typed members.
- Strongest opposing point (Fable): keep `fold` and `literal_fits` exported, because three folders and two literal checks diverged when each generator called helpers its own way (bugs X6, X8).
- Assessment: the divergence came from copies, not from one exported function; a single exported `fold` (S6, taidl/integers.py) is not a second copy.
- Concession: a generator that folds an attribute expression of its own needs `fold`, so S2 lists `fold` and `literal_fits` as contract functions.
- The shape-rule functions stay internal, so the contract is facts plus two helpers, not a rule library.
- Opus's rule is adopted: members raise before `validate()` has run.
- Final position: A, plus `fold` and `literal_fits` listed in S2; members raise before `validate()`.
- Confidence: medium-high.
- Still dividing: whether the helper functions are contract (Fable, now Sonnet) or a by-product (Opus).

### Q3.6

- Strongest opposing argument (Fable): the front end folds `@c.` dims to a canonical form and rejects a provable disagreement, and the backend checks agreement at selection and raises the same coded fault, because decision 16 forbids a silent outcome.
- Check 1: S6 is `fold(tree)` for integer literals and says the oracle's loop-variable folder "stays its own" (shared/plan/S6.md Caveats), so S6 gives no canonical form for symbolic `@c.` dims.
- A symbolic normalizer is therefore new work that no plan item sizes, as both Opus and I said in round 1.
- Check 2: the backend does not hold a `@c.` value that the description fixes.
- In taidl/examples/mini_trn.py lines 19-25, `@c.n` is bound from the kernel's HLO operand shapes (`u8[@c.n * 128]`), and act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:593-603 renders a `shape_check` per rewrite from those shapes.
- That check is the match test of a rewrite against the kernel; a failure means "this rewrite does not apply", and a coded error there would reject every non-matching kernel.
- Check 3: the scheme already gives the backend a defined behavior, "rejects at generation what it can decide and states in its unsupported table what it assumes for the rest" (diagnostics/scheme.md, Run-time norm), so decision 16 is met without a backend error.
- Rebuttal: B (backend error at selection) conflates a description's internal shape consistency with a kernel's match; I reject it.
- Concession to Fable: S4's benefit line must say what happens to the rest, so S4 states "decided at the oracle call; the backend's unsupported table states the assumption".
- Partial step kept: two dims with identical expression text are equal (corpus dims are mostly one symbol such as `@c.n`), and nothing is rewritten.
- Final position: C, with the literal and identical-text cases decided in the front end.
- Confidence: medium.
- Still dividing: Fable holds A and B together; the cut is whether non-selection counts as a defined behavior under decision 16.

### Q3.7

- All three agree on A: import at run time at the pinned tag, with `taidl.shapes` and `taidl.diagnostics` importable without ANTLR.
- Verified: taidl/taidl/__init__.py:5-7 imports `accelerator`, `data_model` and `instruction`, and taidl/taidl/instruction.py:4 imports the ANTLR runtime.
- Fable adds that the shape rules must take the typed node model rather than parse trees, since the oracle's `declared(result_type)` calls tree methods; I accept it, because it ties Q3.7 to Q3.4.
- Opus adds a regression test that imports `taidl.shapes` and asserts `antlr4` is not in `sys.modules`; I accept it.
- Final position: A, with node-model inputs and the no-ANTLR import test.
- Confidence: high (was medium).
- Still dividing: nothing.

### Q3.8

- Strongest opposing argument (Opus): placement B, line 3 only, because line 1 already carries the code and the registry maps each code to one `spec` field, so a suffix repeats what the code determines.
- Assessment: the repeat holds for a machine reader, who gets `code` and `spec` as separate fields in the JSON form (diagnostics/scheme.md, Machine form).
- It does not hold for a person reading a one-line channel, such as a CI summary or the first line of `str(error)`, where the code is a key into a registry the reader may not have open.
- Line 1 has room: the 160-character limit (scheme.md Wording rules) leaves seven characters for ` [S9.4]` in all but the longest messages.
- Concession: the choice is a presentation value judgment once the namespace is fixed, and the machine contract is unaffected.
- Namespace: D (spec section id) with S9 rows as numbered subsections; Fable's point that the owner's papers have no precise use of "clause" means docs say "section id".
- Final position: placement A, namespace D (E is a case of D), and the docs call it "section id".
- Confidence: low-medium (was medium).
- Still dividing: placement A (Fable, Sonnet) against B (Opus); all three agree on the namespace.

### Q3.9

- All three agree: match key D (code assertions, one text test per format rule, one golden three-line test per class) and renderer E (one renderer owned by T11).
- Fable's refinement is adopted: expectation and corpus rows carry `code` and section id, and D11 keeps rendering the S9 page in taidl because it is a different page in a different repo.
- Fable names a structured `found` field as the fix if tests need values; I accept it over a regex.
- Final position: D with E (unchanged).
- Confidence: high (was medium).
- Still dividing: nothing.

### Q3.10

- Strongest opposing argument (Fable and Opus): keep the code letter `V`.
- Verification: diagnostics/scheme.md Classes defines the letter `V` for the class "Violation" and `F` as a scope letter only, so a rename of the class to Fault creates a collision only if the letter changes to `F`.
- My round-1 option D rested on the letter becoming `F` (`FF001`), which nobody proposes.
- Concession: the codes are never renumbered (scheme.md Classes and codes) and the seed already issues `V` codes, so renaming the letter costs more than it saves.
- Kept from round 1: the class is Fault, the norm and D8's tests state "writes no region and no state", and the front end's `Fault = tuple[CommonToken, str]` (taidl/taidl/operations.py:167-168) is renamed to `Rejection`.
- Verified: scheme.md lists the exception `ViolationError`, so the exception class follows the class name; a third-party author catches one name.
- Final position: A, with `V` kept as the letter.
- Confidence: medium.
- Still dividing: whether the letter `V` not matching the word "Fault" needs a glossary line; Fable says the letter names the condition, and I accept a one-line note in the scheme.

### Q3.11

- All three agree on A: fault codes take scope `F` in `taidl/diagnostics.json`, and generators reuse them.
- Strongest opposing point (Fable): `VB001` is a backend `U` row, not a shared fault, because "decidable at generation" describes a backend limit.
- Evidence: diagnostics/bugs.md E13 states "One norm: a defined error (`VO005`, `VB001`)", and registry-seed.md:94 gives `VB001` the section S6.2, the same rule as the oracle's 64-bit code.
- Rebuttal: the rule is language-level; the backend raises it earlier, under the same shared code, so the author of a third generator sees one code for one rule.
- The unsupported row is for what the backend cannot decide; a rule it can decide is a fault.
- Final position: A (unchanged), with `VB001` merged into the shared 64-bit fault.
- Confidence: high.
- Still dividing: only the `VB001` reading (Fable).

### Q3.12

- Strongest opposing argument (Opus and Fable): a removed construct reports its replacement and section id; no converter; seeding waits for the v3 API.
- My round-1 objection was that a coded replacement needs a stub method per removed member, which engineering.md forbids as a placeholder.
- Verification: taidl/taidl/instruction.py:163 and :174 define `set_inplace` and `set_apart` as methods, so a removed method gives a bare `AttributeError`.
- Opus's form answers the objection: one table of removed names consulted by one `__getattr__` hook raises one coded rejection naming the replacement, so there is no stub per member and the table is deleted at the next major release.
- Fable's form, one RF code per removed construct, contradicts Fable's own argument that retired codes are costly, since each removed construct would burn a code at its first release.
- The cost is small: R58 and R60 say no example, tutorial or case study uses these constructs, and the third-party author never sees them.
- Concession: a bare `AttributeError` gives the description writer no path to the fix, so the one-code table is worth its size.
- Final position: A and E, with one code and one table (Opus's form); RF013 to RF017 are not seeded; no converter.
- Confidence: low-medium (changed from C and E).
- Still dividing: one code with a table (Opus, Sonnet) against one code per construct (Fable).

## Summary of agreement

- Unanimous or near-unanimous: Q3.2, Q3.4, Q3.7, Q3.9, Q3.11.
- Split on a small point: Q3.1 (rows), Q3.3 (resolved), Q3.5 (helpers as contract), Q3.8 (placement), Q3.10 (resolved), Q3.12 (code per construct).
- Real division: Q3.6 (backend coded error at selection).

## Escalation to the owner

- None.
- Q3.6 depends on how decision 16 reads for a backend non-selection; the moderator may ask the owner whether "a defined behavior under one simple norm" includes the unsupported-table assumption.
