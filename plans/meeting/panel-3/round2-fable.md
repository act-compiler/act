# Panel 3, round 2, Fable: compiler infrastructure architect

Lens: the boundary between one front end and many back ends, as MLIR, StableHLO and LLVM draw it.
Code paths are relative to /home/user; plan paths are relative to act/plans/.
Standing assumption: both generators stay Python through v3 (shared/decisions.md SD5), and v3 is one coordinated release (owner decision 10).

### Q3.1

- Strongest opposing argument (Opus, option C): the move is a port, not a copy, because the oracle's rules read the oracle's lowered values and must be re-pointed onto taidl's own trees; the move changes no verdict while the R31 checks change verdicts, so the shared rule "structural prep first, behavioral delta second" asks for two items.
- Rebuttal: the port claim is right and strengthens L, not C.
- `declared(result_type)` at act-oracle/taidl_to/generic/shapes.py:31-35 calls `.shape()`, `.dim()`, `.INT()` and `.getText()` on a parse tree, while `agree`, `within` and `count` at :58-80 take tuples, so the entry retargets and the rules copy.
- Two index rows recreate the cross-item dependency the audit flags (reviews/rulings/audit.md:35); two ordered PRs under one row give the same prep-then-delta order with one owner and one dependency column.
- Sonnet and I already agree on B sized L with R31 named in the What lines, and Opus agrees on L and on dropping B23.
- Final position: option B, one S4 item sized L, R31 checks named, the move and the checks as two ordered PRs; unchanged.
- Confidence: medium.
- Still divides: one row with two PRs (Fable, Sonnet) or two rows (Opus); the sizing and ownership are settled.

### Q3.2

- All three take option A with the R25 literal check at act-backend/act_backend/spec_checks.py:394-418 kept untouched until `validate()` ships; nothing divides the panel.

### Q3.3

- All three take option C per item: L15 to Q1, L16 already Q1 (language/plan/frontend.md:16), D10 drops L28 (language/plan/terminology.md:13, after v3), S2 before B13 with the B13 dependency dropped.
- Sonnet's open point on the L16 and L28 sizes is closed by the rows just cited: L16 is S and Q1, L28 is S and after v3.

### Q3.4

- All three take option D with A as the now part: typed model plus typed node model, trees unstable, `CONTRACT_VERSION`, per-member deprecation, and the second non-Python consumer as the trigger for a serialized form; nothing divides the panel.

### Q3.5

- All three take option A.
- Opus adds that the derived members raise before `validate()` runs, so no generator reads half-built facts; I adopt it as the member contract, since a silent default is the kind of fallback engineering.md forbids.
- Final position: option A with members absent until `validate()`; unchanged in substance.

### Q3.6

- Strongest opposing argument (Opus and Sonnet, option C): a symbolic normalizer is an affine decision procedure with no plan item and its own corpus; a checker that rejects what it cannot prove rejects valid descriptions, and one that accepts them adds nothing over `agree`; the backend's failed shape check is a non-selection, not an error, and is panel 4's question.
- Concession on A: my round-1 claim that S6's `fold` gives a canonical form for free is wrong.
- S6 moves `integer()` from act-backend/act_backend/spec_checks.py:118-139, which returns None on any `reference()` at :123, so it folds literal subtrees only and never a `@c.` term.
- A symbolic normalizer is therefore new work with no row, and under decision 11 it is out.
- Sonnet's cheap step stands: two dims with identical expression text are equal, and that is one line in `agree`.
- Rebuttal on B: the scheme's own class table says `V` is raised by "a generator at generation where it can decide" (diagnostics/scheme.md:16), so a backend coded fault at selection is the scheme as written, not a panel-4 novelty.
- The backend outcome today is not a clean non-selection: X1 records that a disagreeing description "makes a rewrite never match or panic in `merge`", and egraph.rs:403-409 panics with "internal error: a rewrite joins the values".
- At selection the `lets` from `shape_vec` bind the `@c.` values (isel_ir2isa_rewrites_rs_generator.py:593-603), so declared versus derived shape is decidable there, and the panic becomes a coded `V` with the same code the oracle raises.
- Final position: C plus B, without A: the front end rejects literal and identical-text disagreement, the oracle raises the fault at the call, the backend raises the same coded fault at selection in place of the `merge` panic, and S4's benefit line names the literal cases only; changed from A plus B plus C.
- Confidence: medium.
- Still divides: whether the backend's coded fault at selection is decided here or by panel 4; the no-normalizer part is now unanimous.

### Q3.7

- All three take option A.
- Opus's PEP 562 lazy package init is the right mechanism for `taidl/taidl/__init__.py:5-7`, and the regression test is `antlr4` absent from `sys.modules` after `import taidl.shapes`.
- One correction to Sonnet and Opus: shapes.py imports nothing, but `declared()` at :31-35 duck-types a parse tree, so the moved module is parser-free only once that entry takes the typed node model of Q3.4.
- Final position: option A with the shape rules taking typed nodes; unchanged.

### Q3.8

- Strongest opposing argument (Opus, placement B): line 1 already carries the code and the registry maps each code to one `spec` field, so a clause suffix on line 1 repeats a fact the code determines, and rustc's layout puts secondary facts on `= note:` lines.
- Rebuttal: the code determines the section only through a registry lookup, and the reader of a CI log or of `str(error)` truncated to its first line has no registry.
- Line 3 is omitted for `X` and `I` and is the line a tool prefix never touches (diagnostics/scheme.md Message format), so it is the weakest carrier of the one id decision 16 ties a rejection to.
- Clang's `[-Wunused-variable]` tail shows the one-line form holds both a head code and a tail rule without a second line.
- Namespace D is unanimous: an S9 row numbered by L is a section id, and a generator's `coverage.md` id never enters a front-end message.
- Final position: placement A and namespace D; unchanged.
- Confidence: medium.
- Still divides: placement A (Fable, Sonnet) or B (Opus); Opus names it a value judgment once the namespace is fixed, and I agree it is one.

### Q3.9

- All three take match key D and renderer E; nothing divides the panel.

### Q3.10

- Strongest opposing argument (Sonnet, option D): a third-party author needs one name in prose, exception class and code letter; a class named Fault with letter `V` is a glossary entry, and letter `F` would collide with scope `F` as `FF001`, so the class letter should change to a third letter.
- Rebuttal: the collision argument rules out `F`, which all three accept, so the choice is `V` against an arbitrary third letter that matches neither Fault nor Violation.
- `V` keeps the scheme's message form `<subject> violates <rule>` (diagnostics/scheme.md Wording rules) honest: the condition is a violation of a rule and the class that reports it is the Fault.
- Opus and I agree the front end's `Fault = tuple[CommonToken, str]` at taidl/taidl/operations.py:167-168 is renamed `Rejection`, which removes the one real ambiguity.
- The never-renumbered rule (scheme.md:10) does not decide it, since the seed is a plan and nothing is released; I withdraw that step from round 1.
- Final position: option A with letter `V` kept and the front-end alias renamed `Rejection`; unchanged.
- Confidence: low on the letter, medium on the class name and the no-write rule.
- Still divides: the letter only; it is a value judgment the moderator may pass to the owner with the two options.

### Q3.11

- All three take option A, fault codes under scope `F` in `taidl/diagnostics.json`, with `VB001` a backend `U` row; nothing divides the panel.

### Q3.12

- Strongest opposing argument (Sonnet, option C): a removed Python method raises `AttributeError` with no code, so a coded replacement message needs a stub method kept in the API, which engineering.md forbids as a "removed" placeholder.
- Rebuttal: `set_inplace` and `set_apart` are methods on `Instruction` at taidl/taidl/instruction.py:163 and :174, and a class-level `__getattr__` with one table of removed names raises one coded rejection for every removed member without a stub per name, which is Opus's mechanism.
- One live lookup is not a `// removed` placeholder; it is the one place the migration lives, and it covers R2, R5, R58 and R60 alike.
- A generic rejection with a release note gives no path from the error to the fix, which decision 16's tie to a defined rule asks for.
- Concession to Opus: one code with a table beats one code per removed construct, since the rule is one ("a v2 construct is removed") and the message names the construct and its replacement.
- Final position: option A as one code with a removed-names table, no converter, and option E, codes seeded after the v3 API is fixed; changed from one code per construct to one code.
- Confidence: medium-high.
- Still divides: Sonnet's C against A; the seeding answer E is unanimous.

## Escalation to the owner

- None.
