# Panel 3 decision: front end and generator contract

| Question | Decision (one line) | Confidence | Owner? |
| -------- | ------------------- | ---------- | ------ |
| Q3.1 | S4 owns all shape and per-word type rules, sized L, R31 checks named, move PR then checks PR; B23 dropped | medium | no |
| Q3.2 | B21 to B23 become one S item: call `validate()` first and delete `spec_checks.py:394-418` | high | no |
| Q3.3 | Per item: L15 to Q1, L16 already Q1, D10 drops L28, S2 before B13 with that dependency dropped | medium-high | no |
| Q3.4 | Typed model plus typed node model, trees unstable, `CONTRACT_VERSION`; serialized form at the second non-Python consumer | high | no |
| Q3.5 | `validate()` attaches derived facts and the call site as typed members that raise before `validate()`; `fold` and `literal_fits` listed | medium-high | no |
| Q3.6 | Front end decides literal and equal-folded dims; oracle faults at the call; backend replaces its `merge` panic with the shared coded fault | medium | confirm |
| Q3.7 | Vendor the shape module into each generated oracle, stamped with its taidl version (decision 25); the module imports nothing | high | no |
| Q3.8 | Namespace is a spec section id; placement on line 1 or line 3 is a value judgment | medium | yes |
| Q3.9 | Tests assert `code` and `object`; rows carry `code` and section id; one `coverage.md` renderer owned by T11 | high | no |
| Q3.10 | Class is Fault with a no-write rule; letter stays `V`; front-end alias `Fault` becomes `Rejection` | medium | no |
| Q3.11 | Fault codes take scope `F` (`VF...`); `VB001` merges into the shared 64-bit fault | high | no |
| Q3.12 | One coded rejection from a removed-names table names the replacement; no converter; seed after the v3 API is fixed | medium | no |

Cited code was re-read in /home/user/taidl, /home/user/act-backend, and /home/user/act-oracle; uncited claims carry no weight.
Plan paths are relative to /home/user/act/plans; code paths are relative to /home/user.

## Q3.1 Shape and type rules: owner and size

- Decision: option B.
  - S4 is the one item for the module move, the per-word type table, and the R31 checks (region values, constant ranges, compare-type fit); size L.
  - Two ordered PRs in one row: PR 1 moves the rules with no verdict change, PR 2 adds the new checks.
  - L12, O6, O7, and B23 fold into S4; L3 stays the spec text and depends on L2 only; D4 attaches codes RF040 to RF050 and depends on S4.
- Reasoning:
  1. The index sizes S4 at L (shared/plan/README.md:10); R62's M has no evidence (audit R62).
  2. The move is a port: `declared()` calls `.shape()`, `.dim()`, `.INT()` on a parse tree (act-oracle/taidl_to/generic/shapes.py:31-35), while `agree` takes tuples (:58-62).
  3. The mass is 543 lines (shapes.py) plus the word table `oracle_visitor.py:19-97`, about 620 lines.
  4. R31's checks exist in no shared copy and are not in S4.md.
  5. Folding the owners removes the L3-O6 and O7-L12-B23 cycles (language/plan/spec.md:12, oracle/plan/checks.md:10-11).
  6. A second index row recreates the cross-item arrow the audit flags; two PRs give the same order.
- Dissent: Opus round 1 wanted two rows; conceded.
- Confidence: medium; no line count of the R31 checks exists, and checks.md:11 sizes O7 as M against the audit's L.

## Q3.2 Backend items B21 to B23

- Decision: option A.
  - One S item that depends on S3: call `Accelerator.validate()` first, then delete `spec_checks.py:394-418`.
  - It leaves the S3, S4, and D4 dependency columns; the R25 literal check stays until S3 ships.
- Reasoning:
  1. The existing check handles only all-literal regions (`spec_checks.py:413-414`, the `None in (tile, region)` skip).
  2. An interim check written in Q1 and deleted by D5 has no output but its own deletion.
  3. Decision 10 ships one release, so no v3 backend ships without S3; the X3 and X4 gap never reaches a release.
  4. Dropping the items (option C) loses the scheduled call and its test (shared/bugs.md X3).
- Dissent: none; confidence high.

## Q3.3 Q1 items that depend on Q2 items

- Decision: option C, per item.
  - L15 moves to Q1 (S, depends on L25 in Q1; language/plan/frontend.md:15, terminology.md:9).
  - L16 is already S and Q1 (frontend.md:16); S5 and O8 need no move.
  - D10 drops L28 (after v3; terminology.md:13) and owns the banned-word list file; L28 consumes it.
  - S2 precedes B13; B13 leaves S2's dependency column.
- Reasoning:
  1. S2 documents the contract once under the renamed members (decisions 19 and 20), so L15 precedes it.
  2. S2 is the root of S3, S5, S6, and S8 (shared/plan/README.md:8-14); moving it to Q2 moves the track.
  3. D10 needs the word list, not the lint (diagnostics/plan/README.md:17).
- Dissent: none after the rows were read; confidence medium-high.

## Q3.4 Form of the contract

- Decision: option A now with option D's trigger.
  - A documented typed Python model plus a small typed node model for lines, operands, attributes, and integer expressions.
  - ANTLR trees stay reachable and are marked unstable.
  - `CONTRACT_VERSION`, a per-member deprecation rule (one minor release), member names per decisions 19 and 20.
  - A serialized form is added at the second non-Python consumer.
- Reasoning:
  1. Today's surface is ANTLR contexts and a dict keyed by `TerminalNode` (taidl/taidl/instruction.py:34-57, :77-84), tied to `antlr4-python3-runtime==4.9.2` (taidl/pyproject.toml:15).
  2. Option B freezes the grammar and the undocumented members (`comp_attr`, `inplace`, `placed`, `d0`; shared/interface.md).
  3. Only four classes are exported (taidl/taidl/__init__.py:9), so a typed model is a small addition.
  4. No non-Python consumer exists; a serialized form is a second contract with no reader.
- Dissent: none; confidence high.

## Q3.5 Derived facts published by `validate()`

- Decision: option A.
  - `validate()` attaches per-line result type and dims (folded expressions over `@c.` where `fold` applies), folded sizes, the value-to-region map, and the call site, as typed members.
  - Each member raises if read before `validate()` has run.
  - `fold` and `literal_fits` are listed contract functions; the shape-rule functions are not.
- Reasoning:
  1. S4 computes these facts anyway; discarding them keeps the re-derivation of X6 to X8 and the backend regex (isel_ir2isa_rewrites_rs_generator.py:91, :134-137).
  2. A separate result object (option B) must be paired with each instruction and can drift.
  3. D2 stores the call site and an outside generator needs it for line 2 of a `U` message; S2.md omits it.
  4. A silent default before `validate()` is a masking fallback (engineering.md).
  5. A generator that folds its own attribute needs `fold`; one exported function is not a second copy.
- Dissent: Opus treats `fold` and `literal_fits` as a by-product; confidence medium-high.

## Q3.6 Shape agreement with `@c.` dims

- Decision: option C plus a narrow B, no symbolic normalizer.
  - Front end: reject literal mismatches; accept dims equal after `fold` or with identical text; everything else is compatible.
  - Oracle keeps the call-time fault.
  - Backend: the `merge` panic becomes the shared coded shape fault (Q3.11) at selection, when a declared and a computed dim are bound and differ.
  - S4's benefit line says it closes X1 and X5 for literal and foldable dims only.
- Reasoning:
  1. `agree` accepts any non-integer dim (shapes.py:58-62), and `integer()` returns None on any `reference()` (act-backend/act_backend/spec_checks.py:123), so `fold` never folds a `@c.` term.
  2. A symbolic normalizer has no plan item; decision 11 excludes it.
  3. The oracle's `check_shapes` is exact per iteration (act-oracle/taidl_to/generic/utils.py:771-786).
  4. X1 records that a disagreeing description makes a rewrite never match or panic (egraph.rs:403-409, "internal error: a rewrite joins the values"); a panic is neither a defined error nor a defined behavior (decision 16).
  5. At selection the `lets` from `shape_vec` bind `@c.` values (isel_ir2isa_rewrites_rs_generator.py:593-603) and the root shape comes from the declared type (:702-708), so the comparison is decidable there.
  6. Sonnet's objection holds for a kernel-versus-pattern mismatch, which stays a non-match; the checked pair is two shapes from the description.
- Weakness: no debater showed that the computed dim is expressible for every operation word; panel 4 sizes the Rust wiring.
- Dissent: Sonnet final keeps C only and reads the unsupported-table assumption (scheme.md:30) as the defined behavior.
- Confidence: medium; the owner confirms the reading of decision 16 (see Escalation).

## Q3.7 Shape rules in the generated oracle

- Decision: option C, by owner decision 25.
  - The generator copies the shape module into each generated oracle, stamped with `taidl.__version__` (taidl/taidl/__init__.py:3, '2.15' today).
  - The module imports nothing from `taidl` and nothing from ANTLR.
  - Its entry takes typed nodes or `(type, dims)` tuples, never parse trees; `declared()` (shapes.py:31-35) is retargeted in S4 PR 1.
  - One test imports the module alone and asserts `antlr4` is absent from `sys.modules`; a drift test compares the copy with the source at the stamped version.
- Conflict: all three debaters took option A (run-time import, lazy package init); decision 25 settles the mechanism, so R39 and review SD12 are superseded.
- What survives: the parser-free property, now required of the copied file (instruction.py:4 and `__init__.py:5-7` show a package import loads ANTLR).
- Confidence: high.

## Q3.8 Clause id: placement and namespace

- Decision, namespace: option D.
  - The id is a spec section id; an S9 row is a numbered subsection such as `S9.4`, so option E is a case of D; L must number the rows.
  - A generator's `coverage.md` id (option F) never appears in an `R` or fault message; `TaidlError.spec` carries the section id as a field.
  - The docs word ("section id" or "clause id") is a glossary matter; the claim that the owner's papers lack a precise use of "clause" is uncited.
- Decision, placement: owner; the panel is at A after Opus conceded.
  - Option A, suffix `[S<id>]` ends line 1: line 1 survives CI logs, `grep`, and the first line of `str(error)`, where no registry is at hand; the 160-character rule leaves room (scheme.md:50).
  - Option B, line 3 only: the code fixes the section through the registry's `spec` field, so a suffix repeats a fact.
  - Option C, both: it serves the one-line channel and the full list, at the cost of a duplicate.
  - Recommendation: A, as R44 and review SD7 have it.
- Evidence: diagnostics/scheme.md (Message format, Registry); registry-seed.md Spec column; tests/generators.md T11 to T13.
- Confidence: high on the namespace, medium on placement.

## Q3.9 Match key and `coverage.md` renderer

- Decision: match key D, renderer E.
  - Tests assert `code` and `object`; expectation and corpus rows carry `code` and section id, with no regex.
  - One text test per format rule and one golden three-line test per class.
  - T11 owns the one `coverage.md` renderer (registry for text, expectations for verdicts); D11 keeps the S9 page in taidl, a different page.
- Reasoning:
  1. A code is never reused or renumbered (scheme.md), so it survives rewording; a regex pins the wording (diagnostics/bugs.md E16).
  2. The format rules are few, so a handful of text tests cover the wording contract.
  3. Two renderers of one page disagree once a column changes (tests/corpus/format.md:9, :22).
- Dissent: none; confidence high.

## Q3.10 Run-time error class

- Decision: option A with the letter `V` kept.
  - The class is Fault; the norm and D8's tests state that a fault writes no region and no state; front-end and kernel cases are rejection or error.
  - The alias `Fault = tuple[CommonToken, str]` (taidl/taidl/operations.py:167-168) is renamed `Rejection`; `ViolationError` (scheme.md:90) follows the class name.
  - The scheme notes once that the letter `V` names the violated rule and the class is Fault.
- Reasoning:
  1. A defined error needs a stated residue, or generators differ after it and T13 has nothing to compare (decision 16).
  2. The oracle's shape fault precedes any write (utils.py:771-786); other faults are not shown to be pre-write, so D8 tests the rule.
  3. `V` cannot collide with scope `F` (a shared fault reads `VF001`); the seed already issues `V` codes.
  4. Option C would make `Fault` mean a front-end rejection too, the ambiguity the alias rename removes.
- Dissent: Sonnet round 1 wanted a new letter; conceded.
- Confidence: medium; "fault" is unchecked against the owner's papers (decision 14).

## Q3.11 Scope letter for fault codes

- Decision: option A.
  - Fault codes take scope `F` in `taidl/diagnostics.json`, like `R` codes (`VF001`); generators reuse them; the tool prefix names the raiser.
  - `VB001` merges into the shared 64-bit fault.
- Reasoning:
  1. A fault is a language rule; every seeded `V` rule cites a spec section (registry-seed.md:87-93).
  2. T13 compares verdicts across generators; one code is a string equality, where options B and C need a mapping table, and codes are never renumbered.
  3. `VB001` is the range rule `VO005` raised at generation: the backend's three checks reject a computed integer that "64 bits cannot hold" (act-backend/act_backend/spec_checks.py:218, :252, :294), and E13 names both for one norm.
  4. What the backend cannot decide goes in its unsupported table, not in a coded fault.
- Dissent: Fable and Opus read `VB001` as a backend `U` row; the code at the lines above is a range check.
- Confidence: high.

## Q3.12 Rejected v3 constructs and seeding

- Decision: options A and E, one code with a table.
  - A removed construct raises one coded `RF` rejection naming the construct, its replacement, and the section id.
  - One removed-names table feeds one lookup (class-level `__getattr__` for members, the `OPERATIONS` lookup for words); no stub per member, no converter.
  - RF013 to RF017 are not seeded; the seed of moving rules waits for the v3 API.
- Reasoning:
  1. A removed method raises a bare `AttributeError` (`set_inplace` and `set_apart` are methods, taidl/taidl/instruction.py:163, :174); a coded message gives the writer the fix.
  2. A table that drives a user-facing error is a tested feature, not a "removed" placeholder (engineering.md).
  3. One code for "a v2 construct is removed" avoids a never-reused code per construct.
  4. A converter rewrites Python programs; decision 11 leaves no capacity.
- Conflict: draft R5 removes `set_inplace`, but decision 22 keeps it unless the panels show it redundant; whether it enters the table waits for panels 1 and 2.
- Evidence: registry-seed.md:23-27; review SD13; taidl/taidl/operations.py:145.
- Dissent: Sonnet (generic rejection plus S8 note) and Fable (code per construct) conceded in round 2.
- Confidence: medium.

## Moderator flags

- Q3.7: decision 25 covers the shape module only; the diagnostics registry reaches the oracle by an unstated route; recommend the same copy-and-stamp.
- The authors' items SD11, SD14, SD15, SD17 to SD20, and R41 stand as the agenda lists them.

## Escalation to the owner

- Q3.6 (Sonnet round 2): decide whether "a defined behavior under one simple norm" (decision 16) includes the backend's unsupported-table assumption for `@c.`-dependent shapes.
  - Reading 1: yes; the backend selects nothing for a disagreeing description and states the assumption.
  - Reading 2: no; the panic at egraph.rs:403-409 becomes the shared coded fault at selection (the moderator's decision).
- Fable and Opus raise no escalation; the value judgment is Q3.8 placement (options above; recommendation A).
