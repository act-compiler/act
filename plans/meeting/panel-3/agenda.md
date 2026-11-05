# Panel 3 agenda: front end and generator contract

Scope: shape and type rules, the object model generators read, the diagnostics scheme, clause ids.
Questions are ordered blocker first, then majors.
Options are neutral; a draft answer from `minutes/rulings.md` is one option among others and carries no weight.

## Reading guide

- Plan paths are relative to `/home/user/act/plans/`; code paths are relative to `/home/user/`.
- "Review SDn" is a finding in `reviews/shared-diagnostics/review.md`; "decisions SDn" is a row in `shared/decisions.md`.
- "Audit Rn" and "audit RAn" are rows in `reviews/rulings/audit.md`.
- Owner decisions are numbered as in `minutes/owner-decisions.md`.

## Questions

### Q3.1: Who owns the shape and per-word type rules, and how large is the item?

- Context:
  - Shape and type rules have seven owners (S4, D4, L3, L12, O6, O7, B23) with dependency cycles between them.
  - R62 merges L3, L12, O6, O7 into one item sized M, while S4 is still sized L and lists those items as dependencies.
  - The audit finds no evidence for the downsizing and notes that R31 widens O7 with region, constant-range and compare-type checks.
- Options:
  - A: one item owned by S4, sized M; D4 only attaches RF049 and RF050 to it; B23 is dropped.
  - B: one item owned by S4, sized L or larger, with the R31 checks folded in explicitly.
  - C: two sequential items, the module move (oracle `shapes.py` and `oracle_visitor.py:19-97` into `taidl/shapes.py`) and the front-end checks (region values, constant ranges, compare-type fit).
  - D: keep L3, L12, O6, O7, B23 as separate items and break the cycles by a stated order.
- Evidence:
  - Review SD1 and SD2; audit R62; draft R62 and R31.
  - `shared/plan/README.md` (S4 row), `shared/plan/S4.md`, `diagnostics/plan/D04-move-checks-to-front-end.md`.
  - `shared/bugs.md` X1 and X5; `shared/overhaul.md` (new `taidl/taidl/shapes.py` row).
  - Audit cites `language/plan/spec.md:12`, `language/plan/frontend.md:12`, `oracle/plan/checks.md:10-11` for the sizes L3 L, L12 L, O6 S, O7 L.
- Binding constraints:
  - Decision 10 (breaking changes batched into one major release).
  - Decision 11 (3-4 engineers, two quarters).

### Q3.2: What do the backend items B21 to B23 become?

- Context:
  - B21 to B23 build region-count, region-type and shape checks in the backend that S3 and S4 move into the front end.
  - S3 and D4 list B21 and B22 as dependencies, while `S3.md` says S3 comes "before B21, B22"; D5 later deletes the backend copy.
  - R25 says the backend keeps its literal region-count check until L5 ships.
- Options:
  - A: redefine B21 to B23 as "call `Accelerator.validate()` first and delete `spec_checks.py:394-418`"; remove them from the S3, S4 and D4 dependency columns.
  - B: the backend writes interim checks, and D5 deletes them when the front end ships (plans as written).
  - C: the backend writes no new checks; it keeps only the R25 literal check until `validate()` ships, and B21 to B23 are dropped.
- Evidence:
  - Review SD2; draft R25.
  - `shared/plan/S3.md`, `shared/plan/S4.md`, `diagnostics/plan/D04-move-checks-to-front-end.md`, D5.
  - `act-backend/act_backend/spec_checks.py:394-418`; `shared/bugs.md` X3 and X4.
- Binding constraints:
  - Decision 11 (capacity).

### Q3.3: How are the Q1 items that depend on Q2 items resolved?

- Context:
  - S2 (Q1) depends on L15 (Q2); S5 (Q1) and O8 (Q1) depend on L16 (Q2); D10 (Q1) depends on L28 (Q2).
  - S2's table depends on B13 while `S2.md` says S2 comes "before B13".
  - S2 states that it lands with the L15 renames so the contract is documented once under the new names.
- Options:
  - A: move L15, L16 and L28 to Q1.
  - B: move S2, S5, O8 and D10 to Q2.
  - C: a mix decided per item, with S2 before B13 and the B13 dependency dropped in every case.
- Evidence:
  - Review SD3; `shared/plan/README.md`; `shared/plan/S2.md` (Caveats, Order and dependencies).
  - `diagnostics/plan/README.md` (D10 row and its critical path note).
- Binding constraints:
  - Decision 10 (one coordinated major release).
  - Decision 11 (capacity, two quarters).

### Q3.4: What form does the front end to generator contract take?

- Context:
  - Today the contract is the Python attribute set of `Accelerator`, `DataModel` and `Instruction` plus ANTLR parse trees, including `IDLV2Parser.ModuleContext` and a dict keyed by `TerminalNode`.
  - `taidl/taidl/__init__.py:9` exports four classes only; the parser pin `taidl/pyproject.toml:15` ties every consumer to ANTLR 4.9.2.
  - R42 rules a documented, typed Python object model and no serialized format; the decoupler position wanted a serialized form.
- Options:
  - A: documented typed Python model plus a small typed node model for lines, operands, attributes and integer expressions; trees stay reachable but are marked unstable; a `CONTRACT_VERSION` and a deprecation rule per member.
  - B: document today's attribute set and the ANTLR trees as the contract (S2 as written), pinned to the parser version.
  - C: a serialized intermediate form now, independent of ANTLR and of Python dict order.
  - D: option A or B now, with a stated trigger (the second non-Python consumer) for adding a serialized form.
- Evidence:
  - Review SD4; draft R42; decisions SD5 and the three positions in `shared/decisions.md`.
  - `shared/interface.md` (member tables, "Documentation of the contract", "Where generators re-parse or re-derive").
  - `shared/plan/S2.md`; `shared/bugs.md` X7.
  - `taidl/docs/expressions.md:42-47`; `taidl/taidl/instruction.py:34-57`.
- Binding constraints:
  - Decision 10 (breaking changes allowed in v3).
  - Decisions 19 and 20 (member names use buffers, state, region).

### Q3.5: Which derived facts does `validate()` publish to generators?

- Context:
  - After S4 the front end computes each line's result type and shape, folded sizes and the value-to-region map, but S2 does not publish them.
  - D2 stores a call site on each built object, and S2 does not list it; line 2 of a `U` message needs it from an outside generator.
  - Generators today re-derive these facts, which is the duplication the track removes.
- Options:
  - A: `validate()` attaches the derived facts to each instruction as documented, typed members, and the call site is a listed member.
  - B: `validate()` returns a separate analysis result object; instructions stay as parsed.
  - C: the front end exports helper functions (shape rules, `fold`, `literal_fits`) that generators call on demand.
  - D: nothing derived is published; outside generators derive their own.
- Evidence:
  - Review SD5 and SD16; decisions SD1 to SD3.
  - `shared/plan/S2.md`, `S3.md`, `S4.md`, `S6.md`; `diagnostics/plan/D02-call-site-and-exceptions.md`.
  - `shared/interface.md` ("Where generators re-parse or re-derive").
- Binding constraints:
  - Decision 10 (breaking changes allowed).

### Q3.6: How is shape agreement decided when dims depend on `@c.` values?

- Context:
  - S4 claims to close X1, but the moved checker treats any non-integer dim as agreeing (`act-oracle/taidl_to/generic/shapes.py:58-62`).
  - A shape that depends on `@c.` is decided only at an oracle call; the backend has no call.
  - S3 states that the front end checks literals and leaves the rest to the oracle's call-time check.
- Options:
  - A: normalize affine `@c.` dims symbolically in the front end.
  - B: the backend checks agreement at selection and raises a coded error.
  - C: keep the call-time check in the oracle only, and restate S4's benefit to the literal-dim cases it closes.
- Evidence:
  - Review SD6; `shared/bugs.md` X1 and X10.
  - `shared/plan/S3.md` (Caveats), `shared/plan/S4.md` (Benefits, Caveats).
  - `act-oracle/taidl_to/generic/shapes.py:58-62`; `act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:593-603, :702-708`.
- Binding constraints:
  - Decision 16 (every run-time violation is a defined error or a defined behavior).

### Q3.7: How does the generated oracle reach the shape rules?

- Context:
  - R39 rules an import of the front end's module at run time, pinned to the same taidl tag.
  - S4 says "import or vendor"; the front end's package init loads the parser (`taidl/taidl/instruction.py:4`), so a generated oracle would load ANTLR.
  - The oracle keeps a call-time entry for expression shapes (`act-oracle/taidl_to/generic/utils.py:771-786`).
- Options:
  - A: import at run time at the pinned tag, and require `taidl.shapes` and `taidl.diagnostics` to import without the parser.
  - B: import at run time at the pinned tag, with the package init unchanged.
  - C: vendor a copy of the module into the generated oracle at generation.
- Evidence:
  - Review SD12; draft R39.
  - `shared/plan/S4.md` (Caveats); `shared/overhaul.md` (`shapes.py` rows).
  - `taidl/taidl/instruction.py:4`; `act-oracle/taidl_to/generic/utils.py:771-786`.
- Binding constraints:
  - Decision 10 (one coordinated release; generators pin one front-end tag).

### Q3.8: Where does the clause id appear in a message, and what does it name?

- Context:
  - R44 rules a suffix such as `[S9.4]`, and L numbers the S9 rows.
  - `diagnostics/scheme.md` puts `= S8.2, S9` on line 3 and a `code` such as `RF025` in brackets on line 1.
  - The test plan speaks of a `coverage.md` clause id and an "S id"; the registry holds a single `spec` field per code.
- Options for placement:
  - A: line 1 ends with `[S<id>]`; line 3 lists further sections only.
  - B: line 3 only, as the scheme has it.
  - C: both, with the suffix on line 1 and the full list on line 3.
- Options for the namespace the id names:
  - D: a spec section id such as `S8.2`.
  - E: an S9 row id numbered by L.
  - F: a generator's `coverage.md` clause id.
- Evidence:
  - Review SD7; draft R44; audit RA11.
  - `diagnostics/scheme.md` (Message format, Registry); `diagnostics/registry-seed.md` (Spec column).
  - `tests/generators.md` T11, T12, T13; `tests/corpus/format.md:9`.
- Binding constraints:
  - Decision 16 (a rejection ties to a defined rule).
  - Decision 14 (terminology precedence) for the word "clause".

### Q3.9: What do expectation rows and tests match, and who renders `coverage.md`?

- Context:
  - T11 and T12 rows match a regex of the message text; D10 asserts `code` and `object`.
  - `docs/coverage.md` is rendered from `expectations.toml` by T11 and from the registry by D6 and D11.
  - E16 records that tests pinning full text break on any wording change.
- Options for the match key:
  - A: `code` and `object`, as D10.
  - B: `code` and clause id, no regex.
  - C: a regex of message text, as T11 and T12.
  - D: code assertions everywhere plus one text test per format rule and one golden three-line test per class.
- Options for rendering:
  - E: one renderer owned by T11 that reads the registry for text and the expectations for verdicts.
  - F: two renderers, D11 for the S9 and unsupported tables and T11 for the verdict table.
- Evidence:
  - Review SD8; `diagnostics/bugs.md` E16.
  - `tests/generators.md` T11, T12 (lines 19 and 31 as cited in the review); `tests/corpus/format.md:9, :22`.
  - `diagnostics/plan/D10-tests-and-lint.md`, `D11-generated-docs.md`, `D06-unsupported-registries.md`.
- Binding constraints:
  - Decision 11 (capacity).

### Q3.10: What is the run-time error class called, and what does it guarantee?

- Context:
  - R55 names a defined run-time error a "fault": it writes no region and no state, and the generator reports it with the clause id.
  - The scheme names the class "Violation" with `<subject> violates <rule>` and states no no-write guarantee.
  - D4 and D7 use "fault" for front-end and kernel errors; the front end already has `Fault = tuple[CommonToken, str]`.
- Options:
  - A: the class is Fault, the norm and D8's tests state the no-write rule, and front-end and kernel cases are called rejection or error.
  - B: the class stays Violation and "fault" names the outcome of a violation.
  - C: "fault" names every defined error of every class, including front-end rejections.
  - D: Fault with the no-write rule as in A, and the class letter `V` renamed to avoid the scope letter `F`.
- Evidence:
  - Review SD9; draft R55.
  - `diagnostics/scheme.md` (Classes, Run-time norm, Exceptions); `diagnostics/plan/D14-run-time-norm.md`, D4, D7, D8.
  - `taidl/taidl/operations.py:167-168`.
- Binding constraints:
  - Decision 16 (no undefined behavior; every violation is a defined error or behavior).
  - Decision 14 (terminology precedence: the owner's papers first when precise).

### Q3.11: Which scope letter do shared fault codes take?

- Context:
  - The seed gives run-time codes the oracle scope (`VO001` to `VO007`), though they are language rules both generators report and T13 compares across generators.
  - The scheme places "shared V" codes in `taidl/diagnostics.json` while its scope table has no shared run-time scope.
  - `R` codes carry scope `F` so two generators reject one description with one code.
- Options:
  - A: fault codes take scope `F` in `taidl/diagnostics.json`, like `R` codes, and generators reuse them.
  - B: scope per raiser (`VO...` for the oracle, `VB...` for the backend), with the same rule under two codes.
  - C: scope per raiser, plus a shared rule table that maps the per-generator codes for T13.
- Evidence:
  - Review SD10; `diagnostics/registry-seed.md:87-93`; `diagnostics/scheme.md` (Classes, Registry, Which layer raises what).
  - `diagnostics/bugs.md` E13 (`VO005`, `VB001`); `tests/generators.md` T13.
- Binding constraints:
  - Decision 16.

### Q3.12: How does a rejected v3 construct report its replacement, and when are its codes seeded?

- Context:
  - R2, R5, R58 and R60 remove or rename constructs, and every new rejection carries a migration note.
  - The seed has codes RF013 to RF017 for `set_inplace` and `set_apart`, which v3 removes before their first release.
  - A code is never reused or renumbered once seeded.
- Options:
  - A: a rejected v3 construct reports its replacement and the clause id; no converter.
  - B: a converter tool rewrites v2 descriptions.
  - C: one migration note in the release (S8) and a generic rejection.
  - D: seed codes for removed constructs now, as in the seed.
  - E: seed codes only after the v3 API is fixed.
- Evidence:
  - Audit RA11; review SD13; draft R2, R5, R58, R60.
  - `diagnostics/registry-seed.md` RF013 to RF017; `diagnostics/plan/D04-move-checks-to-front-end.md` (Caveats); `shared/plan/S8.md`.
  - Audit R58 lists tests and spec pages that call `set_apart`.
- Binding constraints:
  - Decision 10 (breaking changes allowed, batched).
  - Decision 11 (capacity).

## For the authors

- Review SD11: delete the D14 caveat that calls the overflow decision open (`diagnostics/plan/D14-run-time-norm.md`); decision 16, R38 and R56 settle it as an error.
- Review SD14: D1, D7, D13 generate `diagnostics.rs` and `diagnostics.h` in a build script; the review proposes emitting Rust constants with the crate, a small checked C++ header, and D13 after v3.
- Review SD15: S3 owns `Accelerator.validate()`, D4 owns codes and fault order; fix the index line in `diagnostics/plan/README.md:5` that says S has no item IDs.
- Review SD17: the corpus verdict `implementation-defined` has no diagnostic class (`tests/corpus/format.md:9`); route to panel 6.
- Review SD18: S7 and T5, T6 both define the corpus, its location and its runner; S7 keeps only the mini_trn merge and generator CI.
- Review SD19: S9 sits in Q2 with no release gate; R40 puts it after v3.
- Review SD20: remove the session narration in `diagnostics/README.md:4`; say "owner decision" for "owner ruling 16 and 17" in `scheme.md`; resolve the class letter `X` versus bug ids X1 to X10; fix the X9 cite to `act-oracle/taidl_to/template.py:35-36`.
- Draft R41 (rank 0 is `[]` everywhere): `shared/bugs.md` X10 and S4 agree with it and the backend writes `[1]` today; confirm at decision.
- Audit R58: the tests and spec-draft pages that call `set_apart` and `set_no_partial_overlap` are listed in the migration item for panel 2.
