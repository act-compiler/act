# Panel 3 round 1: Opus (front-end maintainer)

Assumptions: the front end stays small enough for one maintainer to review every change, and generators read only what it documents.
Plan rows are taken as written today; rulings are options; owner decisions 10, 11, 14, 16, 19, 20 bind.

### Q3.1

Position (confidence medium): option C, two sequential items under S4 (module move, then the R31 front-end checks), together sized L; D4 attaches codes only; B23 is dropped.
1. The move alone is about 620 lines of rule code: `shapes.py` is 543 lines and the word type table is `oracle_visitor.py:19-97`.
2. The moved code must then be re-pointed from the oracle's lowered values onto taidl's own parse trees, which is a port, not a copy.
3. R31 adds three checks of meaning (region values, constant ranges, compare-type fit) that `S4.md` does not list.
4. The audit finds no evidence for R62's M size, and the component rows it merges are L, L, S, L.
5. Two items match the shared rule "land structural prep as its own commit, then the minimal behavioral delta"; the move changes no verdict, the checks change verdicts.
6. L3 is the normative spec text (Q1) and stays a separate item; S4 implements it, so the L3 to O6 to L3 cycle disappears once O6, O7, L12 are folded into S4.
7. B23 only exists to check what S4 checks; with S4 in the front end it has nothing left to do (see Q3.2).

Evidence:
- `act-oracle/taidl_to/generic/shapes.py` (543 lines); `act-oracle/taidl_to/oracle_visitor.py:19-97`.
- `shared/plan/S4.md` (What, Fail points); `shared/plan/README.md:10` (S4 sized L, depends on L3, L12, O7, B23).
- `reviews/rulings/audit.md:35` (R62 questionable, no evidence for downsizing); `language/plan/spec.md:12` (L3 is the spec rule text, L, Q1).
- `minutes/rulings.md:60` (R31 widens O7).

Would change my mind: a line-count estimate showing the R31 checks fit in under about 150 lines, which would make one L item with two commits as good as two items.

### Q3.2

Position (confidence high): option A; B21 to B23 become "call `Accelerator.validate()` and delete `spec_checks.py:394-418`", with the R25 literal check kept until `validate()` ships.
1. A backend check written in Q1 and deleted by D5 in Q2 is work whose only output is its own deletion.
2. The backend already has a literal region-count check (`check_regions`), so the interim gap R25 worries about is covered without new code.
3. B21 and B22 are each sized S and B23 is M; under decision 11 that is capacity better spent on S3 and S4.
4. Removing B21 to B23 from the S3, S4, and D4 dependency columns also removes the S3 "before B21" contradiction.

Evidence:
- `act-backend/act_backend/spec_checks.py:394-418` (literal region check, skips non-literal sizes at :413-414).
- `backend/plan/defects.md:9-11` (B21 S, B22 S, B23 M); `shared/plan/README.md:9-10`.
- `minutes/rulings.md:49` (R25).

Would change my mind: evidence that `validate()` cannot land before the backend's first v3 release, leaving the backend without B21's element-type check for a released version.

### Q3.3

Position (confidence medium-high): option C, decided per item: L15 moves to Q1; L16 already is Q1; D10 drops its L28 dependency; S2 precedes B13 and drops it.
1. S2 is the root of the S chain: S3, S5, S6, and S8 depend on it, so S2 cannot slip to Q2 without moving most of the track.
2. L15 is sized S and depends only on L25, which is S and Q1; moving L15 to Q1 costs little.
3. L16 is already Q1 in `language/plan/frontend.md:16`, so S5 and O8 need no move; the review's premise is stale for those two.
4. L28 is a banned-synonym lint scheduled after v3; D10's registry coverage test does not need it, only its synonym rule does.
5. D10 keeps the coverage test and the existing wording rules in Q1, and the banned-synonym rule joins L28 after v3.

Evidence:
- `shared/plan/README.md:8-14` (S2 is a dependency of S3, S5, S6, S8).
- `language/plan/frontend.md:15-16` (L15 S Q2 after L25; L16 S Q1); `language/plan/terminology.md:10, :13` (L25 S Q1; L28 after v3).
- `diagnostics/plan/README.md:17` (D10 depends on L28); `backend/plan/generators.md:11` (B13 built from the front end's objects).

Would change my mind: L25 slipping out of Q1, which would make S2 the item to move instead of L15.

### Q3.4

Position (confidence high): option A, with option D's trigger stated: a typed Python model with a small typed node model, trees reachable but marked unstable, `CONTRACT_VERSION`, and a per-member deprecation rule.
1. Today a generator reads `instr_semantics_ast`, `instr_input_asts`, and `constraint_asts`, which are ANTLR contexts named by grammar rules.
2. As maintainer I cannot rename a grammar rule, split a production, or move off ANTLR 4.9.2 without breaking every consumer, so option B freezes my grammar.
3. The node model needs only lines, operands, attributes, and integer expressions, which are four or five frozen dataclasses built in one tree walk.
4. MLIR's stable surface is its op, attribute, and type objects, never the parser; StableHLO adds a serialized form (VHLO) only for consumers across versions.
5. No non-Python consumer exists, so a serialized form now (option C) is a second contract to keep in sync with no reader.
6. `CONTRACT_VERSION` plus "a member is deprecated for one minor release before removal" gives outside teams the only guarantee they need.
7. The trigger for a serialized form is the second non-Python consumer, recorded in the status doc.

Evidence:
- `taidl/taidl/instruction.py:77-84` (`constraint_asts`, `update_asts`, `instr_input_asts`, `instr_semantics_ast` members); `taidl/taidl/__init__.py:9` (four exports).
- `taidl/pyproject.toml:15` (`antlr4-python3-runtime==4.9.2`).
- https://openxla.org/stablehlo/compatibility (portable serialized artifact exists for cross-version consumers).
- `minutes/rulings.md:75` (R42).

Would change my mind: a concrete non-Python generator scheduled inside the two quarters.

### Q3.5

Position (confidence medium-high): option A; `validate()` attaches result type and shape per line, folded sizes, the value-to-region map, and the call site as documented typed members; the rule functions are importable as a by-product, not the contract.
1. S4 computes these facts anyway; publishing them costs a field per fact, while not publishing them recreates the X1 to X10 duplication in every new generator.
2. In MLIR, inferred result types live on the IR itself, not in a side analysis object; generators read the op, not a separate result.
3. A separate analysis object (option B) must be threaded through every generator call and can drift from the instruction it describes.
4. Helper functions alone (option C) make each generator re-run the analysis and re-decide what to do with non-literal dims.
5. The members must read as absent (raise) before `validate()` runs, so no generator reads half-built facts.
6. The call site is needed for line 2 of every coded message from an outside generator, so it is a contract member.

Evidence:
- `shared/plan/S4.md` (What: rules run from `validate()`); review SD5, SD16.
- https://github.com/openxla/stablehlo/blob/main/docs/type_inference.md (shape functions infer result types stored on the op).
- `shared/interface.md` ("Where generators re-parse or re-derive").

Would change my mind: a generator that needs facts per `@c.` binding, which a static member cannot hold.

### Q3.6

Position (confidence medium): option C; the front end decides agreement for literal dims, the oracle keeps the per-call check as a defined fault, and S4's benefit is restated as closing X1 for literal dims only.
1. `agree()` treats any non-integer dim as agreeing, so the front end can reject only provable literal mismatches.
2. StableHLO verifiers make the same choice: an unknown dim is accepted as compatible at verification.
3. StableHLO then calls a run-time mismatch undefined behavior, which decision 16 forbids for TAIDL; the oracle's per-call check turns it into a defined fault (`VO006`).
4. The oracle checks every iteration of every loop at the call (`check_shapes`), so its verdict is exact, while symbolic affine normalization (option A) is sound only for an affine subset and adds an algebra module to the front end.
5. A symbolic checker that rejects a dim pair it cannot prove equal would reject valid descriptions; one that accepts them adds nothing over today.
6. The backend renders a shape check per rewrite, so whether a mismatch there is a non-match or a coded error is a backend question for panel 4; the front end supplies `agree()` for bound dims either way.

Evidence:
- `act-oracle/taidl_to/generic/shapes.py:58-62` (`agree`); `act-oracle/taidl_to/generic/utils.py:771-786` (per-iteration `check_shapes`).
- https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md (shape mismatches at run time are undefined behavior).
- StableHLO `stablehlo/dialect/Base.h` (`verifyCompatibleShapeWithBounds` accepts a non-constant dim).
- `act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:593-603` (shape check rendered per rewrite).

Would change my mind: corpus evidence that most real `@c.` dims are one shared affine form, which a 50-line normalizer would decide exactly.

### Q3.7

Position (confidence high): option A; import at run time at the pinned tag, and `import taidl.shapes` and `import taidl.diagnostics` load no ANTLR module.
1. The oracle's `shapes.py` has no import statement at all, so the module is parser-free by construction.
2. `taidl/__init__.py:5-7` imports `accelerator` and `instruction`, and `instruction.py:4` imports `antlr4`, so any `taidl.*` import loads the parser today.
3. A lazy package init (module-level `__getattr__`, PEP 562) keeps `from taidl import Accelerator` working while `taidl.shapes` stays light; this is about ten lines.
4. Vendoring (option C) creates a second copy, which is the duplication R39 and S4 exist to remove.
5. A test that imports `taidl.shapes` and asserts `antlr4` is not in `sys.modules` keeps the property from regressing.

Evidence:
- `act-oracle/taidl_to/generic/shapes.py:1-20` (no imports); `taidl/taidl/__init__.py:5-7`; `taidl/taidl/instruction.py:4`.
- https://peps.python.org/pep-0562/ (module `__getattr__`).
- `minutes/rulings.md:72` (R39).

Would change my mind: a run-time environment for generated oracles where installing taidl itself is not allowed.

### Q3.8

Position (confidence medium): placement B (line 3 only, first id is the code's registry `spec` field); namespace D (a spec section id, where a numbered S9 row is just a subsection id such as `S9.4`).
1. Line 1 already carries the code, and the registry maps each code to one `spec` field, so a clause suffix on line 1 repeats a fact the code determines.
2. rustc puts the code on line 1, the location on line 2, and secondary facts as `= note:` lines, which is the layout the scheme already copies.
3. Options D and E unify if L numbers S9 rows as subsections, so one namespace serves both.
4. A generator's `coverage.md` id (option F) differs per generator, so two generators would cite one rule under two names, which breaks T13.
5. Decision 16 is met by either placement, because the code and the id both tie the rejection to a rule.

Evidence:
- `diagnostics/scheme.md:36-46` (three-line format, 160-character rule at :50).
- https://rustc-dev-guide.rust-lang.org/diagnostics.html (`error[E0000]:`, `-->`, `= note:` layout).
- `diagnostics/registry-seed.md` (one Spec column per code).

Would change my mind: an owner preference for the id on line 1, which is a value judgment once the namespace is settled.

### Q3.9

Position (confidence high): match key D (code everywhere, one text test per format rule, one golden three-line test per class); renderer E (one renderer, owned by T11).
1. E16 records that tests pinning text break on every wording change; a regex of text (option C) is a softer form of the same coupling.
2. The code is stable by registry rule, so a code key survives the L26 glossary pass and every rewording.
3. MLIR's diagnostic verifier matches message substrings because MLIR has no codes; TAIDL has codes, so it can key on them.
4. Two renderers of one page (option F) will disagree; one renderer that reads the registry for text and the expectations for verdicts has one owner.

Evidence:
- `diagnostics/bugs.md` E16; `tests/corpus/format.md:9, :22`; review SD8.
- https://mlir.llvm.org/docs/Diagnostics/ (diagnostics carry location, severity, message; no stable codes).

Would change my mind: a class whose messages carry values the code cannot express and that tests must pin, such as iteration indices in a fault.

### Q3.10

Position (confidence medium): option A with the letter `V` kept (which also meets D's aim): the run-time class is Fault, it writes no region and no state, and the front end's `Fault` alias becomes `Rejection`.
1. A run-time error needs a guarantee about what it leaves behind, or two generators can differ on state after it and T13 has nothing to compare.
2. "Writes no region and no state" is the simplest norm that is easy to explain, as decision 16 asks.
3. StableHLO leaves run-time shape mismatch undefined; TAIDL must define it, and a name plus a no-write rule is the definition.
4. The front end's `Fault = tuple[CommonToken, str]` names a parse-time rejection, so keeping it would give one word two meanings in one code base.
5. Keeping the code letter `V` avoids a collision with scope letter `F` without renaming any seeded code.

Evidence:
- `taidl/taidl/operations.py:167-168` (`Fault` alias for a front-end rejection).
- `diagnostics/scheme.md:57` (`V` message form); `minutes/rulings.md:27` (R55).
- https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md (run-time shape mismatch left undefined).

Would change my mind: the owner's papers using "fault" for front-end errors, which decision 14 would put first.

### Q3.11

Position (confidence high): option A; fault codes take scope `F` in `taidl/diagnostics.json`, and the raising tool is named by the line-1 tool prefix.
1. A fault is a language rule, so one rule gets one code, as `R` codes already do.
2. T13 compares verdicts across generators; one code makes that a string compare, while option C needs a mapping table to maintain.
3. The scheme already prefixes line 1 with the tool name (`act-backend: error[...]`), so the raiser is never lost.
4. A per-raiser code (option B) gives one rule two codes forever, since codes are never renumbered.

Evidence:
- `diagnostics/scheme.md:45` (tool prefix); `diagnostics/registry-seed.md:87-93` (`VO001` to `VO007` are language rules with spec sections).
- `diagnostics/bugs.md` E13 (`VO005` and `VB001` for one rule).

Would change my mind: a fault whose rule genuinely differs between generators, which would itself be a decision-16 defect.

### Q3.12

Position (confidence medium-high): options A and E; a removed construct raises one coded rejection that names its replacement and clause id; codes are seeded only after the v3 API is fixed; no converter.
1. RF013 to RF017 check `set_inplace` and `set_apart`, which v3 removes, so seeding them now burns never-reused codes on rules that never ship.
2. A deleted method raises a bare `AttributeError` with no code, so the front end keeps one table of removed names and raises one code, such as "'set_apart' is removed in v3; use '<replacement>'".
3. One code with a table is smaller than one stub per removed member and covers R2, R5, R58, R60 alike.
4. A converter must rewrite Python programs that call the API, which is open-ended work against decision 11.
5. A generic rejection with a release note (option C) gives the user no path from the error to the fix.

Evidence:
- `diagnostics/registry-seed.md` RF013 to RF017; review SD13; audit RA11.
- https://peps.python.org/pep-0387/ (Python's policy: removed APIs warn and name the replacement before removal).

Would change my mind: an inventory showing many external v2 descriptions, which would justify a converter.
