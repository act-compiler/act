# Review: shared code and diagnostics

Targets: `plans/shared/` (S1 to S9) and `plans/diagnostics/` (D1 to D14), against the owner decisions, rulings R39 to R44, R55 to R62, and `plans/tests/`.
Paths in the table are relative to /home/user; plan paths are relative to `act/plans/`.

## Verdict

The direction is right: one checker in the front end, a typed Python contract (R42), and one coded message form will remove the X1 to X10 divergences.
The plans are not yet schedulable: shape and type rules have six owners and a dependency cycle, and several Q1 items depend on Q2 items.
The object contract is complete for the two in-tree generators but not stable enough for an outside team, because it exposes ANTLR parse trees and omits the facts the new checker derives.
The diagnostic scheme fits Python, Rust and C++ with little machinery, but it disagrees with R44 on where the clause id goes, with R55 on what a run-time error is called, and with the test plan on whether tests match codes or regexes.
Approve after the one blocker and the nine majors below are resolved in the plan text.

## Findings

| ID | Severity | Where | Finding | Proposed change |
| --- | --- | --- | --- | --- |
| SD1 | blocker | S4, D4, L3, L12, O6, O7, B23 | Shape and per-word type rules are owned by seven items with cycles: L3 needs O6 and O6 needs L3; L12 needs O7 and O7 needs L12; O7 needs B23 and B23 needs L12. R62 merges L3, L12, O6, O7 into one M item, yet S4 is still L and lists them as separate dependencies. | Make S4 the single R62 item, sized M; D4 only attaches RF049 and RF050 to it; drop B23 (see SD2); update every index that names L3, L12, O6, O7 for this work. |
| SD2 | major | S3, S4, D4, B21, B22, B23 | The backend plan builds its own region-count, region-type and shape checks (B21, B22, B23) that S3 and S4 move into the front end; S3 and D4 even depend on them, so a copy is written in order to be deleted by D5. S3's table lists B21, B22 as dependencies while S3.md says "before B21, B22"; S4 likewise with B23. | Redefine B21 to B23 as "call `Accelerator.validate()` first and delete `spec_checks.py:394-418`"; remove them from the S3, S4 and D4 dependency columns. |
| SD3 | major | S2, S5, D10, O8 | Q1 items depend on Q2 items: S2 (Q1) on L15 (Q2); S5 (Q1) on L16 (Q2); D10 (Q1) on L28 (Q2); O8 (Q1) on L16. S2's table depends on B13 while S2.md says "before B13". | Move L15 and L16 to Q1 or S2, S5, D10 to Q2; let S2 precede B13 and drop B13 from its dependencies. |
| SD4 | major | S2, shared/interface.md | The contract a third party must read includes `IDLV2Parser.ModuleContext`, a dict keyed by ANTLR `TerminalNode`, and visitors from `taidl.antlr4`, which `taidl/taidl/__init__.py:9` does not export; it ties every outside generator to ANTLR 4.9.2 (`taidl/pyproject.toml:15`) and to grammar rule names. | S2 adds a small typed node model for lines, operands, attributes and integer expressions; trees stay reachable but marked unstable; export a `CONTRACT_VERSION` and a deprecation rule per member. |
| SD5 | major | S2, S4 | After S4 the front end computes each line's result type and shape, folded sizes and the value-to-region map, but S2 does not publish them; a simulator or RTL generator would re-derive them, the duplication this track removes. | `validate()` attaches the derived facts to each instruction as documented, typed members, listed in the S2 doc page. |
| SD6 | major | S4, X1 | S4 claims to close X1, but the checker it moves treats any non-integer dim as agreeing (`act-oracle/taidl_to/generic/shapes.py:58-62`); `@c.`-dependent shapes are only decided at an oracle call, and the backend has no call. | Either normalize affine `@c.` dims symbolically in the front end, or have the backend check agreement at selection with a coded error; restate S4's benefit to what it closes. |
| SD7 | major | diagnostics/scheme.md, R44, tests/generators.md:38, tests/corpus/format.md:9 | Three spellings of the clause id: R44 rules a suffix such as `[S9.4]`; the scheme puts `= S8.2, S9` on line 3; the test plan speaks of a coverage.md clause id. | Follow R44: line 1 ends with `[S<id>]`; line 3 lists further sections only; tests and corpus key on that id. |
| SD8 | major | tests/generators.md:19, :31; tests/corpus/format.md:9, :22; D6, D10, D11 | Test rows match a regex of message text while D10 asserts `code` and `object`; and `docs/coverage.md` is rendered twice, from `expectations.toml` (T11) and from the registry (D6, D11). | Expectation and corpus rows carry the code and clause id, no regex; one renderer owned by T11 reads the registry for text and the expectations for verdicts. |
| SD9 | major | diagnostics/scheme.md (Classes, Run-time norm), D4, D7, R55 | R55 names a run-time error a "fault" that writes no region and no state; the scheme calls the class "Violation" with `<subject> violates <rule>`, omits the no-write guarantee, and D4 and D7 use "fault" for front-end and kernel errors. | Name the class Fault per R55, state the no-write rule in the norm and in D8's tests, and use "rejection" or "error" for front-end and kernel cases. |
| SD10 | major | registry-seed.md:87-93, scheme.md (Registry) | Fault codes carry the oracle scope (VO001 to VO007) though faults are language rules both generators report and T13 compares across generators; the scheme also puts "shared V" codes in the taidl registry under an `O` scope. | Fault codes take scope `F` in `taidl/diagnostics.json`, like `R` codes; generators reuse them. |
| SD11 | minor | D14 caveat | Says the overflow decision is open; owner decision 16, R38 and R56 settle it as an error. | Delete the caveat. |
| SD12 | minor | S4 caveat, R39 | S4 says "import or vendor"; R39 rules import at the pinned tag. The generated oracle then imports `taidl`, whose package init loads the parser (`taidl/taidl/instruction.py:4`). | Follow R39; require `taidl.shapes` and `taidl.diagnostics` to import without the parser. |
| SD13 | minor | registry-seed.md RF013 to RF017 | Codes are seeded for `set_inplace` and `set_apart`, which R5 and R58 remove in v3; a never-reused code is then retired before its first release. | Do not seed them; seed after the v3 API is fixed. |
| SD14 | minor | D1, D7, D13 | Generating `diagnostics.rs` and `diagnostics.h` in a build script is heavier than needed: the backend crate is already emitted by the Python generator, and the C++ allocator needs a few `I` codes. D13 is optional work in a two-quarter release. | Emit the Rust constants with the crate; keep a small C++ header checked by a test; defer D13 past v3. |
| SD15 | minor | diagnostics/plan/README.md:5, D4, S3 | The index says S has no item IDs and "D1 is the shared code"; D4 and S3 both define `Accelerator.validate()`. | S3 owns `validate()`; D4 owns codes and fault order; update the index line. |
| SD16 | minor | S2, D2 | D2 stores a call site on each built object, but the S2 contract does not list it; an outside generator cannot fill line 2 of a `U` message. | Add the call-site member to the S2 doc page. |
| SD17 | minor | tests/corpus/format.md:9 | The verdict `implementation-defined` has no diagnostic class, and the scheme allows "a defined error or a defined behavior, nothing in between". | Drop the verdict or map each case to a fault or a stated behavior. |
| SD18 | minor | S7, T5, T6 | S7 and T5, T6 both define the corpus, its location and its runner. | S7 points to T5 for format and T6 for the runner, keeping only the mini_trn merge and generator CI. |
| SD19 | minor | S9, R40 | R40 puts docker and CI consolidation after v3; S9 sits in Q2 with no release gate. | Mark S9 as after S8. |
| SD20 | minor | diagnostics/README.md:4, scheme.md, shared/bugs.md:3 | README narrates the session ("spawn tool was not available"); scheme cites "owner ruling 16" and "ruling 17" for owner decisions; class letter `X` collides with bug ids X1 to X10; X9 cites `act-oracle/taidl_to/template.py:33-34`, where the check sits at :35-36. | Remove the narration; say "owner decision"; rename the bug prefix or the class letter; fix the cite. |

## Spot checks against code

- Nineteen catalog rows, past the ten asked, match their file, line and wording:
  - `taidl/taidl/names.py:10, :12, :14`
  - `taidl/taidl/accelerator.py:41, :43, :56`
  - `taidl/taidl/instruction.py:94, :119, :455`
  - `act-backend/act_backend/spec_checks.py:26, :48, :75, :210, :415`
  - `act-oracle/taidl_to/semantic_generator.py:16, :40, :45, :53`; `act-oracle/taidl_to/oracle_visitor.py:289`
- Five duplication claims hold:
  - X2: backend Rust appends bare `/` and `%` (`act-backend/act_backend/expression_writer.py:23-31`); the oracle writes `//` (`act-oracle/taidl_to/expression_writer.py:21`).
  - Element types: 13 in `ELEMENT_TYPES` (`act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:24-37`) against 19 in the oracle (`act-oracle/taidl_to/oracle_visitor.py:13-16`).
  - Rank 0: backend `['1']` (`act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:139-142`); oracle `[]` (`act-oracle/taidl_to/generic/shapes.py:31-35`).
  - Template fill: the oracle returns an unknown `{{key}}` unchanged (`act-oracle/taidl_to/template.py:35-36`).
  - Zero divisor: taidl matches the literal text `0` only (`taidl/taidl/instruction.py:413-415`); `cleanup.sh` is byte-identical in act-backend and act-oracle.
- One citation is off by two lines (SD20); no claim was wrong.

## What is good

- The consolidate-semantics, decouple-later ruling (SD5 of shared/decisions.md, R42) is the right size for two Python consumers.
- The interface table cites every member each generator reads, which makes S2's contract test straightforward.
- The duplication table names a best copy per row, and each X bug carries a reproducer such as `(@c.n - 1) / 4 + 1` at n = 0.
- The class split R, U, V, K, I, X separates "your description is wrong" from "this generator lacks it" cleanly, and `R` from the front end only gives every generator one verdict.
- Keeping `ValueError`, `TypeError`, `SyntaxError` as bases preserves S9 and existing callers; the multiple-inheritance classes construct and print as specified.
- The three-line format is plain text, needs no library in Rust or C++, and degrades to one line for `I` and `X`.
- Exit statuses by class and a panic hook give the backend binary a testable ending for every failure.
- The catalog is accurate and classes every row, so the registry seed is a mechanical step.
