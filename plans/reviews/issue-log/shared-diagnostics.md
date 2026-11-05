# Issue log: shared-diagnostics

Source is `reviews/shared-diagnostics/review.md` (SD1 to SD20).
The shared and diagnostics tracks have no review-response file; each row was checked against `shared/plan/`, `diagnostics/`, and panel 3.
Shared items S1 to S9 are now U1 to U9 in the plans; both names are given where a row predates the rename.
Severity is the review's own label.
Panel questions are cited as Qn.k and owner decisions as ODn.

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| SD1: shape and type rules have seven owners and cycles | blocker | fixed | Q3.1; shared/plan/U4.md (absorbs L12, O6, O7, B23) |
| SD2: backend builds checks the front end moves (B21 to B23) | major | fixed | Q3.2; shared/plan/U10.md (replaces B21 and B22) |
| SD3: Q1 items depend on Q2 items | major | fixed | Q3.3; shared/plan/README.md (U2 precedes B13, no B dependency) |
| SD4: the contract exposes ANTLR trees | major | fixed | Q3.4; shared/plan/U2.md (typed node model, contract version) |
| SD5: derived facts are not published | major | fixed | Q3.5; shared/plan/U3.md (`validate()` attaches typed facts) |
| SD6: S4 does not close X1 for `@c.`-dependent shapes | major | fixed | Q3.6; shared/plan/U4.md; backend selection check |
| SD7: three spellings of the clause id | major | fixed | Q3.8; diagnostics/scheme.md (line 1 ends with `[S<id>]`) |
| SD8: tests match message regex; `coverage.md` rendered twice | major | fixed | Q3.9; shared/plan/U7.md (T11 owns the one renderer) |
| SD9: the class is "Violation", R55 says "fault" | major | fixed | Q3.10; diagnostics/scheme.md (class Fault, no-write rule) |
| SD10: fault codes carry the oracle scope | major | fixed | Q3.11; diagnostics/scheme.md (scope `F`, `VF002`) |
| SD11: D14 caveat calls the overflow decision open | minor | fixed | diagnostics/plan/D14-run-time-norm.md (Q1.2) |
| SD12: S4 says "import or vendor", R39 rules import | minor | superseded by OD25 and Q3.7: the shape module is copied into each generated oracle | shared/plan/U4.md |
| SD13: codes seeded for `set_inplace` and `set_apart` that v3 removes | minor | superseded by OD30: the markers stay as shorthand | diagnostics/registry-seed.md (RF013 to RF016); Q3.12 |
| SD14: build-script generation of `diagnostics.rs` and `.h` is heavy; D13 optional | minor | fixed: the generator emits the Rust constants with the crate, a checked C++ header holds the `I` codes, and D13 lands after v3 | diagnostics/plan/D01-registry-and-format.md; diagnostics/plan/D13-json-and-explain.md; diagnostics/scheme.md |
| SD15: S3 and D4 both define `Accelerator.validate()` | minor | fixed | shared/plan/U3.md; diagnostics/plan/README.md |
| SD16: the call site is not in the S2 contract | minor | fixed | Q3.5; diagnostics/plan/D02-call-site-and-exceptions.md |
| SD17: verdict `implementation-defined` has no diagnostic class | minor | fixed | Q6.4; tests/corpus/norms.md (allowed results) |
| SD18: S7 and T5, T6 both define the corpus | minor | fixed: U7 points to T5 for the format and T6 for the runner | shared/plan/U7.md; shared/plan/README.md |
| SD19: S9 sits in Q2 with no release gate after v3 | minor | fixed: U9 depends on U8 and lands after v3 | shared/plan/U9.md; shared/plan/README.md |
| SD20: session narration, "owner ruling", class letter `X`, X9 cite | minor | fixed in part: narration removed, "owner decision 16" in scheme.md, and the X9 cite is `template.py:32-42`; the class letter `X` still shares a letter with the bug ids X1 to X10 | diagnostics/README.md; diagnostics/scheme.md; shared/bugs.md |
