# Milestones

Six outcome milestones; each closes when every exit criterion holds, whatever the date.
A milestone's items are the waves that end at it; an item belongs to the earliest milestone whose exit needs it (Q8.7).
Each criterion names the CI job, test item, cold-start trial, or named check that makes it observable (Q8.8).
Each milestone holds one negative criterion that a test checks.

## Summary

| Milestone | Outcome | Ends with wave | Items | Weight |
| --------- | ------- | -------------- | ----- | ------ |
| M1 | The specification and the typed contract are in force | 2 | 97 | 166 |
| M2 | One check function serves every generator | 3 | 58 | 122 |
| M3 | Generators read the checked model and the front end's codes | 4 | 49 | 122 |
| M4 | v3 is released on released pins | 5 | 27 | 74 |
| M5 | Usable by anyone | 6 | 5 | 14 |
| M6 | The backend runs on the MLIR platform with the C++ e-graph library and a torch entry | 10 | 20 | 66 |

Wave 7 (10 items, weight 25) follows M5 as listed follow-ons, U9, L19, L20, L23, and D13 among them.
Waves 8 and 9 end at no milestone; M6 counts waves 8 to 10 (wave-8-10.md).

## Standing criteria

- Before M4, private CI installs the private heads in a fresh venv (A26) and runs the quickstart in the image (A20).
- From M4 on, the same check installs the released act 3.0 from the public index and runs the README snippet.

## M1: the specification and the typed contract are in force

- Every operation has a normative shape and element type rule a clause can cite (L3, Q8.4 step a).
- The spec skeleton ships with section ids, and every test tag resolves to one (L1, T1).
- The object model contract test passes for the five fixed names (L29).
- Every same-buffer input-output pair expands to exactly one relation, exact-or-apart by default (L36, L35).
- The region, in-place, and overlap fault rows of S1.3.3 to S1.3.6 hold: the O30 matrix expects each fault with its S1.3 row id, and a fault leaves every region and all state unchanged (L5, O30).
- The typed handoff carries each pair's relation and tie to the allocator (B39).
- The oracle checks the constraint list before any load and evaluates on the pre-state, and the overlap test matrix passes (O10, O30).
- An attribute takes an optional `s<w>` or `u<w>` type, `s64` by default, and the allocator domain follows it (L37, B38).
- Every surviving test carries a language specification section tag (T1).
- The typed front-end contract carries a version, and its contract test passes (U2).
- The end-to-end gate is green in deterministic mode: goldens, chosen assembly, candidate logs, oracle run (B3 on B29).
- B7 names its shape rule source in its exit line (Q8.2).
- The case-study sketches C1 to C4, C9 to C11, and C13 run as an executable corpus (C11).
- Every wave 1 gate has its named outcome (waves.md gate row).
- Assisted weeks are recorded per wave 1 item, the assist factor is recorded per size class (S, M, L, with L3 as the L item), and dates are set from them (Q8.9).
- Every wave 1 gate and every size-L item has a named owner and a close date (waves.md rules).
- The repo topology, branching, private-to-public sync, private CI owner, and 2.x maintenance policy are written (A32).
- The three tutorials sit on their tags, none remains on master, and README.md, pyproject.toml, and the workflow state one tutorial lint scope (A11 with A22).
- Negative: the ASCII and one-sentence-per-line gate finds no violation in tracked text files (A7).

## M2: one check function serves every generator

- `taidl.check` returns a frozen `Model` and raises one `Rejection` listing every problem; `python -m taidl check spec.py` runs it standalone (L43, U3).
- Every generator entry takes a `Model` or calls `check` once, and its body reads only the `Model`, checked by a test in each generator (U10, B21 with B22).
- The traced Python surface builds a `Model`, and each rejection names the author's file and line (U12 with L42's exit, D17).
- The front end holds the shape and per-word element-type rules, and the S8.3 table is generated from them (U4, Q8.4 step b).
- The agreement test compares B7's Rust shape table with the taidl shape module and passes (Q8.2).
- The spec consistency job runs on every push (L8).
- Every loop is unrolled within the Q5.1 bound on the O19 stress kernel: 60 s trace plus compile, 2x per existing loop test (O20).
- Backend runtime diagnostics carry codes, and C++ errors return instead of asserting (D7, B25).
- The allocator places each pair under its relation, with one allocation row per relation (B30, B40).
- Generator metadata reaches each generator through the typed contract (L40).
- The guides for adding a generator exist (L41, A27): the worked generator of L41 is the U11 template, A27 refers to it without restating it, and it runs in CI against the corpus on the pinned taidl, and a reader builds a stub generator from the A27 page alone (A27 verification).
- A failing candidate is dropped with a recorded reason, and the oracle raises only on invalid input, naming instruction and clause (B41, O31).
- Integer `divide`, `remainder`, and float-to-integer `convert` take fixed values, with no data fault (O23).
- Negative: no `set_shapes_` in the generated crate and no C++ candidate parser in act-backend (B6, B4; backend/plan/isel.md:35).

## M3: generators read the checked model and the front end's codes

- Run-time-independent checks live in the front end, and the duplicates are deleted from each generator (D4, D5).
- The conformance corpus job passes with expected verdicts (U7, after T6).
- The oracle reads only the checked `Model` and carries the copied shape module stamped with its taidl version (O7, owner 25).
- Backend checks after `check` reject an architect-correctable fault and exclude an unsupported instruction with a reason (B14).
- The exclusion report is printed at generation, stored in the crate, and named when a kernel needs an excluded instruction (B46, D18).
- The failure policy is in force for every generator: the D16 endings tests pass, `act-backend/tests/test_endings.py` with a row per ending class of `failure-policy.md` and `act-oracle` with one test per `KO021` to `KO040` code, and no `I` code stops a run (D16).
- The backend validates its input against a documented input specification and detects formats other than XLA HLO text (B42, B43).
- The converter script runs on the corpus under taidl 2.x and ships in the last 2.x release with act's own ISA specifications converted, each passing T37's per-file `Model` check (U13 with L44's exit, A28).
- The U12 spelling is frozen before that release (U12).
- The two cold-start trial participants are named (A24, K18).
- One architect outside the team has written an ISA specification on the traced surface, and the findings are filed against L30 and D15 (A24).
- The check verdict runner passes on the seeded corpus (T6, T7).
- The registry coverage test and wording lint pass (D10).
- Every message carries its clause id on line 1 (D3).
- Every IR-to-IR rewrite rule has a name and a test (B8, P3).
- The overlap matrix, rejection, and cross-check cases pass (T30).
- The oracle checks attribute types as implicit range constraints, and the width layer cases pass (O32, T32).
- The oracle's output is identical with the encoding block, derived attributes, and generator metadata stripped (O33).
- Negative: the oracle emitter holds no string templates for emitter calls (O3).

## M4: v3 is released on released pins

- The release gate passes on the private heads before any v3 tag: encoding unit, T12, T13, T36, T37, T38, and the deletion (A31).
- taidl 3.0 is released first; taidl-to 3.0 and act-backend 2.0 pin it and are released; act 3.0 pins them and is released last (U8, then A23; owner 26).
- The tutorial draft and the Claude-driven fresh-container run pass on the private heads before U8 (A24, Q8.8).
- The C12 gap-matrix pass runs again on the release candidate, and a generated report gives the backend share of instructions per case study (C12).
- The v3 input is stated as XLA HLO text, with the JAX and PyTorch export route, in the README and the input specification (A23, B42).
- The release followed the A16 checklist, and the scheduled compatibility build is green on the released pins (A16, A18).
- Sibling pins and dependency floors agree with act (A17, A15).
- Differential tests pass, oracle against backend, on the supported intersection (T13).
- The expectation runner passes every corpus entry against its row, whose status is `supported`, `partial`, `excluded`, or `rejected`, with the documented code for the last two (T12).
- The corpus reaches the generators through the released taidl 3.0 pin (T16, T19).
- Each generated oracle's copied shape module matches its taidl version (T23).
- CI tiers run in deterministic mode within their budgets (T27).
- The final deletion pass leaves no PR-defense test (T18, T4).
- The encoding unit lands whole before taidl 3.0: `decode(encode(x)) == x` holds on the corpus ISA specification, and the derived attribute cases pass (L38, L39, B44, B45, T33, T34; Q10.4).
- The backend overlap rows pass: allocation per relation, and a copy for a tied shared input (T31, B11).
- The generator metadata cases pass, and the template generator ships with its expectations table (T35, U11).
- The surface, `check`, and exclusion cases pass (T38).
- Each corpus file's `Model` from the v2 string form equals the `Model` traced from its converted file, up to call sites (T37).
- taidl 3.0 holds no string surface, no `taidl/antlr4/`, no grammar tests, and no ANTLR dependency (U14 with L45's exit).
- The failure policy cases pass for every generator (T36).
- Negative: no public repo holds a pre-release tag, a commit pin, or a branch pin (named check over tags and manifests, A17).

## M5: usable by anyone

- One tutorial is written on the released v3, its paper titles and setup text match the papers, and every exercise has a runnable solution (A24 with A14, A13).
- The tutorial solutions run in CI against the released pins (A21).
- A cold-start trial passes: two people outside the team, a fresh environment, the released act 3.0, the README and tutorial, a change to the Q7.5 example ISA specification, no maintainer help, within 4 hours each (Q8.8 resolution rows).
- A Claude-driven fresh-container run of the same steps passes as a separate CI job; it does not replace the trial (Q8.8).
- Negative: the README holds no news, roadmap, or release-history lines (A3).

## M6: the MLIR platform, the C++ e-graph library, and the torch entry

- Public builds pin one released StableHLO version, derive LLVM from its tag, and ship a prebuilt importer (U15).
- The importer reads StableHLO portable artifacts, and the StableHLO corpus passes differentially against the HLO path (B47, T39).
- The dual-engine harness passes the three-tier swap gate, and egg is deleted (B52).
- The torch adapter's pairing check has a recorded result, and torch tests pass per operation and dtype with a coverage table (B53, A29, T40).
- The oracle runs on fsim with every golden and norm row holding (O36).
- The platform gate passes: B3 on the one C++ driver, oracle on fsim, and the Rust crate is deleted (B56, T41).
- Negative: the e-graph library and the allocator include no MLIR header, and no public build file writes an LLVM commit (named check).
