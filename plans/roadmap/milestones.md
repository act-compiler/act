# Milestones

Five outcome milestones; each closes when every exit criterion holds, whatever the date.
A milestone's items are the waves that end at it; an item belongs to the earliest milestone whose exit needs it (Q8.7).
Each criterion names the CI job, test item, cold-start trial, or named check that makes it observable (Q8.8).
Each milestone holds one negative criterion that a test checks.

## Summary

| Milestone | Outcome | Ends with wave | Items | Weight |
| --------- | ------- | -------------- | ----- | ------ |
| M1 | The specification and the typed contract are in force | 2 | 97 | 166 |
| M2 | One front end validates for every generator | 3 | 60 | 132 |
| M3 | Generators read the front end's checks and codes | 4 | 42 | 105 |
| M4 | v3 is released on released pins | 5 | 23 | 66 |
| M5 | Usable by anyone | 6 | 7 | 16 |

Wave 7 (10 items, weight 30) follows M5 as listed follow-ons, U9, L19, L20, and L23 among them.

## Standing criteria

- The Q8.1 graph checker passes at every milestone: each ID in one row, no unknown token, no cycle, no dependency on a later wave.
- Before M4, private CI installs the private heads in a fresh venv (A26) and runs the quickstart in the image (A20).
- From M4 on, the same check installs the released act 3.0 from the public index and runs the README snippet.

## M1: the specification and the typed contract are in force

- Every operation has a normative shape and element type rule a clause can cite (L3, Q8.4 step a).
- The spec skeleton ships with section ids, and every test tag resolves to one (L1, T1).
- The object model contract test passes for the five fixed names (L29).
- Every same-buffer input-output pair expands to exactly one relation, exact-or-apart by default, and the region, in-place, and overlap fault rows hold (L36, L35, L5).
- The typed handoff carries each pair's relation and tie to the allocator (B39).
- The oracle checks the constraint list before any load and evaluates on the pre-state, and the overlap test matrix passes (O10, O30).
- An attribute takes an optional `s<w>` or `u<w>` type, `s64` by default, and the allocator domain follows it (L37, B38).
- Every surviving test carries a language specification section tag (T1).
- The typed front-end contract carries a version, and its contract test passes (U2).
- The end-to-end gate is green in deterministic mode: goldens, chosen assembly, candidate logs, oracle run (B3 on B29).
- B7 names its shape rule source in its exit line (Q8.2).
- The case-study sketches C1 to C4, C9 to C11, and C13 run as an executable corpus (C11).
- Every wave 1 gate has its named outcome (waves.md gate row).
- Assisted weeks are recorded per wave 1 item, and dates are set from them (Q8.9).
- Negative: the ASCII and one-sentence-per-line gate finds no violation in tracked text files (A7).

## M2: one front end validates for every generator

- Every generator calls `validate()` first, checked by a test in each generator (U10, B21 with B22).
- The conformance corpus job passes with expected verdicts (U7).
- The front end holds the shape and per-word element-type rules, and the S8.3 table is generated from them (U4, Q8.4 step b).
- The agreement test compares B7's Rust shape table with the taidl shape module and passes (Q8.2).
- The spec consistency job runs on every push (L8).
- Every loop is unrolled within the owner-set bound on the O19 stress kernel (O20).
- Backend runtime diagnostics carry codes, and C++ errors return instead of asserting (D7, B25).
- The allocator places each pair under its relation, with one allocation row per relation (B30, B40).
- Generator metadata reaches each generator through the typed contract, and the guides for adding a generator exist (L40, L41, A27).
- The failure policy is in force: a failing candidate is dropped with a recorded reason, and the oracle raises only on invalid input, naming instruction and clause (D16, B41, O31).
- The backend validates its input against a documented input specification and detects formats other than XLA HLO text (B42, B43).
- Integer `divide`, `remainder`, and float-to-integer `convert` take fixed values, with no data fault (O23).
- Negative: no `set_shapes_` in the generated crate and no C++ candidate parser in act-backend (B6, B4; backend/plan/isel.md:35).

## M3: generators read the front end's checks and codes

- Run-time-independent checks live in the front end, and the duplicates are deleted from each generator (D4, D5 with B23).
- The oracle runs the front end's checks and carries the copied shape module stamped with its taidl version (O7, owner 25).
- The front-end verdict runner passes on the seeded corpus (T6, T7).
- The registry coverage test and wording lint pass (D10).
- Every message carries its clause id on line 1 (D3).
- Every IR-to-IR rewrite rule has a name and a test (B8, P3).
- The overlap matrix, rejection, and cross-check cases pass (T30).
- The oracle checks attribute types as implicit range constraints, and the width layer cases pass (O32, T32).
- The oracle's output is identical with the encoding block, derived attributes, and generator metadata stripped (O33).
- Negative: the oracle emitter holds no string templates for emitter calls (O3).

## M4: v3 is released on released pins

- taidl 3.0 is released first; taidl-to 3.0 and act-backend 2.0 pin it and are released; act 3.0 pins them and is released last (U8, A23, owner 26).
- The release followed the A16 checklist, and the scheduled compatibility build is green on the released pins (A16, A18).
- Sibling pins and dependency floors agree with act (A17, A15).
- Differential tests pass, oracle against backend, on the supported intersection (T13).
- The expectation runner reports supported, or unsupported with the documented code, for every corpus entry (T12).
- The corpus reaches the generators through the released taidl 3.0 pin (T16, T19).
- Each generated oracle's copied shape module matches its taidl version (T23).
- CI tiers run in deterministic mode within their budgets (T27).
- The final deletion pass leaves no PR-defense test (T18, T4).
- The encoding unit lands whole before taidl 3.0: `decode(encode(x)) == x` holds on the corpus ISA specification, and the derived attribute cases pass (L38, L39, B44, B45, T33, T34; Q10.4).
- The backend overlap rows pass: allocation per relation, and a copy for a tied shared input (T31, B11).
- The generator metadata cases pass, and the template generator ships with its expectations table (T35, U11).
- The failure policy cases pass for every generator (T36).
- Negative: no public repo holds a pre-release tag, a commit pin, or a branch pin (named check over tags and manifests, A17).

## M5: usable by anyone

- One tutorial is written on the released v3, and every exercise has a runnable solution (A24, A13).
- The tutorial solutions run in CI against the released pins (A21).
- A cold-start trial passes: two people outside the team, a fresh environment, the released act 3.0, the README and tutorial, one small ISA specification, no maintainer help, within a stated time limit (Q8.8 panel default; the owner sets the parameters).
- A Claude-driven fresh-container run of the same steps passes as a separate CI job; it does not replace the trial (Q8.8).
- Negative: the README holds no news, roadmap, or release-history lines (A3).
