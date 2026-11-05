# Test suite track plan

Planning files for the test-suite work across taidl, act-backend, act-oracle, and act.
Every item carries a T id, a size, a quarter, a `Breaking` bullet, and the existing files it changes.
Nothing here was run against the repos; every claim cites a file and line read on the planning day.

## Files and folders

| entry | holds |
| --- | --- |
| [review.md](review.md) | what works, what is weak, verdict per test file, counts |
| [bugs.md](bugs.md) | test defects TB1 to TB22 with fix sketches |
| [overhaul.md](overhaul.md) | target layout, per-file and per-fixture action, ordered steps |
| [inventory/](inventory/README.md) | per-repo inventory tables, duplication and coupling, items T1 to T4 |
| [corpus/](corpus/README.md) | the conformance corpus, items T5 to T10, T24 to T26, T29, T30, T32 to T36, T38, coverage of the outline |
| [generators.md](generators.md) | expectations, differential tests, end-to-end, goldens, CI tiers, items T11 to T15, T23, T27, T31 |
| [migration.md](migration.md) | ordered PR steps and what each deletes, items T16 to T18, T28, T37 |
| [decisions/](decisions/README.md) | design decisions in force, items T19 to T21, answers T22, justifications |

## Assumptions

- Breaking changes are allowed, batched into one coordinated major release (TAIDL v3 and matching generator releases) with a migration note and no working aliases.
- T6 and T12 are breaking: v3 diagnostics carry a code and a clause id, and cases match on the code (Q3.9).
- Size: S is under one engineer-week, M is one to three engineer-weeks, L is one engineer-month or more.
- Quarters give the order; the roadmap track computes the load against three to four engineers over two quarters.
- Q1 holds T1 to T8, T10 to T12, T14, T16, T17, T26 to T30, T32, T35, T37, T38; Q2 holds T4, T9, T13, T15, T18, T23 to T25, T31, T33, T34, T36.
- Terms follow the owner's papers first, then StableHLO and MLIR.
- The user's file is the "ISA specification"; the language document is the "TAIDL language specification", cited as the language specification.
- Other terms: reject (generation, kernel compile) and fault (run time), element type, region, hbm, test oracle, compiler backend, assembly code, unsupported (coverage status).
- The language specification is the draft in `taidl/docs/` with sections S1 to S12; every corpus case and tag cites an S id.
- Accelerator scope is tensor accelerators plus GPU tensor-core instructions (mma, wgmma, TMA), with no SIMT model.
- TAIDL v3 has no undefined behavior: every run-time violation is a fault or a defined behavior under one norm, and the corpus states which per case.
- TAIDL assumes no hazard model; the ISA author declares overlap behavior.
- Overlap: each same-buffer input-output pair has one relation (tied, apart, exact-or-apart by default, or may overlap), and a permitted overlap reads the state before the issue.
- Markers (`write` keywords `inplace`, `apart`, `no_partial_overlap`, `may_overlap`) expand to constraints, and a violated one is the fault of S1.3.5.
- Authoring surface (decision 41, Q11.1 to Q11.4): an ISA specification is a Python file whose instructions are decorated functions traced once; v3 has no string form, so every corpus case is written in the surface and a v2 file arrives through the converter (T37).
- Check function (decision 42): `taidl.check(acc)` returns a `Model`, collects every problem into one `Rejection`, and runs once before any generator; the T6 runner and T38 cases test it.
- Layers: widths, encodings, derived attributes, and generator metadata are optional; an attribute is `s64` by default, and the cases of T32 to T35 cover each layer.
- Failure policy (decision 40): a backend limit on a valid instruction is a recorded exclusion with a printed reason, a candidate failure is a recorded drop, an oracle stops only on invalid assembly or binary input, and T36 holds the rows.
- Tensor data never faults: integer divide and remainder by 0 and out-of-range `convert` take fixed values, so no fault case reads tensor data.
- Versions: taidl, taidl-to, act 3.0; act-backend 2.0.
- Releases follow the act 2.x structure: released pins only, no pre-release tags, no commit or branch pins in public CI.
- New test dependencies: `hypothesis` (T9, T24) and `mutmut` (T25); no StableHLO interpreter.
- Other tracks are cited by letter: L language specification, B backend, O oracle, A act umbrella, D diagnostics, U shared code.
- Dependencies are named by item id only; the id's file is found through the index tables.
