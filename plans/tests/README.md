# Test suite track plan

Planning files for the next one to two quarters of test-suite work across taidl, act-backend, act-oracle, and act.
Every item carries a T id, a size, a quarter, a `Breaking` bullet, and the existing files it changes; the quarters assume three to four engineers on the track.
Nothing here was run against the repos; every claim cites a file and line read on the planning day.

## Files and folders

| entry | holds |
| --- | --- |
| [review.md](review.md) | what works, what is weak, verdict per test file, counts |
| [bugs.md](bugs.md) | test defects TB1 to TB22 with fix sketches |
| [overhaul.md](overhaul.md) | target layout, per-file and per-fixture action, ordered steps |
| [inventory/](inventory/README.md) | per-repo inventory tables, duplication and coupling, items T1 to T4 |
| [corpus/](corpus/README.md) | the conformance corpus, items T5 to T10, coverage of the outline |
| [generators.md](generators.md) | expectations, differential tests, end-to-end, goldens, items T11 to T15 |
| [migration.md](migration.md) | ordered PR steps and what each deletes, items T16 to T18 |
| [decisions/](decisions/README.md) | debate rounds, rulings T19 to T21, questions T22, justifications |

## Assumptions

- Breaking language or API changes are allowed, batched into one coordinated major release (TAIDL v3 and matching generator releases) with a migration note and no compatibility shims.
- Only T6 and T12 can break compatibility, and only if error messages gain clause ids (T22 question 2).
- Size: S is under one engineer-week, M is one to three engineer-weeks, L is one engineer-month or more.
- Q1 is sized at about twenty engineer-weeks of test work for three to four engineers, so T1 to T8, T10 to T12, T14, T16, and T17 fill it; T9, T13, T15, T4, and T18 are Q2.
- Terms follow the owner's papers first, then StableHLO and MLIR: ISA description (not spec, for a TAIDL program), reject (not refuse), element type (not dtype), region (not tile), test oracle, compiler backend, assembly code.
- The language spec is the draft in `taidl/docs/` with sections S1 to S10; every corpus case and tag cites an S id.
- Accelerator scope is tensor accelerators plus GPU tensor-core instructions (mma, wgmma, TMA), with no SIMT model.
- TAIDL v3 has no undefined behavior: every run-time violation is a defined error the oracle detects and reports, or a defined behavior under one norm; the corpus states which per case.
- An in-place mark means the output reuses the marked input's storage, and the corpus tests that aliasing in both generators.
- Other tracks are cited by letter: L language spec, B backend, O oracle, A act umbrella.
- Dependencies are named by item id only; the id's file is found through the index tables.
