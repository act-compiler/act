# Open questions for the owner

Each question has a number, the items it unblocks, and a suggested default.
An unanswered question is planned at its default.

## T22. Open questions for the owner

- 1. Where does the shared test kit (T3) live.
  - Default: split, `taidl.testing` for the corpus loader and tolerances, `act/testkit/` for goldens and oracle import.
- 2. Should error messages carry the S9 or coverage.md clause id so T6 and T12 match on the id, not on prose.
  - Default: yes, as a suffix like `[S9.4]`, in the v3 release.
- 3. Is `hypothesis` an acceptable test-only dependency for T9.
  - Default: yes.
- 4. Do the generator repos keep a `mini_trn` smoke test after T14, or rely on act alone.
  - Default: keep one smoke test per repo.
- 5. Which real ISA descriptions beyond `mini_trn` and `mini_trn_double` go on the T10 list in Q1.
  - Default: one GPU tensor-core description (mma or wgmma with TMA loads) written by A, and no SIMT model.
- 6. Is the GPU platform a CI target for goldens, or reference-tolerance only.
  - Default: reference-tolerance only, as `act-oracle/tests/test_mini_trn.py:22` does today.
- 7. Does the slow differential tier (T13) run nightly in act, or on every act PR.
  - Default: nightly and on release.
- What: the list above, answered in one sitting.
- Benefits: unblocks T3, T6, T9, T12, T13, T14, T15.
- Caveats: defaults are the lead's; the owner may hold a standing preference the inventory did not see.
- Fail points: a question answered differently in two repos; the answer is recorded once in the act plan.
- Order and dependencies: before step 3 of the migration.
- Verification: every question has an answer or the word default beside it.
- Breaking: question 2 only.
- Files: none.
