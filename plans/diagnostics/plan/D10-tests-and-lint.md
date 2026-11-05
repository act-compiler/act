# D10 Registry coverage test and wording lint

- What: a test fails when a registry code has no triggering test, no spec reference (`R`, `U`, `V`), or no row in the generated docs.
- What: tests assert `code` and `object`, not full text (E16); one golden test per class pins the three-line form.
- What: a lint rejects banned words in messages and diagnostic names ("refuse", "honor", "tile" for a region, "machine state", `buffer` in `taidl`) using `work/terminology/words-to-avoid.md`.
- Benefits: the scheme cannot drift; a new rejection without a code fails CI.
- Caveats: the lint shares its word list with L28; keep one list.
- Fail points: a trigger test that passes for another reason; the test asserts the code, so a different fault fails it.
- Order and dependencies: D1, T5, T6, L28.
- Verification: remove a code's test on a branch and see CI fail; add "refuse" to a message and see the lint fail.
