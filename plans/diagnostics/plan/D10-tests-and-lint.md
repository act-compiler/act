# D10 Registry coverage test and wording lint

- What: a test fails when a registry code has no triggering test, no spec reference (`R`, `U`, `V`), or no row in the generated docs.
- What: tests assert `code` and `object`, not full text (E16); one text test per format rule and one golden two-line test per class pin the wording contract; expectation and corpus rows carry `code` and section id, with no regex.
- What: a lint rejects banned words in messages and diagnostic names ("refuse", "honor", "tile" for a region, "machine state", `buffer` in `taidl`) using `work/terminology/words-to-avoid.md`.
- Benefits: the scheme cannot drift; a new rejection without a code fails CI.
- What: D10 owns the banned-word list file; L28 consumes it.
- What: one `coverage.md` renderer, owned by T11, is the only renderer of that page; D11 keeps the S9 page in `taidl`, a different page.
- Fail points: a trigger test that passes for another reason; the test asserts the code, so a different fault fails it.
- Order and dependencies: D1, T5, T6.
- Verification: remove a code's test on a branch and see CI fail; add "refuse" to a message and see the lint fail.
