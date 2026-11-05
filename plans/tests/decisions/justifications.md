# Delete-and-rewrite justifications

Every removal that is not a merge, with why nothing of value is lost.
Everything else is a merge whose PR names the corpus case, row, or differential case that carries the fact on.

- `act-backend/tests/test_generators.py:175-393` layout tests: deleted, not rewritten, because rustfmt runs at generation and `conftest.py:468-472` checks every generated tree; the one test there that shows a mark reaching the e-graph (`:243`) stays as a per-PR impl test, since the markers stay.
- `act-backend/tests/test_refusals.py:164-176` and :200-205 egraph regex tests: deleted after T11 and T12 land and the slow tier runs per pull request, because the invariant they guard (generator words equal reader words) is then checked by compiling each word.
- `act-backend/tests/test_mini_trn.py:68-78` and `test_mini_trn_double.py:70-80`: deleted after T14, because the tied-pair fact is checked by the T31 tied row on the assembly, while the Rust text pin at :72 is not.
- `act-backend/tests/test_mini_trn.py:49-56` call counts: deleted after T13, because a cheaper correct selection must not fail the suite; the reference and golden checks remain.
- `act-backend/tests/conftest.py:25-39` `incremental` marker: deleted with T14, because it hides later results after one failure and the chain it served moves to fixtures.
- `act-oracle/tests/conftest.py:75-155` `LOWERED`: rewritten as `expectations.toml` and a corpus runner over S8 words, because a hand copy of a generator table cannot be the source of truth for coverage.md.
