# Duplication and hidden coupling

What is tested twice across files and repos, and where fixtures couple to modules by name.
Each line cites the two or more places involved.

## Duplicated coverage

- The mini_trn attention golden is checked in `act-backend/tests/test_mini_trn.py:40-43` and `act-oracle/tests/test_mini_trn.py:22-35`, each with its own `regenerate.py`.
- Operation-word lists are derived three times: `taidl/tests/test_grammar.py:279` (`WORDS`), `act-backend/tests/test_refusals.py:146` (`grammar_words`), `act-oracle/tests/conftest.py:75` (`LOWERED`).
- Five helpers exist in both conftest files: `load_bf16_matrix` (backend :49, oracle :270), `import_oracle` (:162, :285), `load_golden` (:210, :336), `bf16` (:217, :344), `attention_reference` (:222, :349).
- Sum-from-a-start semantics are tested in `act-backend/tests/test_reader.py:280`, `test_matching.py:1177-1189`, and `test_constructs.py:569-575`.
- Formatter layout is tested by hand in `act-backend/tests/test_generators.py:175-393` and on every generated tree in `conftest.py:468-472`; rustfmt runs at generation.

## Hidden coupling in fixtures

- `act-backend/tests/conftest.py:463-476` `work_dir` reads `request.module.EXAMPLE_DIR`, a module constant the fixture finds by name.
- The same fixture asserts formatter cleanliness at :468-472, so a formatting failure fails every test of the module at setup.
- `assembly` and `output` at :487-495 chain on `work_dir` and `request.module.EXAMPLE_DIR`, so one module can test one example only.
- `act-oracle/tests/test_mini_trn.py:20` generates an oracle at import time, outside any fixture.
- `act-oracle/tests/conftest.py:75-155` `LOWERED` is a hand copy of the generator table; a drift fails `test_every_word_lowers` at `test_refusals.py:44` without saying which side is wrong.
- `act-backend/tests/conftest.py:25-39` implements an `incremental` marker that skips the rest of a class after one failure, so one failing setup hides the later results of `test_mini_trn.py:24-43`.
