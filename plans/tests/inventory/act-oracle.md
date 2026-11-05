# Inventory: act-oracle

The test files of the act-oracle repo, 148 tests, with a verdict each.
Style values are those of the inventory key.

| file | what it tests | style | verdict | reason |
| --- | --- | --- | --- | --- |
| `tests/test_kernel.py` (60, 1744 lines) | the kernel API: modes, loops, debug, declarations, regions | cli, spec | keep, split by theme (T17) | :46-1091 is the oracle's own API contract; :1287-1728 (looped attributes, region shapes) is S5.2 material for T7 and T29; the seeds at :590-598 and :1229-1240 become `program.toml` cases |
| `tests/test_refusals.py` (12) | ISA specifications the oracle rejects | spec, impl | reshape into T11 and T12 | :44 `test_every_word_lowers` and :67 depend on the `LOWERED` copy in `conftest.py:75`; :50 deletes from `oracle_visitor.TYPES` |
| `tests/test_templates.py` (19) | one instruction at a time | pr, spec | merge into T8 | mixed bag; :510 holds no assertion |
| `tests/test_elementwise.py` (1), `test_reduce.py` (4), `test_structural.py` (14), `test_compare_select_clamp.py` (4), `test_linalg.py` (5), `test_constant_iota.py` (10) | each tensor operator at each element type against NumPy | spec | keep, tag S8.3 and S8.4, extend in T9; tests of the removed words become `rejected` cases (T7) | the only reference-value tests in the four repos; `test_constant_iota.py:107-157` assert inside conftest helpers only |
| `tests/test_constraints.py` (5), `tests/test_state.py` (5) | constraints checked per call, state updates in order | spec | keep, tag S5.3 and S5.4 | small and direct |
| `tests/test_examples.py` (2) | examples generate side by side, no bytecode cache | pkg | keep | packaging contract |
| `tests/test_mini_trn.py` (5), `tests/test_mini_trn_double.py` (2) | attention and other kernels against golden and reference | e2e, golden | merge into T14 | generation runs at import at :20; `skipif(PLATFORM == 'GPU')` at :22 |
| `tests/conftest.py` (364 lines) | element type tables, `LOWERED`, tolerances, helpers | - | reshape (T2, T3, T11) |
| `tests/regenerate.py`, `tests/regenerate.sh` | rewrite an example's goldens from its oracle | script | keep | each golden comes from the first kernel the example lists |
| `tests/pytest.sh` | runs the suite in the image | script | keep | the entry point of CI | `LOWERED` :75 is a hand copy of the generator's table and the natural seed of coverage.md; `ULPS` :227 is an undocumented tolerance policy |

## Counts

| keep | reshape | merge | delete |
| --- | --- | --- | --- |
| 110 | 12 | 26 | 0 |
