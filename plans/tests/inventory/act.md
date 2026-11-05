# Inventory: act

The umbrella repo holds one test and one version-check script.

| file | what it tests | style | verdict | reason |
| --- | --- | --- | --- | --- |
| `tests/test_end_to_end.py` (1) | a 4x4 bf16 matmul through oracle and backend | e2e | keep, becomes the home of T14 | :69-101 is the only test that runs both generators on one ISA description |
| `tests/check_versions.py` | every version pin names the same release | script | keep | not a test; run by CI |

## Counts

| keep | rewrite | merge | delete |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |
