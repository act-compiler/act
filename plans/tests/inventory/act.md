# Inventory: act

The umbrella repo holds one test and one version-check script.

| file | what it tests | style | verdict | reason |
| --- | --- | --- | --- | --- |
| `tests/test_end_to_end.py` (1) | a 4x4 bf16 matmul through oracle and backend | e2e | keep, becomes the home of T14 | :69-101 is the only test that runs both generators on one ISA specification |
| `tests/check_versions.py` | every version pin names the same release | script | keep, extend (T16) | not a test; run by CI; expects taidl, taidl-to, act 3.0 and act-backend 2.0 |
| `tests/conftest.py` | the `--platform` option, CPU or GPU, that the end-to-end test reads | fixture | keep | ten lines; T14 keeps the option |
| `tests/pytest.sh` | runs the suite in the image | script | keep | the entry point of CI |

## Counts

| keep | reshape | merge | delete |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |
