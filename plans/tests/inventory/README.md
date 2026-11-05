# Inventory

Per repo, per test file: what it tests, its style, and a verdict with the lines that justify it.
Line numbers are from the trees as read on the planning day and move with other tracks' edits.

## Files

| file | holds |
| --- | --- |
| [taidl.md](taidl.md) | the five taidl test files |
| [act-backend.md](act-backend.md) | the act-backend test files and conftest |
| [act-oracle.md](act-oracle.md) | the act-oracle test files and conftest |
| [act.md](act.md) | the act end-to-end test and version check |
| [coupling.md](coupling.md) | duplicated coverage across repos and hidden fixture coupling |
| [cleanup.md](cleanup.md) | items T1 to T4 |

## Style key

- `spec`: checks a rule the language specification states; survives any implementation change.
- `pr`: regression for one PR; a bespoke ISA specification or kernel exercising one fix.
- `impl`: pins an implementation detail (internal table, log format, generated text, formatter layout).
- `e2e`: generates, compiles, and simulates a real example.
- `golden`: compares to a stored output file.
- `cli` or `pkg`: contract of the command line, the build, or the wheel; neither spec nor implementation.
