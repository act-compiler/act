# Tests, CI, and tooling in this repo

tests/test_end_to_end.py runs one accelerator through both generators, compiles one 4x4 matmul, and simulates it.
This is the only automated check of the whole chain.
The plan widens it just enough to cover the glossary's features and keeps the runtime small.
Nothing was run in this meeting.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| A19 | Cover state variables and constraints in the end-to-end test | L, O, B, T | M | Q2 |
| A20 | Test the README usage snippet | A1 | S | Q1 |
| A21 | Run the current tutorial's solutions in CI | A10, A11 | M | Q2 |
| A22 | Decide the tutorials' lint scope | A11 | S | Q2 |

## A19 Cover state variables and constraints

- What: the accelerator in tests/test_end_to_end.py:25-62 has three instructions and one data model; no state variable, constraint, or in-place mark appears.
- Benefits: a spec feature that breaks one generator is caught from the top.
- Caveats: each added feature must be supported by both generators, otherwise the test fails for a documented unsupported feature (L glossary: unsupported feature).
- Fail points: runtime grows or a feature is unsupported in one generator; notice by CI duration and a clear failure message naming the section id.
- Order and dependencies: L lists features; O and B state which they support; T decides test layout (the T track may own this file).
- Verification: the test passes on amd64 and arm64; each added feature has its own small test function.

## A20 Test the README usage snippet

- What: README.md:36-45 shows `Accelerator('MyAccel')` and two generator calls with "# ... define ISA spec ..."; no test runs that exact flow.
- Benefits: the first thing a reader copies works.
- Caveats: the snippet is an outline, not runnable text; the test reuses the `accelerator` helper and checks only that the calls are importable.
- Fail points: the README and test drift; notice by a test that extracts the code block.
- Order and dependencies: after A1 so the snippet's words are final.
- Verification: the test fails if the imports in README.md change without the test.

## A21 Run the current tutorial's solutions in CI

- What: tutorials/pldi26/exercise*/solution files are not exercised by any workflow step (.github/workflows/test.yml); a copy and run of exercise1 and exercise3 solutions would catch API drift.
- Benefits: tutorial breakage shows up on the commit that causes it, not on the day of the event.
- Caveats: the run needs the generated oracle, which costs minutes; exercise2 uses large `.dat` data.
- Fail points: runtime; notice by CI duration, and keep to one run per pull request.
- Order and dependencies: after A10 (pin) and A11 (layout); O and B must keep generated names stable.
- Verification: a deliberate API rename in a scratch branch turns the step red.

## A22 Decide the tutorials' lint scope

- What: pyproject.toml excludes `tutorials` from ruff (`extend-exclude`), while `shfmt -d .` covers tutorial scripts (.github/workflows/test.yml:30).
- Benefits: one stated rule: tutorials are exempt from ruff, covered by shfmt and the ASCII check.
- Caveats: the owner's rule says formatting is enforced by `format/`, which this repo lacks (no format directory; .editorconfig only).
- Fail points: two formatters disagree on a tutorial script; notice by a red CI step.
- Order and dependencies: with A11; no other repo.
- Verification: pyproject.toml and the workflow state the same scope in one sentence of README.md.
