# Tests, CI, and tooling in this repo

tests/test_end_to_end.py runs one accelerator through both generators, compiles one 4x4 matmul, and simulates it.
This is the only automated check of the whole chain.
The plan widens it just enough to cover the glossary's features, runs the quickstart as the release gate, and keeps the runtime small.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| A19 | Cover state variables and constraints in the end-to-end test | none | M | Q2 |
| A20 | Run the quickstart in the image | A25 | S | Q1 |
| A21 | Run the new tutorial's solutions in CI | A24 | M | Q2 |
| A22 | Decide the tutorial lint scope | A24 | S | Q2 |
| A26 | Gate the tag with a fresh-venv install | A25 | S | Q2 |

## A19 Cover state variables and constraints

- What: the accelerator in tests/test_end_to_end.py:25-62 has three instructions and one data model; no state variable or constraint appears; add one instruction per relation (tied, apart, exact-or-apart, may overlap) through its marker.
- Benefits: a spec feature that breaks one generator is caught from the top.
- Caveats: each added feature must be supported by both generators, otherwise the test fails for a documented unsupported feature, which a generator rejects with a coded error.
- Fail points: runtime grows or a feature is unsupported in one generator; notice by CI duration and a clear failure message naming the section id.
- Order and dependencies: the coverage tables of both generators say which features they support; the test-suite items decide layout and may own this file.
- Verification: the test passes on amd64 and arm64; each added feature has its own small test function.

## A20 Run the quickstart in the image

- What: the quickstart example file of A25 runs inside the built image, calling both generators on the accelerator it defines; the run is the release gate for A23.
- Benefits: the first thing a reader copies works on every tag.
- Caveats: the run costs generator time; the example reuses the structure of the `accelerator` helper in tests/test_end_to_end.py.
- Fail points: README and example drift; notice by a test that compares the README block with the example file.
- Order and dependencies: after A25.
- Verification: the test fails when the example or the README block changes without the other.

## A21 Run the new tutorial's solutions in CI

- What: the solution files of the new tutorial (A24) run in a workflow step (.github/workflows/test.yml); a copy and run of the first and last exercise would catch API drift.
- Benefits: tutorial breakage shows up on the commit that causes it, not on the day of the event.
- Caveats: the run needs the generated oracle, which costs minutes; an exercise with large `.dat` data stays out of the per-pull-request run.
- Fail points: runtime; notice by CI duration, and keep to one run per pull request.
- Order and dependencies: after A24; the generators keep generated names stable.
- Verification: a deliberate API rename in a scratch branch turns the step red.

## A22 Decide the tutorial lint scope

- What: pyproject.toml excludes `tutorials` from ruff (`extend-exclude`), while `shfmt -d .` covers tutorial scripts (.github/workflows/test.yml:30); state one scope for the new tutorial.
- Benefits: one stated rule, for example ruff and shfmt both cover the tutorial and the ASCII check covers its text.
- Caveats: this repo has no `format/` directory (.editorconfig only).
- Fail points: two formatters disagree on a script; notice by a red CI step.
- Order and dependencies: after A24.
- Verification: pyproject.toml and the workflow state the same scope in one sentence of README.md.

## A26 Gate the tag with a fresh-venv install

- What: a CI job per tag installs the tag into a fresh venv on the stock runner and runs the front end and oracle half of the quickstart; the backend half needs cargo, CMake, a C++17 compiler, OR-Tools, and nlohmann/json, so it runs only in the image (A20).
- Benefits: the README's first `pip install git+...@v3.0` line is proven on every tag.
- Caveats: no wheels in v3; the README points at act-backend/README.md for the backend toolchain.
- Fail points: a pin that resolves in the image and not on a stock runner; notice by this job.
- Order and dependencies: after A25.
- Verification: the job is green on the v3.0 tag and red on a tag whose pin names a missing release.
