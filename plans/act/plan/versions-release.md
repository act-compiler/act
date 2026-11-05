# Version pins and release coordination across repos

The act repo pins taidl, act-oracle, and act-backend by git tag (pyproject.toml:15-17) and states every version in several places.
tests/check_versions.py compares those places; it does not check what sits outside this repo.
The plan makes a release a short, ordered procedure with the gaps in the check closed.
Observed today: the pins in act agree with the three sibling pyproject.toml files.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| A15 | Close gaps in check_versions.py | T | S | Q1 |
| A16 | Write the release order and a checklist | L, O, B | M | Q1 |
| A17 | Check sibling pins and dependency floors against act | A15, O, B | M | Q2 |
| A18 | Add a scheduled compatibility build | A16, T | M | Q2 |

## A15 Close gaps in check_versions.py

- What: the environment check compares docker/act-env.env `ACT_ENV_BUILD_TAG` with requirements.yml (tests/check_versions.py:99) but `ACT_ENV_TAG` (docker/act-env.env:2) is never read; both hold `py3.14-jax0.11.2` today.
- Benefits: the tag the ecosystem image builds on cannot drift from the environment that was built.
- Caveats: the Dockerfiles (docker/Dockerfile.amd64, .arm64) were not reviewed and may name versions the check does not see.
- Fail points: a drift passes CI silently; notice by editing one tag in a scratch copy and running the check (owner's run).
- Order and dependencies: T reviews check_versions.py coverage; then A4 renames identifiers in the same file.
- Verification: a deliberate mismatch in each place produces a failure line in the check output.

## A16 Write the release order and checklist

- What: one doc in this repo (or the README table's neighbor) listing: taidl tag, then act-oracle and act-backend tags pinning it, then act pins all three, README table row, sentence below the table, image push; today the order is implicit in pyproject.toml pins and the README table (README.md:68-75).
- Benefits: a release is a checklist, not a memory; reviewers know what changes in which repo.
- Caveats: the rules say docs hold no history; the checklist describes the procedure, never past releases.
- Fail points: a tag pushed before its dependency exists makes `pip install` fail; notice in CI step "Build the ecosystem image" (.github/workflows/test.yml:23-24).
- Order and dependencies: L fixes the spec version the language tag carries; O and B state their taidl pins (act-oracle/pyproject.toml:15, act-backend/pyproject.toml:16).
- Verification: dry-run the procedure for the next patch release in a branch; every place check_versions.py lists changes in one commit group per repo.

## A17 Check sibling pins and dependency floors against act

- What: current facts: taidl 2.15, taidl-to 2.11, act-backend 1.16 in each pyproject.toml; both generators pin taidl at v2.15; act/requirements.yml pins `numpy=2.5.3`, `nanobind==2.8.0`, `jax==0.11.2` while act-oracle declares `numpy>=2.0`, `nanobind>=2.8.0`, `jax==0.11.2`, `jaxlib==0.11.2`.
- Benefits: a sibling bump (for example jax) cannot silently break the image.
- Caveats: `pip check` (tests/check_versions.py) catches only conflicts among installed packages, not sibling repos at HEAD.
- Fail points: a sibling repo moves its taidl pin without a new act release; notice by comparing the three `taidl @ git+...@v` strings in one scripted check.
- Order and dependencies: O and B keep their pins stable during Q1; A15 first.
- Verification: the script prints the pin of each package per repo and fails on a mismatch.

## A18 Add a scheduled compatibility build

- What: CI runs on push and pull request to master only (.github/workflows/test.yml:3-7); add a weekly run so a moved dependency (jax, ortools, a base image) shows up without a commit.
- Benefits: breakage is found within a week, not at release time.
- Caveats: costs runner minutes on amd64 and arm64; the GPU path (`--platform GPU`, README.md:89) is not in CI.
- Fail points: flaky network failures produce noise; notice by tracking failure causes for four weeks.
- Order and dependencies: after A16 and T's CI changes; no other repo blocks it.
- Verification: the scheduled run is green for four consecutive weeks or each red run has a recorded cause.
