# Version pins and release coordination across repos

The act repo pins taidl, act-oracle, and act-backend by released git tag (pyproject.toml:15-17) and states every version in several places.
tests/check_versions.py compares those places; it does not check what sits outside this repo.
The v3 release follows the act 2.x structure: each release pins released versions of its dependencies, with no pre-release tag and no commit or branch pin in public.
Development happens in private repos, where a pin may name a commit.

## v3 versions

| Package | Repo | Now | v3 |
| --- | --- | --- | --- |
| taidl | taidl | 2.15 | 3.0 |
| taidl-to | act-oracle | 2.11 | 3.0 |
| act-backend | act-backend | 1.16 | 2.0 |
| act-ecosystem (ACT) | act | 2.4 | 3.0 |
| Docker image | act | `devanshdvj/act:v2.4` | `devanshdvj/act:v3.0` |

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| A15 | Close gaps in check_versions.py | none | S | Q1 |
| A16 | Write the release order and a checklist | none | M | Q1 |
| A17 | Check sibling pins and dependency floors against act | A15 | M | Q2 |
| A18 | Add a scheduled compatibility build | A16 | M | after v3 |
| A23 | Release ACT 3.0 | A15, A16, A17, A20, A26, A28 | M | Q2 |
| A28 | Converter in the last 2.x release and act's own ISA specifications | A16 | M | Q1 |

## A15 Close gaps in check_versions.py

- What: the environment check compares docker/act-env.env `ACT_ENV_BUILD_TAG` with requirements.yml (tests/check_versions.py:99) but `ACT_ENV_TAG` (docker/act-env.env:2) is never read; both hold `py3.14-jax0.11.2` today.
- Benefits: the tag the ecosystem image builds on cannot drift from the environment that was built.
- Caveats: the Dockerfiles (docker/Dockerfile.amd64, .arm64) may name versions the check does not see.
- Fail points: a drift passes CI silently; notice by editing one tag in a scratch copy and running the check.
- Order and dependencies: A4 renames identifiers in the same file afterwards.
- Verification: a deliberate mismatch in each place produces a failure line in the check output.

## A16 Write the release order and checklist

- What: one doc in this repo listing the order of the act 2.x structure: taidl tag; then act-oracle and act-backend tags pinning that taidl tag; then act pins all three, the README table row, the sentence below the table, and the image push.
- Benefits: a release is a checklist, not a memory.
- Caveats: the doc describes the procedure and never past releases; it names released tags only.
- Fail points: a tag pushed before its dependency exists makes `pip install` fail; notice in CI step "Build the ecosystem image" (.github/workflows/test.yml:23-24).
- Order and dependencies: none; the generators' taidl pins (act-oracle/pyproject.toml:15, act-backend/pyproject.toml:16) follow the order.
- Verification: dry-run the order for the next patch release in a branch; every place check_versions.py lists changes in one commit group per repo.

## A17 Check sibling pins and dependency floors against act

- What: sub-item of A23; a scripted check that prints the `taidl @ git+...@v` pin of each package and the floors of numpy, nanobind, and jax (act/requirements.yml `numpy=2.5.3`, `nanobind==2.8.0`, `jax==0.11.2` against act-oracle's `numpy>=2.0`, `nanobind>=2.8.0`, `jax==0.11.2`, `jaxlib==0.11.2`).
- Benefits: a sibling bump (for example jax) cannot silently break the image.
- Caveats: `pip check` (tests/check_versions.py) catches only conflicts among installed packages.
- Fail points: a sibling repo moves its taidl pin without a new act release; notice by the script.
- Order and dependencies: after A15.
- Verification: the script fails on a deliberate mismatch.

## A18 Add a scheduled compatibility build

- What: CI runs on push and pull request to master only (.github/workflows/test.yml:3-7); add a weekly run on the released pins so a moved dependency (jax, ortools, a base image) shows up without a commit.
- Benefits: breakage is found within a week, not at release time.
- Caveats: costs runner minutes on amd64 and arm64; the GPU path (`--platform GPU`) is not in CI.
- Fail points: flaky network failures produce noise; notice by tracking failure causes for four weeks.
- Order and dependencies: after A16.
- Verification: the scheduled run is green for four consecutive weeks or each red run has a recorded cause.

## A23 Release ACT 3.0

- What: the act part of the v3 release train (U8), in the order of A16: the sibling releases taidl 3.0, taidl-to 3.0, and act-backend 2.0 exist as released tags first; then act changes in one commit group.
- The group sets the three pins in pyproject.toml to `v3.0`, `v3.0`, and `v2.0`, and `version` to `3.0`.
- It adds the README table row (ACT v3.0, TAIDL v3.0, TAIDL-TO v3.0, ACT-Backend v2.0, image `devanshdvj/act:v3.0`) and the one sentence "ACT v3.0 moves the ecosystem to ...", and removes the older sentences (A3).
- It adds the migration note: every rename and removal of the v3 batch, generated from the rename table, plus the `targets/` to `accelerators/` move, the removal of the string form, and the converter command of A28.
- It pushes the image `v3.0` after the quickstart gate (A20) and the fresh-venv gate (A26) pass.
- Benefits: act is the only repo that pins all three siblings, so the train cannot finish without this item.
- Caveats: no pre-release tag and no commit or branch pin appears in public; the siblings' private development pins stay private.
- Fail points: check_versions.py fails when the newest row, the pins, and the sentence disagree; notice in CI step "Check the pinned versions".
- Order and dependencies: after A15, A16, A17, A20, A26, and A28; U8 and the tutorial (A24) follow it.
- Verification: `python tests/check_versions.py` passes in the v3 image; `pip install` of the tag works in a fresh venv.

## A28 Converter in the last 2.x release and act's own ISA specifications

- What: the converter of Q11.4 is a script on the taidl 2.x line, run under taidl 2.x; it reads loaded v2 `Instruction` objects and prints the authoring surface; it is not a `taidl` subcommand in v3.
- What: the last 2.x release of taidl (the act 2.x structure of A16) carries the script and its README usage line; the v3 removed-names rejection of `add_semantics` names that release and the one command to run.
- What: act runs the script over `tests/test_end_to_end.py` and the quickstart outline, commits the converted files, and gates each with the equivalence test of T37 (`Model` from the string form equals `Model` traced from the converted file, up to call sites).
- What: a file the script cannot convert is rewritten by hand from the HLO-text printer's output; the migration note lists the construct and the hand route.
- What: the released converter pins only released versions, as every public release does.
- Benefits: an architect with a v2 ISA specification has one command and a proof of equality, and act's own files move without a hand rewrite.
- Caveats: no outside owner of a v2 specification is known, so the script carries no support promise beyond the migration note; a release can wrap it later if one appears.
- Fail points: a converted file that builds a different `Model`; the T37 test fails on that file.
- Order and dependencies: after A16 (the release order); the script is the shared track's, the `Model` equality is T37, and A23 and A24 wait for it.
- Verification: the converted `tests/test_end_to_end.py` passes in the v3 image; the migration note names the converter's release tag.
