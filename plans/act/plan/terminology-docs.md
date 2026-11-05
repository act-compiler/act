# Terminology and docs cleanup (act repo)

The act repo's README, package docstrings, tests, and scripts must use the TAIDL glossary (/home/user/work/spec/glossary.md) and one name per generator.
Docs state what is: no release news, no roadmap, no history sentences.
Every item below is a plan item; no file in /home/user/act has been changed yet.
Item rows cite file:line from the tree as of this meeting.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| A1 | Apply glossary terms and plain words in README.md | L glossary, A8 | S | Q1 |
| A2 | Fix README structure, punctuation, and CI description | none | S | Q1 |
| A3 | Remove news, roadmap, and release-history lines from README | owner decision | S | Q1 |
| A4 | Rename vague identifiers and comments in tests/ | A1, T | S | Q1 |
| A5 | Fix act/ docstrings and the distribution name | A8 | S | Q1 |
| A6 | Comment pass on docker/ scripts | none | S | Q2 |
| A7 | ASCII and one-sentence-per-line gate for text files | A12 | S | Q1 |
| A8 | One name per generator across the four repos | L, O, B | M | Q1 |

## A1 Apply glossary terms in README.md

- What: replace "DSL" and "IRs" (README.md:16-17), "leverages" (:17), "essential software tooling" (:12), "tool generators" (:111,116), "sub-projects" (:51), "ISA spec" (:41) with the glossary terms or plain words; pick one name for this repo instead of "top-level repository" and "unified package" (:99,116).
- Benefits: the front page matches the spec and the other READMEs.
- Caveats: the glossary lacks entries for oracle, backend, kernel, ecosystem (see glossary_objections.md); wording waits on them.
- Fail points: a half-applied glossary leaves two terms for one idea; notice with `grep -n` for each banned synonym in README.md and tests/.
- Order and dependencies: after L publishes S10; after A8 for generator names; O and B READMEs get the same edit in their tracks.
- Verification: grep each banned synonym from the glossary table over README.md, act/, tests/ returns nothing; owner reads the diff.

## A2 Fix README structure, punctuation, and CI description

- What: "PyTorch, and JAX" to "PyTorch and JAX" (README.md:9); "we present ... Ecosystem, an ecosystem" to one clause (:12); H1 "Citing ACT" and "Acknowledgements" to H2 (:124,167); CI paragraph (:92-93) to one sentence per step (version check, format and lint, end-to-end) matching .github/workflows/test.yml:25-33, with the `ruff` command in a fenced block.
- Benefits: README follows the rules on sentences, headings, and fenced commands.
- Caveats: the CI sentence must change whenever the workflow steps change; keep it short so it does not drift.
- Fail points: an edit that breaks check_versions.py parsing of the table section; notice by running `python tests/check_versions.py` in the image (owner's run, not this meeting).
- Order and dependencies: independent; do before A1 to keep diffs separate (one concern per commit).
- Verification: re-read README.md:92-93 against the workflow; heading levels are H1 once.

## A3 Remove news, roadmap, and release history from README

- What: delete or reduce the NOTE (README.md:2-4), which announces v2.4 and unreleased `act-allo`, `act-oracle` v3, `act-e2e`; reduce the five "ACT vX moves the ecosystem" lines (:77-81) to the newest one.
- Benefits: the README states only what is; the compatibility table already holds each release's pins.
- Caveats: tests/check_versions.py requires a line opening with the newest ACT version below the table (tests/check_versions.py around the `opening` variable); older lines carry environment facts (Ubuntu 24.04, JAX 0.11.2 at :80) that live elsewhere in docker/act-env.env and requirements.yml.
- Fail points: deleting the newest sentence fails the version check; notice in CI step "Check the pinned versions".
- Order and dependencies: owner decides first (open question); then release process A16 decides whether the newest sentence stays.
- Verification: version check passes; no date, "New", or future-tense word remains in README.md.

## A4 Rename vague identifiers and comments in tests/

- What: "HBM" to "off-chip memory" and "tiles" to "regions" (tests/test_end_to_end.py:31,43,55); `spec` used twice for unrelated module specs to `oracle_spec` and `kernel_spec` (:82-93); in tests/check_versions.py rename `values`, `page`, `places`, `found`, `SCRIPTS` (:11,18,23,69,104) to names that state their content.
- Benefits: a reader never confuses a Python module spec with a TAIDL spec.
- Caveats: cannot run tests in this meeting; renames are local, so risk is low, but run the end-to-end test and version check after.
- Fail points: a missed use of a renamed local raises NameError; notice by `./tests/pytest.sh`.
- Order and dependencies: after A1 (terms); T track may rewrite the same file, so coordinate before editing test_end_to_end.py.
- Verification: pytest and check_versions pass in the image; ruff format and lint clean.

## A5 Fix act/ docstrings and the distribution name

- What: act/taidl/__init__.py:1 "re-exports from taidl package" to a glossary-based line (front end); the distribution is `act-ecosystem` (pyproject.toml:2) while README.md:99 calls it the `act` package.
- Benefits: one name for what pip installs and what Python imports, or an explicit explanation of both.
- Caveats: renaming the distribution changes `pip install` and `version('act-ecosystem')` in tests/check_versions.py; the Docker image and tutorials may reference it.
- Fail points: a stale reference breaks the version check; notice by grep for `act-ecosystem` over all four repos and the image build.
- Order and dependencies: after A8; default is to keep the distribution name and fix the README sentence instead.
- Verification: `grep -rn act-ecosystem /home/user/act*` matches only intended places; install and import work in the image.

## A6 Comment pass on docker/ scripts

- What: shorten multi-clause comments to one-liners in docker/chown.sh:15 ("Use busybox (tiny image) to perform chown"), docker/entrypoint.sh:4 ("Default to UID/GID 1000 if not provided"), docker/run.sh:57-63 ("Helper:" comments); Dockerfiles not reviewed this meeting.
- Benefits: scripts follow the comment doctrine; low risk, no behavior change.
- Caveats: scripts are not runnable here (Docker unavailable); edits are comments only; shfmt covers the whole tree in CI (.github/workflows/test.yml:30).
- Fail points: a comment edit that touches code; notice by `shfmt -d .` and a diff limited to comment lines.
- Order and dependencies: independent.
- Verification: diff shows only lines starting with `#`.

## A7 ASCII and one-sentence-per-line gate

- What: add the owner's check (`git ls-files | xargs grep -nP '[^\x00-\x7F]'`) as a CI step once tutorials are ASCII; today README.md, act/, tests/, docker/ are clean and only tutorials/ contain curly quotes, box-drawing, arrows, and emoji.
- Benefits: the rule is enforced, not remembered.
- Caveats: PNG and .dat files must be excluded; archived tutorials would fail unless removed from master (A11) or exempted.
- Fail points: grep treats binaries as matches; use `-I` and notice by a first CI run.
- Order and dependencies: after A11/A12; T track owns CI changes.
- Verification: the command prints nothing on master.

## A8 One name per generator across the four repos

- What: today the oracle generator is "TAIDL-TO (Test Oracle) Generator" (README.md:18), "Test Oracle generator" (:101), "Test Oracle Generator" (:117); repo `act-oracle`, distribution `taidl-to`, module `taidl_to`, function `generate_oracle`; the backend is "ACT-Backend Generator", "Compiler Backend Generator" (:20,118).
- Benefits: a reader finds one name in prose, repo, package, and function.
- Caveats: renaming a distribution or repo is a breaking release for pins (act/pyproject.toml:16 names `taidl-to` at `act-oracle.git`); prose-only harmonization is cheap, package renames are not.
- Fail points: pins and docs disagree after a partial rename; notice by check_versions.py and a grep of every old name over the four trees.
- Order and dependencies: L glossary adds the generator names; O and B agree on names; then A1 and A5.
- Verification: a table of old name to new name in each repo is empty of old names after the change.
