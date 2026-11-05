# act track findings

All fixes are plan items; nothing in /home/user/act was edited (applied = no).
A1..A22 refer to files under `plan/`.

## Old | new | where | applied?

| old | new | where | applied? |
| --- | --- | --- | --- |
| "PyTorch, and JAX" | "PyTorch and JAX" | README.md:9 | no (A2) |
| "we present **ACT (...) Ecosystem**, an ecosystem that" | "ACT (...) is an ecosystem that" | README.md:12 | no (A2) |
| "essential software tooling" | "software tools" | README.md:12 | no (A1) |
| "A Python-based DSL", "tensor IRs" | "language", "tensor intermediate representations" or the glossary term | README.md:16-17, tutorials/pldi26/README.md:13 | no (A1) |
| "TAIDL leverages ... compactly and precisely model" | "TAIDL models ... with XLA-HLO operations" | README.md:17 | no (A1) |
| "TAIDL-TO (Test Oracle) Generator", "Test Oracle generator", "Test Oracle Generator", "ACT-Backend Generator", "Compiler Backend Generator" | one name per generator | README.md:18,20,102-103,117-118 | no (A8) |
| "adding new tool generators", "New Tool Generators" | "adding new generators" | README.md:111,116 | no (A1) |
| "Top-level repository", "(top-level repository)", "unified `act` Python package" | one agreed name for this repo | README.md:99,116 | no (A1) |
| "the sub-projects' CI" | "the other repositories' CI" | README.md:51 | no (A1) |
| "# Citing ACT", "# Acknowledgements" (H1) | "##" | README.md:124,167 | no (A2) |
| "runs the two checks" (CI runs three steps) and an inline `ruff` command | one sentence per step; command in a fenced block | README.md:92-93, .github/workflows/test.yml:25-33 | no (A2) |
| New-release note and four old "ACT vX moves the ecosystem" lines (history) | delete or reduce to the newest line | README.md:2-4,77-81 | no (A3, owner decision) |
| "# ... define ISA spec ..." | "# ... build the spec ..." | README.md:41 | no (A1) |
| "from HBM", "to HBM" | "from off-chip memory" | tests/test_end_to_end.py:31,55 | no (A4) |
| "4x4 tiles of d1" | "4x4 regions of d1" | tests/test_end_to_end.py:43 | no (A4) |
| `spec` used for two unrelated module specs | `oracle_spec`, `kernel_spec` | tests/test_end_to_end.py:82-93 | no (A4) |
| `values`, `page`, `places`, `SCRIPTS`, `found` | names that say what they hold | tests/check_versions.py:11,18,23,69,104 | no (A4) |
| "re-exports from taidl package" | "the TAIDL front end, re-exported" | act/taidl/__init__.py:1 | no (A5) |
| ACT_ENV_TAG never compared | compare it with ACT_ENV_BUILD_TAG | docker/act-env.env:2, tests/check_versions.py:99 (reads only ACT_ENV_BUILD_TAG) | no (A15) |
| "scratchpad", "buffer", "HBM", "tile" | "data model", "off-chip memory", "region" | tutorials/pldi26/exercise1/README.md:23,78,180,313; exercise2/README.md (30 HBM, 12 scratchpad) | no (A12) |
| "Leveraging", "stay tuned" | plain statement | tutorials/pldi26/exercise2/README.md:688,702; exercise4/README.md:63 | no (A12) |
| non-ASCII: curly quotes, box drawing, arrows, emoji | ASCII | tutorials/*/README.md:60-61, demos/*.md:8-29, setup.md (last line) | no (A12; archived copies untouched) |
| copy.sh usage "{exercise1\|exercise2\|exercise3}" | exercise4 not copyable | tutorials/pldi26/copy.sh:8 | no (A13) |
| docker image `v1.2` in the current tutorial | image that carries the API the exercises use | tutorials/pldi26/docker.sh:10-12, setup.md:38-39 | no (A10) |
| paper title "ACT: Automatically Generating Compiler Backends..." | "Automatically Generating ML Compiler Backends..." as in the BibTeX | tutorials/pldi26/setup.md:157 vs README.md:153 | no (A14) |

Non-ASCII scan (`git ls-files | xargs grep -lP '[^\x00-\x7F]'`, text files): README.md, act/, tests/, docker/ are clean; only tutorials/ contain non-ASCII.

## Cross-repo observations

- Pins agree: taidl 2.15 (taidl/pyproject.toml:7), taidl-to 2.11 (act-oracle/pyproject.toml:7), act-backend 1.16 (act-backend/pyproject.toml:7), both generators pin taidl v2.15, act pins all three (act/pyproject.toml:15-17).
- Names for one generator differ per layer: repo `act-oracle`, distribution `taidl-to`, module `taidl_to`, function `generate_oracle` (act-oracle/taidl_to/generator.py:21); repo, distribution and module of the backend agree (`act-backend`, `act_backend`).
- This repo's distribution is `act-ecosystem` (pyproject.toml:2) while the import package and README call it `act` (README.md:99).
- Floors differ in kind: act-oracle uses `numpy>=2.0`, `nanobind>=2.8.0`, `jax==0.11.2`; act/requirements.yml pins `numpy=2.5.3`, `nanobind==2.8.0`, `jax==0.11.2`; compatible, but nothing checks the overlap.
- The current tutorial pins image v1.2 (docker.sh:10), which the README table pairs with TAIDL v2.3 (README.md:73); the exercises use `act.generators` (exercise1/solution/QKV.py:3-4); no local tags exist to confirm v1.2 ships it.
- The older tutorials use `from taidl import Accelerator` and `qkv.generate_oracle()` (micro25/exercise1/solution/QKV.py:3,108); current taidl has no such method (taidl/taidl/accelerator.py defines `add_data_model`, `add_initial_state`, `add_instruction`). That was correct for their release; not a factual error today only if the archived branch pins that release.
- README does not say which tutorial is current; pldi26 is the latest by date (pldi26/README.md:33, 16 Jun 2026), and every tutorial is now in the past.

## Tutorial layout proposal

- Measured: pldi26 vs asplos26 differ only in event text (README agenda, setup.md, docker.sh, image names) and the generator API calls; pldi26 vs micro25 also adds exercise4 and backend sections 3.1-3.2 (diff of exercise3/README.md).
- The three trees also hold three copies of figures/ (3 PNGs each) and exercise2 `data/*.dat`.
- Events already live on branches (`git clone --branch pldi26`, tutorials/pldi26/setup.md:21; `--branch micro25`, micro25/setup.md:21).
- Proposal: master keeps one `tutorials/` (exercises, demos, figures, copy/clean/solve scripts, one docker.sh) plus `tutorials/events/<event>.md` for agenda, venue, setup and the image pin; past events stay on their branches and tags, removed from master.
- Alternative if master must keep them: `tutorials/exercises/` shared and `tutorials/<event>/` holding only README, setup.md, docker.sh; needs the exercise code to run on one release, so micro25 and asplos26 would need the new API (not factual-only edits).
- Not done this quarter before the owner decides; see plan/tutorials.md.
