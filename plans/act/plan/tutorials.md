# Tutorials

The act repo holds three conference tutorials (micro25, asplos26, pldi26) as near-copies, and none is current: pldi26 is the latest and dates from about ACT 1.2.
The plan archives all three on their tags, removes them from master, and writes one tutorial on the v3 release.
Nothing in tutorials/ has been edited.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| A9 | Retire the "current tutorial" label | none | S | Q1 |
| A10 | List the tutorial syntax that fails on the v3 API | none | M | Q1 |
| A11 | Archive all three tutorials on their tags and remove them from master | A10 | M | Q1 |
| A12 | Glossary and ASCII pass on the new tutorial | A24 | M | Q2 |
| A13 | Give every exercise of the new tutorial a runnable solution | A24 | S | Q2 |
| A14 | Align paper titles and setup text in the new tutorial | A24 | S | Q2 |
| A24 | Write one tutorial on the v3 release | A10, A11, A23, A25 | L | Q2 |

## A9 Retire the "current tutorial" label

- What: README.md names no tutorial and links none until A24 lands; the only mention is "tutorials" in the repository list (README.md:99).
- Benefits: a reader is never sent to an archive as if it were current.
- Caveats: the README link to the new tutorial waits for A24.
- Fail points: a README link to a path that A11 moves; notice with a link check in review.
- Order and dependencies: independent.
- Verification: README.md links no tutorial path that A11 removes.

## A10 List the tutorial syntax that fails on the v3 API

- What: the archived trees call the pre-v3 API: `from taidl import Accelerator` and `qkv.generate_oracle()` (micro25/exercise1/solution/QKV.py:3,108) while the entry is the free function `generate_oracle(accelerator, base_dir)`; pldi26 pins image v1.2 (docker.sh:10-12).
- Benefits: the list feeds the A23 migration note and the A24 exercises.
- Caveats: needs Docker and network; the list names syntax that fails, not a fix.
- Fail points: a missed call stays in the new tutorial; notice by running every solution on the v3 release.
- Order and dependencies: none; the v3 renames of the taidl docs (L26) fix the new names.
- Verification: each archived solution either runs on the v3 release or has a row in the list.

## A11 Archive all three tutorials

- What: tag the pre-removal tree of micro25, asplos26, and pldi26 (`git clone --branch <event>` already names the branch, pldi26/setup.md:21); then remove `tutorials/` from master.
- Benefits: the lint and ASCII gates run on the live tree only; 57 archived files hold non-ASCII text, and the trees stay untouched on their tags.
- Caveats: external links to `tutorials/micro25/` break; the README says where the archives live.
- Fail points: a missing tag strands an archive; notice by `git ls-remote` for each tag before deleting.
- Order and dependencies: after A10; ruff `extend-exclude = ["tutorials"]` (pyproject.toml) and the shfmt scope change with A22.
- Verification: each tag resolves and carries its tree; CI is green; `git ls-files tutorials` lists nothing until A24.

## A24 Write one tutorial on the v3 release

- What: one exercise tree on the released v3 pins (act 3.0, image `v3.0`), with the layout chosen inside this item; the default is one tree plus `tutorials/events/<event>.md` for agenda and logistics.
- Benefits: one place to fix a typo; the exercises teach the glossary terms (buffer, state, hbm, region, ISA specification).
- Caveats: the exercises mirror their solution files, so text and code change together; the old figures and `data/*.dat` come from the tags.
- Fail points: an exercise ships unverified; notice by running it in the v3 image (A21).
- Order and dependencies: after A10, A11, A23, and A25; the migration note of A23 lists the renames the exercises use.
- Verification: copy.sh and solve.sh accept every exercise; the quickstart (A25) and the tutorial use the same accelerator vocabulary.

## A12 Glossary and ASCII pass on the new tutorial

- What: no word of plans/glossary/words-to-avoid.md and no non-ASCII character in the new tutorial; define every symbol (alpha, beta, `g_theta`, `get_h(i)`) at first use.
- Benefits: the tutorial teaches the same words as the language specification.
- Caveats: code blocks mirror the solution files; diff each block against its file.
- Fail points: README text and QKV.py comments diverge; notice by the diff.
- Order and dependencies: after A24.
- Verification: the banned-word grep and the ASCII check over the new tutorial print nothing.

## A13 Give every exercise a runnable solution

- What: every exercise has boilerplate, a solution, and an entry in copy.sh and solve.sh (pldi26's exercise 4 had a README only, copy.sh:8).
- Benefits: the claim "four hands-on exercises" holds.
- Caveats: authoring work; the backend must accept the exercise's ISA specification.
- Fail points: an unverified exercise ships; notice by running it in the image.
- Order and dependencies: after A24.
- Verification: solve.sh passes and both generators run for each exercise.

## A14 Align paper titles and setup text

- What: one title per paper in the new tutorial's README and setup.md (the archived setup.md cites "ACT: Automatically Generating Compiler Backends ..." and README.md "Automatically Generating ML Compiler Backends ...").
- Benefits: one citation per paper.
- Caveats: keep each venue's own title.
- Fail points: a wrong link; notice by opening it.
- Order and dependencies: with A12.
- Verification: titles match the published records.
