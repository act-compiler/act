# Tutorial layout and content

The act repo holds three conference tutorials (micro25, asplos26, pldi26) as near-copies.
Past events are archived artifacts; pldi26 is the newest (tutorials/pldi26/README.md:33, 16 Jun 2026), but README.md does not name a current tutorial.
The plan removes the copies from master, fixes the current tutorial's terminology and API pin, and leaves the archived trees untouched until the layout is decided.
Nothing in tutorials/ has been edited.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| A9 | Name the current tutorial in README.md or retire the label | owner decision | S | Q1 |
| A10 | Check the current tutorial against the pinned image and API | B, O | M | Q1 |
| A11 | Replace the three copies by one tutorial plus per-event pages | A9, A10 | L | Q2 |
| A12 | Terminology and ASCII pass on the current tutorial | A1, L glossary | M | Q1 |
| A13 | Make exercise 4 runnable like exercises 1-3 | A11 | S | Q2 |
| A14 | Align paper titles and setup text | A1 | S | Q1 |

## A9 Name the current tutorial

- What: README.md never links a tutorial except "tutorials" in the repository list (README.md:99); today (Nov 2026) all three events are past.
- Benefits: a reader knows which tutorial to follow and which are archives.
- Caveats: "current" may mean "newest reusable"; the owner decides whether pldi26 stays labeled by event.
- Fail points: a README link to a path that A11 later moves; notice with a link check in review.
- Order and dependencies: before A11; independent of other tracks.
- Verification: README.md names one tutorial path that exists.

## A10 Check the current tutorial against the pinned image and API

- What: tutorials/pldi26/docker.sh:10-12 and setup.md:38-39 pin `devanshdvj/act:v1.2`; exercises import `act.taidl` and `act.generators` (exercise1/solution/QKV.py:3-4); README table pairs v1.2 with TAIDL v2.3 (README.md:73); no local tags exist, so the claim that v1.2 ships `act.generators` is unverified.
- Benefits: a factual error in the one tutorial still in use gets caught before attendees hit it.
- Caveats: needs Docker and network, unavailable in this meeting; micro25 and asplos26 use `from taidl import Accelerator` and `qkv.generate_oracle()` (micro25/exercise1/solution/QKV.py:3,108), which current taidl lacks (taidl/taidl/accelerator.py defines no `generate_oracle`).
- Fail points: attendees get ImportError; notice by running every exercise's solution inside the pinned image.
- Order and dependencies: B and O confirm the generated API names (`targets/QKV/...`, `backends/QKV`, `asm/`) at the image's versions; then A12.
- Verification: copy.sh, then python QKV.py, pytest on each exercise's solution pass in the image named by docker.sh.

## A11 Replace three copies by one tutorial plus event pages

- What: diffs show pldi26 and asplos26 differ in agenda, venue, setup.md, docker.sh, and API calls; micro25 adds no exercise 4; each tree carries its own figures/ (3 PNGs) and exercise2 data/*.dat.
- Proposal: master keeps `tutorials/` once (exercises, demos, figures, scripts, one docker.sh) with `tutorials/events/<event>.md` for agenda and logistics; past events stay on their branches (setup.md:21 clones `--branch pldi26`, micro25/setup.md:21 `--branch micro25`).
- Benefits: one place to fix a typo; the lint and ASCII gates run on one tree; repo size drops.
- Caveats: setup.md links `git clone --branch <event>`; each branch must exist and carry its tree; removing directories from master breaks any external link to `tutorials/micro25/`.
- Fail points: a missing branch or tag strands the archived tutorial; notice by `git ls-remote` for each event branch before deleting.
- Order and dependencies: after A9, A10; ruff config `extend-exclude = ["tutorials"]` (pyproject.toml) and shfmt scope may change; T track owns CI.
- Verification: each old URL either resolves on its branch or is redirected in README; CI green; `git ls-files tutorials` shows one exercise tree.

## A12 Terminology and ASCII pass on the current tutorial

- What: banned words per glossary in tutorials/pldi26: scratchpad (exercise1/README.md 12 hits, exercise2/README.md 12), HBM (exercise2/README.md 30), buffer, tile; sloppy words "Leveraging" (exercise2/README.md:688,702) and "stay tuned" (exercise4/README.md:63); symbols beta and alpha for attributes and `g_theta`, `CSP1-CSP4`, `get_h(i)` in exercise3 (section 3.1) without definition; non-ASCII in README.md:60-61, demos/*.md, setup.md.
- Benefits: the tutorial teaches the same words as the spec.
- Caveats: the archived trees keep their words; edit only pldi26. Code blocks in exercises mirror solution files, so doc and code edits must agree.
- Fail points: README text and QKV.py comments diverge; notice by diffing every code block with the solution file.
- Order and dependencies: after L glossary and A1; after A10 so the API is settled.
- Verification: grep for banned synonyms and non-ASCII over tutorials/pldi26 returns nothing (comment lines inside pre-existing solution files excepted if the owner says so).

## A13 Make exercise 4 runnable

- What: copy.sh usage lists only exercise1..3 (tutorials/pldi26/copy.sh:8); exercise4 has a README but no boilerplate or solution, and uses double-quoted strings.
- Benefits: the fourth exercise can be checked; the "four hands-on exercises" claim (tutorials/pldi26/README.md:19) holds.
- Caveats: needs a solution file; authoring work, not editing.
- Fail points: an unverified exercise ships; notice by running it in the image.
- Order and dependencies: after A11, A10; B confirms the backend accepts the modified ISA.
- Verification: solve.sh and copy.sh accept exercise4 and the oracle and backend generate.

## A14 Align paper titles and setup text

- What: tutorials/pldi26/setup.md:157 cites "ACT: Automatically Generating Compiler Backends ..." while README.md:153 lists "Automatically Generating ML Compiler Backends ..."; setup also names an arXiv link and USB tar `act-v1.2-<arch>.tar`.
- Benefits: one citation per paper.
- Caveats: the arXiv title may differ from the OOPSLA title; keep each venue's own.
- Fail points: a wrong link; notice by opening it.
- Order and dependencies: with A12.
- Verification: titles in README.md and setup.md match the published records.
