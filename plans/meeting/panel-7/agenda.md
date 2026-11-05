# Panel 7 agenda: vocabulary and usability

Scope: glossary questions left open within the owner decisions, the act repo, tutorials, releases, and use by a newcomer.
Questions are ordered blockers first, then majors.
Options are labeled A, B, C; a draft answer is one option among them and is named by its ruling ID.
Owner decisions 6, 14, 19, 20, and 21 are binding and are not reopened.

Source shorthand:

- AGn: finding n in plans/reviews/act-glossary/review.md.
- Rn: draft answer n in plans/minutes/rulings.md.
- RAn: unsettled question n in plans/reviews/rulings/audit.md.
- An: plan item n in plans/act/plan/.
- Decision n: row n of plans/minutes/owner-decisions.md.

### Q7.1: What happens to the three tutorials and to plan items A9-A14, given that no tutorial is current?

- Context: Decision 6 says there is no current tutorial; `pldi26` is the latest and dates from about ACT 1.2.
- Context: A9 asks which tutorial is current, A10 checks "the current tutorial" at image v1.2, and A12 edits pldi26 terms; all three presuppose a current tutorial (AG13, blocker).
- Context: v3 renames break any pre-v3 tutorial again, and the tutorials use syntax the current grammar rejects (R63).
- Option A: mark all three tutorials archived; write one new tutorial on the v3 release candidate in Q2; fold A12 and A14 into it (AG13 proposal).
- Option B: refresh one tutorial against current ACT in A10, then merge the three copies in A11 (R50 as drafted).
- Option C: master keeps one `tutorials/` tree plus `tutorials/events/<event>.md` pages; past events stay on their branches and tags (plans/act/findings.md layout proposal, A11).
- Option D: master keeps a shared `tutorials/exercises/` and per-event folders holding only README, setup.md, and docker.sh; needs the exercise code to run on one release (plans/act/findings.md alternative).
- Option E: leave the three trees untouched and write no tutorial for v3.
- Sub-question R54: archived tutorials contain non-ASCII (curly quotes, box drawing, arrows, emoji).
- Sub-option R54-a: exempt archived tutorials from the ASCII rule until replaced (R54).
- Sub-option R54-b: keep archived tutorials out of the checked tree (audit R54 row).
- Sub-option R54-c: ask the owner to amend the ASCII rule, which allows non-ASCII in binary assets only (audit R54 row).
- Evidence: plans/act/plan/tutorials.md (A9-A14); plans/act/findings.md rows for tutorials and "Tutorial layout proposal"; plans/act/plan/tests-docker.md A21, A22; plans/reviews/act-glossary/review.md AG13, AG20; plans/reviews/rulings/audit.md R54; plans/minutes/rulings.md R50, R54, R63.
- Evidence: tutorials/pldi26/docker.sh:10-12 pins image v1.2; tutorials/micro25/exercise1/solution/QKV.py:3,108 uses `generate_oracle()` as a method that current taidl lacks.
- Binding: Decision 6 (no current tutorial); Decision 10 (breaking changes batched into one major release); Decision 5 (no edits outside plans/ during the meeting).

### Q7.2: How is one vocabulary kept, and what is its authority over the competing glossary sources?

- Context: glossary/terms.md, rulings.md, and words-to-avoid.md still rule "tensor buffer", "control register", "global memory", "data slice", and their identifiers; the README override block leaves two rules per concept for at least eight concepts (AG1, blocker).
- Context: terminology/conflicts.md, terminology/words-to-avoid.md, the spec-draft glossary, backend/naming.md, oracle/plan/naming.md, and act/glossary_objections.md contradict the reconciled glossary (AG2).
- Context: some pointers lead to /home/user/work/, a second tree that already differs from plans/ (AG19).
- Option A: apply the override table to the three glossary files, delete the README override block, mark terminology/ as source comparison only, delete its conflicts and words-to-avoid files, regenerate the spec glossary from terms.md, and delete resolved objection files (AG1, AG2 proposals).
- Option B: keep both folders as living documents and add a "supersedes" line to each conflicting row.
- Option C: keep terms.md as the only glossary; generate words-to-avoid.md from its banned column; delete resolved rows of rulings.md (AG20 proposal).
- Option D: collapse glossary/ and terminology/ into one folder with one table per concept group.
- Option E: apply the override table only to terms.md and leave the other files as history.
- Evidence: plans/glossary/README.md (override list); plans/glossary/terms.md; plans/glossary/rulings.md; plans/glossary/words-to-avoid.md; plans/terminology/conflicts.md; plans/terminology/words-to-avoid.md; plans/act/glossary_objections.md; plans/reviews/act-glossary/review.md (AG1, AG2, AG19, AG20 and the second table).
- Binding: Decision 14 (the owner's papers first when precise, otherwise StableHLO and MLIR); Decisions 19, 20, 21 override the glossary rows they touch.

### Q7.3: Does the act repo get its own v3 release item, and what are the v3 version numbers?

- Context: the v3 release train (S8) depends on act items, but no A item re-pins the three siblings, adds the v3 row and image tag, updates the README snippet, or writes the migration note (AG14).
- Context: A16 and S8 both define the release order; A17 and S8 both extend check_versions.py; the version numbers of the v3 trio are undefined (AG17).
- Context: pins today are taidl 2.15, taidl-to 2.11, act-backend 1.16, with act pinning all three (plans/act/findings.md cross-repo observations).
- Option A: add item A23 "act v3 release" (pins, README row, image, migration note) in Q2 after S8's front-end tag; merge A16 and A17 into S8 and A23; state v3 numbers (AG14, AG17 proposals).
- Option B: keep A16-A18 as separate act items and add a single cross-reference to S8.
- Option C: S8 owns the whole release train including the act repo steps; the act plan keeps only the check_versions.py changes (A15).
- Option D: version the three generators and act independently of the TAIDL major number.
- Evidence: plans/act/plan/versions-release.md (A15-A18); plans/reviews/act-glossary/review.md AG14, AG17; plans/act/findings.md (ACT_ENV_TAG row, pins).
- Binding: Decision 10 (one coordinated major release: TAIDL v3 and matching generator releases); Decision 11 (3-4 engineers, two quarters).

### Q7.4: What is the supported install path, and is an install from a pinned release tested?

- Context: CI builds the Docker image only; "install from a pinned release" is never tested.
- Context: `pip install git+...@v2.4` needs Python 3.14, git, a Rust toolchain, ANTLR, and OR-Tools on a plain host, and none of this is stated or checked (AG15).
- Option A: add a release gate that installs the tag into a fresh venv on the stock runner and then runs the quickstart; document the system requirements (AG15 proposal).
- Option B: declare Docker the only supported path; document it and test only the image.
- Option C: support both; test the pip path on a schedule (A18) rather than as a release gate.
- Option D: ship prebuilt wheels so that no Rust toolchain or ANTLR is required on the host.
- Evidence: plans/reviews/act-glossary/review.md AG15; plans/act/plan/versions-release.md A16, A18; .github/workflows/test.yml:23-33; README.md install section; docker/ scripts.
- Binding: Decision 10; Decision 12 and 13 (no issues, no pull request); no dependency may be added without asking (engineering.md).

### Q7.5: What is the newcomer entry point of the act repo?

- Context: README.md has no single entry point; the usage snippet is an outline ("# ... define ISA spec ...") and nothing links the spec, the coverage tables, or a tutorial (AG16).
- Context: A1-A3 reword the page; A3 removes news and history lines, with the owner to decide first; A20 tests the README snippet; R51 treats the README trims as plan items executed with owner sign-off.
- Option A: a complete 15-line accelerator quickstart run by A20 (not only imported), linking the spec, coverage tables, and tutorial, with Docker and testing detail moved to `docs/` (AG16 proposal).
- Option B: keep the outline snippet; the README links to a tutorial as the first example.
- Option C: the quickstart lives in the spec or the taidl repo; the act README only lists repos, versions, and install.
- Option D: README states only what is (A1-A3) and adds no quickstart in v3.
- Sub-question A3: delete the news note and the four old "ACT vX moves the ecosystem" lines, or reduce them to the newest line, which tests/check_versions.py requires below the table.
- Evidence: plans/act/plan/terminology-docs.md A1-A3; plans/act/plan/tests-docker.md A20; README.md:36-45, 77-81; tests/check_versions.py; plans/reviews/act-glossary/review.md AG16; plans/minutes/rulings.md R51.
- Binding: Decision 5 (no edits outside plans/); Decision 6 (no current tutorial, so no link to one as current).

### Q7.6: Which API names add buffers and state, and what is the singular of "state"?

- Context: Decision 19 says a data model has buffers and state, but the glossary README fixes no method names, and no singular for one state entry (AG3).
- Context: R14 adds typed state kinds (descriptors, async groups), so "a named integer" may be too narrow; R27 still says "State registers" (AG8).
- Option A: `add_buffer(name, access_dims, base_element_dims, element_type)` and `add_state(name, value)`; keep `State` and `@s.`; unban "state variable" as the singular (AG3 proposal).
- Option B: same method names; the singular is "state entry" or "state" only, and "state variable" stays banned.
- Option C: `Accelerator.data_model` becomes an object holding `.buffers` and `.state`; methods hang off it.
- Option D: keep a single `add_data_model`-style entry point with a kind argument for buffer or state.
- Sub-question: define state as "named values of an integer or declared kind" to cover R14 (AG8), or keep "named integer" until typed kinds land.
- Evidence: plans/glossary/README.md line 7; plans/glossary/terms.md rows "tensor buffer", "control register", "initial value"; taidl/taidl/accelerator.py:33,44,46; taidl/taidl/data_model.py:28,33; plans/reviews/act-glossary/review.md AG3, AG8 and override table; plans/minutes/rulings.md R14, R27.
- Binding: Decision 19 (buffers, not "tensor buffers"; state, not "initial state"); Decision 21 (`@s.` stays); Decision 14.

### Q7.7: What does prose call the built-in buffer `d0`, now that `hbm` stays `hbm`?

- Context: Decision 19 keeps `hbm`, but the glossary renames it to `global_memory` and bans HBM; A4 and findings.md rewrite "HBM" to "off-chip memory", which the glossary bans in favor of "global memory" (AG4).
- Context: the kernel parameter is `hbm` (decorator.py:142) and the Rust variant is `Buffer::HBM`; the papers call `d0` "global memory" and use HBM as an alias; MLIR gpu and CUDA say global memory.
- Option A: the term is "hbm" (the built-in buffer `d0`, sized by the `hbm` kernel argument); ban "global memory", "off-chip memory", "DRAM", "main memory"; drop every HBM rename (AG4 proposal).
- Option B: identifier `hbm`; prose "global memory" (the papers' term), with HBM named as the identifier's spelling.
- Option C: identifier `hbm`; prose "off-chip memory" (the spec-draft glossary term), HBM only for a concrete chip.
- Option D: allow "hbm", "global memory", and "HBM" as synonyms, each with a stated context.
- Sub-question: for the GPU scope, state whether "device memory" is `hbm` and whether shared memory and register fragments are buffers.
- Evidence: plans/glossary/terms.md row "global memory"; plans/glossary/rulings.md `d0` row; plans/terminology/description-and-memory.md row "Built-in off-chip memory"; plans/act/findings.md HBM rows; plans/act/plan/terminology-docs.md A4; act-oracle/taidl_to/generic/decorator.py:142; act-backend/act_backend/instruction_info.py:71; plans/reviews/act-glossary/review.md AG4.
- Binding: Decision 19 (`hbm` stays `hbm`); Decision 14 (papers first when precise); Decision 15 (GPU tensor-core scope).

### Q7.8: Which words name a run-time violation, a generation-time rejection, and a case the spec leaves open?

- Context: Decision 16 and R55 define a "fault" (writes no region and no state; reported with the clause id), but the glossary has no "fault", still defines "undefined behavior", and lets "reject" cover run time (AG6).
- Context: "implementation-defined" lets a generator decide a case, which Decision 16 does not allow as an open case.
- Context: repos use "refus" 29, 214, and 85 times in taidl, oracle, and backend; `test_refusals.py` is a file name.
- Option A: add "fault"; limit "reject" to generation and kernel compile time; delete "undefined behavior"; delete "implementation-defined"; add "undefined behavior" and "UB" to the banned words (AG6 proposal).
- Option B: as A, but keep "implementation-defined" restricted to coverage status in each generator's docs/coverage.md.
- Option C: keep "reject" for every stage, with the sentence naming the stage (glossary/rulings.md row), and use "fault" only for run-time cases.
- Option D: use "error" for both and drop "fault".
- Sub-question: rename "refus" to "reject" in tests, identifiers, and messages (`test_refusals.py` to `test_rejections.py`), or only in prose.
- Evidence: plans/glossary/terms.md rows "reject", "undefined behavior", "implementation-defined"; plans/glossary/rulings.md rows on "refuse" and rejection stages; plans/minutes/rulings.md R55; plans/reviews/act-glossary/review.md AG6; taidl/taidl/instruction.py:116,145,451.
- Binding: Decision 16 (no undefined behavior in v3; every run-time violation is a defined error or a defined behavior under one simple norm).

### Q7.9: What does the glossary say about the in-place mark and the placement marks?

- Context: Decision 17 defines the in-place mark as "the output reuses the marked input's storage"; R5 removes `set_inplace` as redundant, and R58 removes the placement marks, yet the glossary keeps both rows (AG7).
- Context: the audit holds that text identity is a sound derivation but leaves open equal regions spelled differently and partial overlap; the semantics track keeps `set_inplace`.
- Option A: keep `set_inplace` and define the mark per Decision 17; delete the placement-mark row (AG7 proposal if the owner confirms).
- Option B: remove `set_inplace` and define an "in-place alias" as an input and an output with identical region texts (R5); delete the placement-mark row.
- Option C: keep the row and mark it pending until the semantics panels rule on R5 and RA2; delete the placement-mark row now.
- Option D: keep both rows until the v3 migration list is final.
- Evidence: plans/glossary/terms.md rows "in-place mark", "placement mark"; plans/minutes/rulings.md R5, R58; plans/reviews/rulings/audit.md R5, R58, RA2; plans/reviews/act-glossary/review.md AG7, AG18; act-backend/examples/mini_trn/mini_trn.py:97-99; act-backend/tests/test_extraction.py:507.
- Binding: Decision 17 (the output reuses the marked input's storage); Decision 16.

### Q7.10: Which remaining identifier renames land in v3, and how is the rename set specified and checked?

- Context: R2 renames internal names such as `computation_attr` "per the reconciled glossary", but that glossary still carries renames that Decisions 19-21 forbid (audit R2 row).
- Context: AG5 notes the only reason to rename `update` to `assignments` was the control register term; conflicts.md defaults keep `computation_attr` and rename the grammar rule `instruction` to `line`, while the glossary renames it to `operation`.
- Context: the identifier columns of terms.md cite only front-end line numbers; consumers in act, both generators, both mini_trn examples, and tests are missing, and some line cites are off (AG10, AG11).
- Option A: rename all glossary identifiers in v3 (`computational_attrs`, `addressing_attrs`, `element_type`, `access_dims`, `base_element_dims`, rule `operation`, token `OPERATOR`), keep `update`, `update_asts`, `update_text`, and drop the five renames that depended on "control register" (R2 with AG5).
- Option B: rename only public Python API names; keep grammar rule and token names, with docs saying the term (terminology conflicts #6, #7 defaults).
- Option C: rename only identifiers that collide with another term; defer the rest.
- Option D: rename nothing in v3 beyond Decisions 19-21 and the `hbm` question.
- Sub-question: publish a per-repo table of old name, new name, and consumer files generated by grep, and add a CI grep that old names return nothing after v3, or cite symbols instead of line numbers (AG10, AG11).
- Evidence: plans/glossary/terms.md identifier column; plans/glossary/rulings.md "Instructions" and "Semantics language" tables; plans/terminology/conflicts.md #6, #7; taidl/taidl/antlr4/IDLV2.g4:6,8,28,45; taidl/taidl/antlr4/IDLV2Expr.g4:8,11; taidl/taidl/accelerator.py:62-65; plans/reviews/act-glossary/review.md AG5, AG10, AG11; plans/reviews/rulings/audit.md R2.
- Binding: Decision 10 (breaking changes allowed, batched); Decision 19 (buffers, state); Decision 20 (`Region` stays; the rename list for it is empty); Decision 21 (`@s.` stays); Decision 14.

### Q7.11: What are the canonical names of the generators, repos, packages, and entry points?

- Context: the oracle generator is "TAIDL-TO (Test Oracle) Generator", "Test Oracle generator", and "Test Oracle Generator" in README.md; the backend is "ACT-Backend Generator" and "Compiler Backend Generator"; repo `act-oracle`, distribution `taidl-to`, module `taidl_to`, function `generate_oracle` (AG12).
- Context: the distribution is `act-ecosystem` while the import package and README say `act`; R52 keeps the distribution name and fixes the README sentence.
- Context: R53 says the terms `oracle`, `backend`, `kernel` were added, while the glossary adds "test oracle", "compiler backend", "kernel function" and bans the bare words (AG21); the short form "kernel" is carried by both "kernel function" and "tiled kernel" (AG9); R36 keeps the public `kernel(...)` entry name.
- Context: the generated directory `targets/<Name>/` uses the banned "target"; the glossary renames it to `accelerators/<name>/` (glossary/rulings.md).
- Option A: prose "test oracle generator" and "compiler backend generator"; keep repo, distribution, and function names; one glossary row per name; the short form "kernel" only for the `@kernel` decorator and call (AG9, AG12 proposals).
- Option B: rename distributions, modules, and functions to match the prose names in v3.
- Option C: harmonize prose only and add no glossary rows for repo or package names.
- Option D: keep `targets/` and exempt the path in the glossary (act glossary_objections default).
- Sub-option on `act-ecosystem`: keep and fix the sentence (R52), or rename the distribution to `act`.
- Evidence: plans/act/plan/terminology-docs.md A1, A5, A8; plans/act/findings.md generator-name rows and cross-repo observations; plans/act/glossary_objections.md; plans/glossary/terms.md rows "generator", "test oracle", "kernel function", "tiled kernel"; act-oracle/taidl_to/generator.py:21,22,31; act-backend/act_backend/generator.py:95; act/tests/test_end_to_end.py:81; act/pyproject.toml:2; plans/minutes/rulings.md R36, R52, R53; plans/reviews/act-glossary/review.md AG9, AG12, AG21.
- Binding: Decision 14; Decision 10 (a distribution rename is a breaking release for pins).

### Q7.12: How does a user migrate from v2 to v3?

- Context: v3 breaks R2 renames, R5, R58, and R60 constructs; the minutes have no ruling on the user migration path (RA11).
- Context: A23 as proposed (Q7.3) includes a migration note listing renames.
- Option A: a rejected v3 construct reports its replacement with the clause id (R44); no converter (RA11 proposal).
- Option B: ship a converter script that rewrites v2 ISA descriptions to v3.
- Option C: a migration note only, listing every rename and removal; no replacement text in errors.
- Option D: keep a deprecated alias for each renamed v2 name for one release.
- Evidence: plans/reviews/rulings/audit.md RA11; plans/minutes/rulings.md R2, R5, R44, R58, R60; plans/reviews/act-glossary/review.md AG14; plans/act/findings.md (tutorials using `from taidl import Accelerator`).
- Binding: Decision 10 (breaking changes allowed, batched into one major release); Decision 11 (capacity).

## For the authors

- Reword R26's reason: keep `Buffer` because it is the owner's term (Decision 19), not because "tensor buffer" is the papers' term (audit R26 row).
- Reword R53 to name the terms actually added: "test oracle", "compiler backend", "kernel function" (AG21).
- Reword R27 from "State registers" to "state" (AG8).
- R2 should name glossary/README.md line 7 as its source and strike the `add_tensor_buffer` and `add_control_register` renames from glossary/rulings.md (audit R2 row).
- Restore R34 or mark the number retired (RA1).
- Retitle A19 to "state and constraints" and drop the in-place clause until Q7.9 resolves (AG18).
- Point plan/terminology-docs.md:3 and semantics/README.md:4 at plans/glossary/terms.md instead of /home/user/work/ (AG19).
- Merge A22 into A11 and A14 into the tutorial item; defer A6, A13, and A18 (AG20).
- Fix line cites in terms.md: instruction.py:453 has no "refus" (the word is at :116, :145, :451); data_model.py:38,42 omits :48 (AG11).
- Add a glossary rule that "oracle" alone stays only in repo, package, and directory names (glossary/rulings.md).
- Delete the "HBM (for d0)" row of words-to-avoid.md and flip the rows "region -> data slice", "slice", "tile, window", and "region value" to the Decision 20 form (override table).
- Source column: cite "Decision 19" and "Decision 20" for the rows they override, not [ACT] or [TAIDL] (override table).
- Compare ACT_ENV_TAG with ACT_ENV_BUILD_TAG in check_versions.py (A15; docker/act-env.env:2).
- Add exercise 4 to copy.sh usage only if the tutorial survives Q7.1 (tutorials/pldi26/copy.sh:8).
