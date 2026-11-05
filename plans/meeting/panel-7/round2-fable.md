# Panel 7, round 2: Fable (ML-compiler researcher lens)

Owner decisions bind; where a question is settled by one, the entry records the settlement and does not argue it.

### Q7.1

- All three debaters hold Option A with R54-b; Opus adds Option C for the layout of the one v3 tutorial tree.
- Layout is decided inside the new tutorial item, and Option C (one tree plus `tutorials/events/<event>.md`) is the layout I would pick there, since Option D needs three exercise sets to run on one release and they do not.
- Final position: Option A with R54-b; layout per Option C inside the tutorial item.
- Confidence: high.
- Nothing divides the panel.

### Q7.2

- Sonnet and Opus hold Option A; I held Option C plus the AG2 deletions, which is the same end state: terms.md defines, words-to-avoid.md is generated from its banned column, rulings.md keeps only open rows, terminology/ and the spec-draft glossary are deleted or reduced to pointers, and every /home/user/work/ pointer goes to plans/glossary/terms.md.
- Opus adds a CI grep of the banned column over docs, which is the only check that survives v3 and costs one job.
- Final position: Option A with words-to-avoid.md generated and a CI grep of the banned column.
- Confidence: high.
- Nothing divides the panel.

### Q7.3

- Strongest opposing argument (Opus): Decision 10 says "matching generator releases", and the plain reading of "matching" is one major number across the four repos, so act-backend goes 1.16 to 3.0 and the compatibility table becomes a check instead of a lookup.
- Rebuttal: Opus's own external evidence cuts the other way, since torchvision numbers its releases apart from torch and carries a torch-to-torchvision table, which is the practice act/README.md already has in its Version Compatibility table.
- Rebuttal: Decision 10 says "matching generator releases", not "matching version numbers", and act v2.4 already pins taidl 2.15, taidl-to 2.11, and act-backend 1.16 (act/pyproject.toml:15-17), so "matching" has meant "released together" for eight table rows.
- Sonnet's point stands: act/README.md:4 already promises "an even faster simulator in `act-oracle` v3", so taidl-to 3.0 is public.
- Jumping act-backend from 1.x to 3.0 skips a major that never existed, and a reader of the table then asks what 2.x was.
- Final position: Option A; numbers taidl 3.0, taidl-to 3.0, act-backend 2.0, act 3.0, image tag `v3.0`; the act-backend number is a value judgment and goes to the owner with both options if the moderator finds the evidence balanced.
- Confidence: high on A23 and the merge of A16 and A17; medium on act-backend 2.0.
- Divides the panel: act-backend 2.0 (Fable, Sonnet) against 3.0 (Opus).

### Q7.4

- Strongest opposing argument (Sonnet, joined by Opus): `generate_backend` builds a Rust crate at generation time and needs cargo, CMake, make, a C++17 compiler, OR-Tools, and the nlohmann/json headers (act-backend/README.md:28-34), so prebuilt wheels cannot remove the toolchain, and a fresh-venv gate on a stock runner must install that toolchain, which is the Docker image again.
- Verified: act-backend/README.md:30-34 lists that toolchain and names the `act-env` images as the place that holds it; act/docker/Dockerfile.amd64:85-98 installs the nlohmann headers and the ANTLR jar; the CI workflow builds the image and runs tests inside it (act/.github/workflows/test.yml:23-34).
- Concession: for the full flow (front end, oracle, backend) Docker is the only path the project can test at release cost, so Option B is right for the backend half, and my round-1 claim that the gate costs one CI job understated the OR-Tools and nlohmann setup it would duplicate.
- Correction to AG15: the ANTLR tool is not an install requirement, since taidl ships its generated parsers (`taidl/taidl/antlr4/IDLV2ExprLexer.py` and siblings are packaged, and taidl/pyproject.toml:15 pins only the Python runtime), so the jar in the image serves grammar regeneration only.
- What survives of Option A: `pip install` of the tag plus `generate_oracle` needs Python 3.14, git, JAX, nanobind, pip's cmake, and a C++ compiler (act-oracle/pyproject.toml:15-21), all present on a stock runner, so a fresh-venv gate for the front end and the oracle half of the quickstart is cheap and tests the README's first command.
- Final position: Option B for the supported full flow, with the quickstart run inside the image as the release gate; plus a fresh-venv gate on the stock runner that installs the tag and runs the oracle half of the quickstart; the README pip line states the backend toolchain by pointing at act-backend/README.md.
- Confidence: medium-high.
- Divides the panel: whether the oracle-only pip gate is worth one more CI job (Fable yes; Sonnet and Opus test the image only).

### Q7.5

- All three debaters hold Option A, run by A20, with A3 reduced to the newest "moves the ecosystem" line and no tutorial link until the v3 tutorial exists.
- Sonnet's measurement decides my round-1 threshold: the smallest complete accelerator is the 37-line `MM` function with one buffer and three instructions (act/tests/test_end_to_end.py:30-66), and it cannot shrink below three instructions because `d0` is `u8` (taidl/taidl/accelerator.py:31), so load and store must carry a `bitcast_convert` each.
- A 37-line block is still one screen, so it stays in the README verbatim, copied from a repo file that A20 runs, so the two cannot drift.
- Final position: Option A, sized by the end-to-end accelerator (about 37 lines), kept in a tested example file and shown in full in the README; A3 keeps the newest line only.
- Confidence: high.
- Nothing divides the panel beyond the exact length.

### Q7.6

- Method names are agreed by all three: `add_buffer(name, access_dims, base_element_dims, element_type)` and `add_state(name, value)`; no object layer and no kind argument.
- Strongest opposing argument on the singular (Opus): the code already says "Declare a machine state variable, @s.<name>" (taidl/taidl/accelerator.py:47), so "state variable" is the term in use, while "state entry" has no source anywhere.
- Verified: accelerator.py:47 reads exactly that; the glossary's own cites for "state" are to this method.
- Sonnet's counter is that "variable" collides with `var_type` and `var_name` (taidl/taidl/data_model.py:28,33) and with the ACT paper's "tensor variables"; the first collision disappears when Q7.10 renames `var_type` to `element_type`, and the second is consistent use, since both are named values of a declared kind.
- Concession: my round-1 worry that "state variable" reads as mutable program state is a reading from PyTorch, where the owner's code and `@s.` (Decision 21) are the closer context; one named value that instructions read and `update` writes is a variable in the plain sense.
- Final position: Option A in full: the method names above, singular "state variable" unbanned, state defined for v3 as "a named value of a declared kind, an integer by default" to cover R14.
- Confidence: high on names; medium on the singular.
- Divides the panel: "state variable" (Opus, now Fable) against "state entry" (Sonnet).

### Q7.7

- Settled by the owner: the prose term for the built-in buffer `d0` is "hbm" (Decision 19), and no alternative is an option.
- Consequences for the plan: delete the "global memory" row and the "HBM (for d0)" banned row, drop the A4 and findings.md rewrites of "HBM" to "off-chip memory", and ban "global memory", "off-chip memory", "DRAM", and "main memory" as prose for `d0`.
- GPU sub-question, agreed by all three: in the tensor-core scope device memory is `hbm`, and shared memory and register fragments are buffers.
- Final position: Option A as the owner ruled it.
- Confidence: high.
- Nothing divides the panel after the ruling.

### Q7.8

- All three debaters hold Option A on the words: add "fault" for a defined run-time error, limit "reject" to generation and kernel compile time, delete "undefined behavior" and "implementation-defined", ban "UB".
- Strongest opposing argument on the sub-question (Opus, joined by Sonnet): messages and READMEs are user-visible, so a prose-only rename leaves users reading "refused" in the text that rejects their spec, and the v3 batch is the one time the rename costs nothing extra.
- Verified: act-backend/README.md:48 says "A spec the generator refuses raises `ValueError`, which names what is refused", and the docstrings at taidl/taidl/instruction.py:116,145,453 say "Refuse" or "refused".
- Concession: the public word a user reads is the one the glossary rules, so the rename of "refus" to "reject" covers messages, public docstrings, READMEs, and `test_refusals.py`, done from the Q7.10 table in the v3 batch; my round-1 split between prose and code kept two words for one concept.
- Cite fix for the authors: "refus" in instruction.py is at :116 and :145 as "Refuse" and at :453 as "refused"; the agenda's :451 is a blank line.
- Final position: Option A, with the "refus" rename applied to identifiers, messages, docs, and test file names in v3.
- Confidence: high.
- Nothing divides the panel after the concession.

### Q7.9

- Strongest opposing argument (Sonnet): Decision 17's text "the output reuses the marked input's storage" presupposes a mark, so R5's removal of `set_inplace` conflicts with a binding decision and the glossary can define the mark now (Option A).
- Rebuttal: the glossary can and should carry Decision 17's wording now, which both Option A and Option C do; what Option A adds is a ruling that the mark is spelled `set_inplace`, and whether the mark is an explicit call or the identity of region texts is the question R5 and RA2 put to the semantics panels.
- Rebuttal: Sonnet's reading of "marked" is evidence for those panels, not a reason for this one to pre-empt them; a glossary row whose identifier column says "pending R5" states what is true today.
- Agreed by all three: the placement-mark row goes now, since R58 removes `set_apart` and `set_no_partial_overlap` and no example calls them.
- Final position: Option C: the row defines the in-place mark in Decision 17's words, its identifier column is pending R5 and RA2, and the placement-mark row is deleted.
- Confidence: medium.
- Divides the panel: fix `set_inplace` in the glossary now (Sonnet) or leave the spelling to the semantics panels (Fable, Opus).

### Q7.10

- Strongest opposing argument (Sonnet): grammar rule and token names are not visible in an ISA description, so renaming `instruction` to `operation` and `OPERATION` to `OPERATOR` gives a newcomer nothing; Sonnet would change on evidence that ANTLR error messages show these names.
- Verified: taidl collects ANTLR's own messages unchanged (taidl/taidl/instruction.py:23-30 appends `msg` from `syntaxError`), and ANTLR's default messages spell expected tokens by their symbolic names when a token has no literal, so `OPERATION` appears in "mismatched input ... expecting" text; rule names do not appear in those messages.
- Correction to my round 1: the token rename is user-visible through messages; the rule rename is visible only in the spec's grammar listing and the parser class names (`InstructionContext`), which is Opus's argument, not mine.
- The rule rename still lands in v3, because the spec will print the grammar, and a rule `instruction` beside prose "operation" is two names for one construct; the rename touches `IDLV2.g4:6,8` and regenerated parser files, a small cost inside the batch.
- Final position: Option A: rename the glossary identifiers in v3 including rule `operation` and token `OPERATOR`; keep `update`, `update_asts`, `update_text`; drop the five control-register renames; publish a grep-generated per-repo table citing symbols and add a CI grep that old names return nothing.
- Confidence: high on the API names and the token; medium on the rule name.
- Divides the panel: the grammar rule rename (Fable, Opus yes; Sonnet no).

### Q7.11

- All three debaters hold Option A: prose "test oracle generator" and "compiler backend generator", repo, distribution, module, and function names kept, one glossary row per name, "kernel" alone only for `@kernel` and its call, `act-ecosystem` kept with the README sentence fixed (R52), and `targets/<Name>/` renamed to `accelerators/<name>/` in the v3 batch.
- Sonnet's point sharpens the case for keeping the inner names: a newcomer imports only `act.taidl` and `act.generators` (act/tests/test_end_to_end.py:11-12), so `taidl_to` and `act-oracle` are never typed.
- Final position: Option A with R52.
- Confidence: high; medium on the `targets/` rename, which depends on no external script reading the path.
- Nothing divides the panel.

### Q7.12

- All three debaters hold Option A plus a migration note generated from the Q7.10 rename table, with no converter and no working aliases.
- Sonnet's refinement is right: a removed method must leave a stub that raises with the replacement name and the clause id, since otherwise a v2 script dies with a bare `AttributeError` before any R44 message runs; a stub that only raises is a defined rejection, not an alias.
- Final position: Option A with raising stubs for removed methods and the migration note generated from the rename table.
- Confidence: high.
- Nothing divides the panel.

### Escalation to the owner

- None; no owner decision in this panel's scope has an evidence-backed counter from this lens.
