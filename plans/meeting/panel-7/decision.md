# Panel 7 decision: vocabulary and usability

Owner decisions 6, 10, 11, 14, 15, 16, 17, 19, 20, 21 bind, with two owner inputs received during the meeting: the prose term for `d0` is "hbm", and `set_inplace` stays unless panels 1 and 2 show it truly redundant.
Every decisive citation below was re-read in the repos; a claim the moderator could not verify is marked so.

| Question | Decision (one line) | Confidence | Owner? |
| -------- | ------------------- | ---------- | ------ |
| Q7.1 | Option A with R54-b: archive all three tutorials on their tags, one new tutorial on the v3 release candidate, layout chosen inside that item | high | no |
| Q7.2 | Option A containing C: terms.md is the only glossary, words-to-avoid.md generated from its banned column, superseded files deleted, CI grep of the banned column | high | no |
| Q7.3 | Option A: add A23 "act v3 release", merge A16 and A17 into S8 and A23; taidl 3.0, taidl-to 3.0, act 3.0, image v3.0; act-backend number open | high on item, low on numbers | yes, act-backend 2.0 or 3.0 |
| Q7.4 | Option B for the full flow with the quickstart run in the image as release gate, plus a fresh-venv gate that installs the tag and runs the oracle half | medium | no |
| Q7.5 | Option A: a complete accelerator (about 37 lines) kept in a tested example file, shown in full in the README, run by A20; A3 keeps the newest line only | high | no |
| Q7.6 | Option A: `add_buffer(name, access_dims, base_element_dims, element_type)`, `add_state(name, value)`; singular "state variable" unbanned; state is "a named value of a declared kind, an integer by default" | medium-high | no |
| Q7.7 | Option A as the owner ruled: the term is "hbm", the built-in buffer `d0`; ban "global memory", "off-chip memory", "DRAM", "main memory" | high | no |
| Q7.8 | Option A: "fault" at run time, "reject" at generation and kernel compile time; delete "undefined behavior" and "implementation-defined", ban "UB"; rename "refus" to "reject" in messages, docstrings, READMEs, and public identifiers as its own v3 item | high on words, medium-high on scope | no |
| Q7.9 | Option C: the row defines the in-place mark in Decision 17's words, its identifier column is pending panels 1 and 2 (R5, RA2); delete the placement-mark row now | high on the pending form | pending panels 1 and 2 |
| Q7.10 | Option A: rename the public API names, token `OPERATOR`, and rule `operation`; keep `update`, `update_asts`, `update_text`; drop the five control-register renames; grep-generated per-repo table citing symbols plus a CI grep | medium-high on API and token, medium on the rule | no |
| Q7.11 | Option A: prose "test oracle generator" and "compiler backend generator"; keep repo, distribution, module, and function names; keep `act-ecosystem` (R52); `targets/` to `accelerators/` in the v3 rename table | high on prose, medium on the path | no |
| Q7.12 | Option A plus a migration note and raising stubs, all generated from the Q7.10 table; no converter, no working alias | high | no |

### Q7.1

- Decision: Option A with R54-b; the layout of the one v3 tree is decided inside the tutorial item, with Option C (one tree plus `tutorials/events/<event>.md`) as the default there.
- Reasoning 1: Decision 6 says no tutorial is current, so A9, A10, and A12 act on an object that does not exist.
- Reasoning 2: the newest tree pins image v1.2 and the micro25 solution imports `from taidl import Accelerator` and calls `qkv.generate_oracle()` as a method, while the entry is the free function `generate_oracle(accelerator, base_dir)`; a refresh is a rewrite.
- Reasoning 3: v3 renames (R2, R5, R58, R60) break a refreshed pre-v3 tutorial again, so Option B pays twice under Decision 11.
- Reasoning 4: Options C and D lay out trees about to be archived, and Option D needs three exercise sets to run on one release, which step 2 shows they do not.
- Reasoning 5: 57 files under tutorials/ hold non-ASCII; R54-a and R54-c amend the ASCII rule for a transient case, while R54-b (trees stay on tags, off master) needs no rule change.
- Evidence: tutorials/pldi26/docker.sh:10-12; tutorials/micro25/exercise1/solution/QKV.py:3,108; act-oracle/taidl_to/generator.py:21; rulings.md R63; engineering.md ASCII rule; the grep count 57 re-run by the moderator.
- Dissent: none in round 2; Opus fixed the layout now and conceded it to the tutorial item.
- Confidence: high.

### Q7.2

- Decision: Option A containing Option C; overridden rows cite "Decision 19", "Decision 20", or "Decision 21" in the source column; every /home/user/work/ pointer goes to plans/glossary/terms.md; a CI grep of the banned column runs over docs.
- Reasoning 1: the README override block leaves two live rules for at least eight concepts, so a reader cannot know which wins without a two-step read (AG1).
- Reasoning 2: words-to-avoid.md duplicates the banned column and drifts; generation removes the second source (AG20).
- Reasoning 3: rulings.md narrates how terms were reached, which the docs discipline forbids in a living file; only open rows stay.
- Reasoning 4: Options B and E keep the contradictions alive as "history", and Option D is Option A with more file moves.
- Evidence: plans/glossary/README.md:5-10 (override list, verified); terms.md:17,18,23 still rule "tensor buffer", "global memory", "control register"; AG1, AG2, AG19, AG20; engineering.md "A doc records no history".
- Dissent: none.
- Confidence: high.

### Q7.3

- Decision: Option A; A23 "act v3 release" owns pins, README row, image tag, the "moves the ecosystem" sentence, and the migration note; A16 and A17 merge into S8 and A23; numbers taidl 3.0, taidl-to 3.0, act 3.0, image `v3.0`; act-backend 2.0 or 3.0 goes to the owner.
- Reasoning 1: act is the only repo that pins all three siblings and holds the table and image tag, so a release train without an act item cannot finish (AG14).
- Reasoning 2: check_versions.py requires the newest row, the pins, and a sentence opening "ACT v<n>" below the table to agree, so A23 is mostly filling what the script checks.
- Reasoning 3: the README already promises "an even faster simulator in act-oracle v3", so taidl-to 3.0 is public either way.
- Reasoning 4: the act-backend number is a value judgment on what "matching generator releases" means in Decision 10; the evidence is balanced.
- Option 3.0 best argument (Opus): one major number across four repos names one concept, and the table becomes a check rather than a lookup.
- Option 2.0 best argument (Sonnet, Fable): act-backend is at 1.16, eight table rows already show "matching" as "released together", and a jump to 3.0 leaves a reader asking what 2.x was; Opus's torchvision example also numbers apart from torch.
- Evidence: act/pyproject.toml:15-18; act/README.md:4,66-82; act/tests/check_versions.py:36-62; AG14, AG17.
- Dissent: Opus holds 3.0; Opus's Angular 2-to-4 precedent is uncited and carries no weight.
- Confidence: high on the item, low on the act-backend number.

### Q7.4

- Decision: Docker is the supported full flow (Option B), documented first, with the Q7.5 quickstart run inside the built image as the release gate; a second gate installs the tag into a fresh venv on the stock runner and runs the front end and oracle half of the quickstart; no wheels in v3; the README pip line points at act-backend/README.md for the backend toolchain.
- Reasoning 1: CI builds the image, checks versions, lints, and runs the end-to-end test inside it; the README's first command, `pip install git+...@v2.4`, is never run.
- Reasoning 2: AG15 overstates the install cost: taidl depends only on `antlr4-python3-runtime==4.9.2` and ships its generated parsers, act-backend depends only on taidl, and act-oracle needs numpy, jax, jaxlib, nanobind, and cmake from pip; a stock runner with Python 3.14 can do that.
- Reasoning 3: generating a backend needs cargo, CMake, make, a C++17 compiler, OR-Tools, and nlohmann/json on the machine, so a stock-runner gate for the backend half rebuilds the image, and prebuilt wheels cannot remove it.
- Reasoning 4: a scheduled test (Option C) finds a broken tag after the newcomer hit it; the venv gate costs one job per tag and tests exactly the first command.
- Reasoning 5: Opus's "stop advertising the pip path" fixes the mismatch by subtraction, but the pip path for the front end and oracle works and is cheap to prove, so it stays supported for that half.
- Evidence: act/.github/workflows/test.yml:23-34; act/README.md:24-30; taidl/pyproject.toml:15; taidl/taidl/antlr4/ (generated parsers checked in); act-oracle/pyproject.toml:15-22; act-backend/pyproject.toml:15-17; act-backend/README.md:34-36; act/docker/Dockerfile.amd64:92-98,118-127.
- Dissent: Opus holds image-only; the deciding fact, who the users are, has no citation from anyone.
- Confidence: medium.

### Q7.5

- Decision: Option A; the quickstart is the smallest complete accelerator (load, compute, store, about 37 lines), kept in a tested example file that A20 runs and shown in full in the README; the README links the spec and the coverage tables; the tutorial link waits for the v3 tutorial (Decision 6); A3 keeps only the newest "moves the ecosystem" line; Docker and testing detail move below or into docs/.
- Reasoning 1: the usage snippet reads `# ... define ISA spec ...`, so nothing in the README runs or teaches vocabulary (AG16).
- Reasoning 2: the end-to-end test accelerator is the only complete one in act, and it cannot drop below three instructions because `d0` is `u8`, so load and store each need a `bitcast_convert`.
- Reasoning 3: a snippet CI imports but does not run rots unseen; copying from a file A20 runs keeps README and code from drifting.
- Reasoning 4: check_versions.py requires exactly one sentence opening "ACT v<newest>" below the table; the older four lines are history.
- Evidence: act/README.md:36-45,77-82; act/tests/test_end_to_end.py:11-12,29-66; taidl/taidl/accelerator.py:31; act/tests/check_versions.py:61-65; AG16; R51.
- Dissent: none beyond the exact length.
- Confidence: high.

### Q7.6

- Decision: Option A; `add_buffer(name, access_dims, base_element_dims, element_type)` and `add_state(name, value)`; `State` and `@s.` stay; "state variable" is unbanned as the singular, "machine state variable" stays banned; state is defined as "a named value of a declared kind, an integer by default" (R14); R27 says "state"; the terms.md "control register" row flips to "state" with the Decision 19 source cite.
- Reasoning 1: Decision 19 fixes the nouns, and the flat methods spell them; Options C and D add an object or a kind switch for two things with different arguments.
- Reasoning 2: "state variable" is the term already shipped: the docstring "Declare a machine state variable", the error-message kind table `'@s.': 'state variable'`, and docs/expressions.md use it; "state entry" has no source.
- Reasoning 3: the collision with `var_type` and `var_name` fades when Q7.10 renames `var_type` to `element_type`; state is written by `update` lines, so "variable" is accurate.
- Evidence: taidl/taidl/accelerator.py:33-35,46-47; taidl/taidl/instruction.py:397; taidl/docs/expressions.md:3,13; act/act/taidl/__init__.py:3; terms.md:23; AG3, AG8; R14.
- Dissent: Sonnet and Fable held "state entry" or "a state" in round 1 and both conceded to "state variable" in round 2.
- Confidence: medium-high; the singular is a naming choice the owner may overrule.

### Q7.7

- Decision: Option A as the owner ruled; the term is "hbm", the built-in buffer `d0`, sized by the `hbm` kernel argument; ban "global memory", "off-chip memory", "DRAM", "main memory", and uppercase "HBM" as a name for `d0`; delete the terms.md "global memory" row and the words-to-avoid "HBM (for d0)" row; drop the A4 and findings.md rewrites of "HBM" to "off-chip memory"; the row's source column may note that the papers write "global memory (HBM)", as a pointer and not a synonym.
- Reasoning 1: the owner input settles the prose term; Decision 19 already keeps the identifier.
- Reasoning 2: one spelling for identifier and prose removes a mapping step, and the kernel parameter and the backend's buffer name already spell it.
- Reasoning 3: GPU scope (Decision 15): device memory is `hbm`; shared memory and register fragments are buffers, so one data-model rule covers every accelerator.
- Evidence: owner input; act-oracle/taidl_to/generic/decorator.py:142; act-backend/act_backend/instruction_info.py:70-71; terms.md:18; AG4.
- Dissent: none after the ruling.
- Confidence: high.

### Q7.8

- Decision: Option A; "fault" is a defined run-time error that writes no region and no state and is reported with the clause id (R55); "reject" covers generation and kernel compile time only; delete the "undefined behavior" and "implementation-defined" rows and add "undefined behavior", "UB", and "implementation-defined" to the banned column; coverage status says "unsupported"; the "refus" rename covers messages, docstrings, READMEs, and public identifiers in the v3 batch as its own plan item fed by the Q7.10 table; `test_refusals.py` follows the test-suite item (panel 6).
- Reasoning 1: Decision 16 forbids undefined behavior and open cases, so both rows define what cannot exist; a banned word is their only place.
- Reasoning 2: a compiler reader expects a static-versus-dynamic split (verifier failure versus trap); Option D loses it and Option C leaves it to each sentence, the present ambiguity.
- Reasoning 3: StableHLO uses both banned words for open cases; v3 departs on purpose, so importing the words imports the open case (Option B).
- Reasoning 4: "refused" is user-visible in the backend README and the generator docstring, so a prose-only rename leaves users reading the banned word.
- Cite fix: "refus" in taidl/taidl/instruction.py is at :116 and :145 ("Refuse") and :453 ("refused"); :451 is blank.
- Evidence: R55; terms.md:86,88,89; act-backend/README.md:48; act-oracle/taidl_to/generator.py:24-25; counts 27, 215, 73 in .py, .md, .rs (re-run); act-oracle/tests/test_refusals.py and act-backend/tests/test_refusals.py exist.
- Dissent: Fable held prose-only in round 1 and conceded.
- Confidence: high on the words, medium-high on the rename scope.

### Q7.9

- Decision: Option C; the row reads "in-place mark: the output reuses the marked input's storage" (Decision 17), its identifier column reads "pending panels 1 and 2 (R5, RA2)", and the placement-mark row is deleted now (R58).
- Reasoning 1: the owner input routes the `set_inplace` question to panels 1 and 2 and says it stays unless shown truly redundant; this panel follows their ruling and does not pre-empt it.
- Reasoning 2: the owner input also says overlap alone does not settle read-after-write, so a glossary row that defines the mark as text identity (Option B) is excluded now.
- Reasoning 3: R58 removes `set_apart` and `set_no_partial_overlap` and no example calls them, so the placement-mark row names nothing in v3.
- Evidence for panels 1 and 2: `set_inplace` is checked as one flag per input; mini_trn softmax repeats one region text in input and output; test_extraction.py:507 marks a `d0` input in place for a `d2` output, so text identity does not cover an existing fixture, as the audit R5 row states.
- Evidence: owner input; Decision 17; rulings.md R5, R58; audit R5, RA2; taidl/taidl/instruction.py:163-172; act-backend/examples/mini_trn/mini_trn.py:97-99; act-backend/tests/test_extraction.py:507; terms.md:44-45; AG7.
- Dissent: Sonnet (round 1) and Opus (round 2) would fix `set_inplace` in the row now, since Decision 17's "marked input" presupposes a mark; the argument is strong and is forwarded to panels 1 and 2 as evidence, but the owner input assigns the ruling to them.
- Confidence: high on the pending form; the row resolves when panels 1 and 2 rule.

### Q7.10

- Decision: Option A; rename `add_data_model` to `add_buffer`, `add_initial_state` to `add_state`, `dimensions` to `access_dims`, `unit_dim` to `base_element_dims`, `var_type` to `element_type`, `computation_attr` to `computational_attrs`, `addressing_attr` to `addressing_attrs`, `DataModel` to `Buffer`, token `OPERATION` to `OPERATOR`, and rule `instruction` to `operation`; keep `update`, `update_asts`, `update_text`, `Region`, `@s.`; drop the five renames that depended on "control register"; publish a grep-generated per-repo table of old name, new name, and consumer files, cite symbols instead of line numbers, and add a CI grep that old names return nothing after v3.
- Reasoning 1: Decision 10 makes v3 the one window; deferring (Option C) or renaming nothing (Option D) leaves `add_data_model` naming one buffer and `var_type` naming an element type, against Decision 19 in the code itself.
- Reasoning 2: the API names are what a newcomer types first, and `var_type` for an element type mismatches StableHLO and PyTorch vocabulary.
- Reasoning 3: the rule `instruction` names one SSA line of the semantics while the class `Instruction` names an ISA instruction; the spec prints the grammar, so one document would use "instruction" for two concepts, and StableHLO and MLIR call that line an operation (Decision 14).
- Reasoning 4: the syntax-error listener passes ANTLR's `msg` through unchanged, so a token with no literal can appear by its symbolic name in "expecting" text; this was not reproduced (no antlr4 runtime on the host) and is plausible, not decisive; the token rename rests on the rule rename, since `operation` beside `OPERATION` would be two spellings of one word.
- Reasoning 5: the rule rename touches IDLV2.g4:6,8, the regenerated parsers, and the call sites that iterate `.instruction()`, a bounded cost inside the batch.
- Evidence: taidl/taidl/accelerator.py:33-35,46,60-65; taidl/taidl/antlr4/IDLV2.g4:6,8,28,45; IDLV2Expr.g4:8,11; taidl/taidl/instruction.py:23-30,291; taidl/taidl/operations.py:174; terminology/conflicts.md #6, #7; AG5, AG10, AG11; audit R2.
- Dissent: Opus (round 2) keeps rule names because consumers already call the node `line` and only code sees the rule; Sonnet (round 1) kept rules and tokens and conceded in round 2.
- Confidence: medium-high on API names and the token, medium on the rule.

### Q7.11

- Decision: Option A; prose "test oracle generator" and "compiler backend generator"; repo `act-oracle`, distribution `taidl-to`, module `taidl_to`, and function `generate_oracle` stay; one glossary row per name maps each spelling to its concept; bare "kernel" only for the `@kernel` decorator and its call, prose "kernel function" and "tiled kernel"; `act-ecosystem` stays and the README sentence is fixed (R52); `targets/<Name>/` becomes `accelerators/<name>/` in the v3 rename table and migration note; "oracle" alone stays only in repo, package, and directory names.
- Reasoning 1: the README spells the oracle generator three ways and the backend two; one prose pair per Decision 14 ends it, and the papers back "test oracle" and "compiler backend generator".
- Reasoning 2: a newcomer imports only `act.taidl` and `act.generators`, so inner names are never typed; renaming them breaks every pin for no new meaning (Decision 10).
- Reasoning 3: the generated path is user-visible and uses a banned word; the batch is the one time to move it, pending evidence of external scripts that read it.
- Evidence: act/README.md:15-20,36-45; act/pyproject.toml:6; act/tests/test_end_to_end.py:11-12; act-oracle/taidl_to/generator.py:21-22,31; act-backend/act_backend/generator.py:95; act-oracle/taidl_to/generic/decorator.py:142; R36, R52, R53; AG9, AG12, AG21.
- Dissent: none; the path rename flips on evidence of external scripts reading `targets/`.
- Confidence: high on prose, medium on the path.

### Q7.12

- Decision: Option A plus a migration note and raising stubs; a rejected v3 construct reports its replacement with the clause id (R44); each removed or renamed public method leaves a stub that only raises, naming the replacement and the clause id; the migration note lists every rename and removal; stubs, note, and messages are generated from the Q7.10 table; no converter and no working alias.
- Reasoning 1: ISA descriptions are Python scripts, so a failing call is the first thing a v2 user sees; without a stub it is a bare `AttributeError` before any R44 message runs.
- Reasoning 2: a stub that only raises admits no v2 call, so it is a defined rejection, not an alias; working aliases keep two names per concept alive, against Decision 10 and one term per concept.
- Reasoning 3: a converter must rewrite arbitrary Python, beyond Decision 11 capacity for a one-time migration.
- Reasoning 4: one table feeds stubs, note, and messages, so they cannot disagree.
- Evidence: audit RA11; R2, R5, R44, R58, R60; AG14; tutorials/micro25/exercise1/solution/QKV.py:3,108 (the first v2 scripts to hit the messages).
- Dissent: none in round 2.
- Confidence: high.

## Escalation to the owner

- No debater raised an escalation against an owner decision; all three round-2 files state "None".
- Owner value judgment, Q7.3: act-backend goes to 2.0 (own major bumped in the same train; eight table rows already read "matching" as released together) or 3.0 (one major across four repos; the table becomes a check); the panel's default is 2.0.
- Owner choice, Q7.6: the singular "state variable" is unbanned on the strength of shipped docstrings, messages, and docs; the owner may name another singular.
- Dependency, Q7.9: the in-place mark row stays pending until panels 1 and 2 rule on R5 and RA2; test_extraction.py:507 is forwarded to them as a fixture that text identity does not cover.
- Dependency, Q7.8: the rename of `test_refusals.py` follows panel 6's test-suite item.
