# Panel 7, round 1: Fable (ML-compiler researcher lens)

Assumptions: a newcomer arrives from MLIR, XLA, or PyTorch and expects the entry path those projects give: one install command, one runnable example, one glossary that matches the identifiers.
Where I cite the papers I rely on the glossary source tags (plans/glossary/README.md source table) rather than a fresh read of the PDFs.

### Q7.1

Position: Option A (archive all three, one new tutorial on the v3 release candidate), with R54-b for the ASCII sub-question.

1. Decision 6 says there is no current tutorial, so A9, A10, and A12 rest on a false premise and cannot be executed as written.
2. The tutorials are already broken against current taidl: micro25 calls `qkv.generate_oracle()` as a method and imports `from taidl import Accelerator`, while the oracle entry is the free function `generate_oracle(accelerator, base_dir)`.
3. pldi26 pins image v1.2, and R63 says the tutorials use syntax the current grammar rejects; a Q1 refresh (Option B) would be broken again by the v3 renames in Q2, so it is wasted work under Decision 11 capacity.
4. Options C and D decide the on-disk layout of trees that will be archived; the layout matters only for the one v3 tutorial, so decide it inside the new tutorial item, not now.
5. Option E leaves a newcomer with nothing that runs, which contradicts the StableHLO and MLIR norm of a runnable tutorial per release.
6. On R54: 57 files under tutorials/ contain non-ASCII today, and engineering.md exempts only binary assets; a per-folder exemption is a rule change the owner must make (R54-c), while leaving the archived trees on their event branches and out of master (R54-b) needs no rule change and is what A11 already proposes.

Evidence: plans/minutes/owner-decisions.md row 6; tutorials/micro25/exercise1/solution/QKV.py:3,108; act-oracle/taidl_to/generator.py:21; tutorials/pldi26/docker.sh:10-12; plans/minutes/rulings.md R63, R54; plans/act/plan/tutorials.md A9-A12; plans/reviews/rulings/audit.md R54; .claude/engineering.md ASCII rule; https://openxla.org/stablehlo/spec (versioned spec with examples per op).
Confidence: high.
Would change my mind: the owner stating that pldi26 must stay runnable at v1.2 for an upcoming event, which would justify Option B for that one tree.

### Q7.2

Position: Option C (terms.md is the only glossary; words-to-avoid.md generated from its banned column; rulings.md reduced to open rows), plus the AG2 deletions from Option A.

1. Two rules per concept for eight concepts (AG1) means a reader cannot know which wins without reading the README override block, and the engineering rule "state what a thing IS" forbids keeping the overridden row.
2. words-to-avoid.md duplicates the banned column of terms.md, so it can be generated and never drift (AG20).
3. rulings.md narrates how terms were chosen; the docs rule puts rationale only in a living status doc, so a resolved row is deleted and an open row stays.
4. terminology/conflicts.md and the spec-draft glossary contradict terms.md (AG2); Option B's "supersedes" lines keep every contradiction alive and multiply the reading cost for a newcomer.
5. Option D merges folders but keeps two tables per concept group, which does not remove the duplication.
6. Option E leaves rulings.md and words-to-avoid.md as "history", which the docs rule forbids in committed files.
7. MLIR keeps one glossary page and dialect docs cite it rather than restating terms; one source of truth is the community norm.

Evidence: plans/reviews/act-glossary/review.md AG1, AG2, AG19, AG20; plans/glossary/README.md override list; .claude/engineering.md docs discipline; https://mlir.llvm.org/docs/ (single Glossary page under the docs index).
Confidence: high.
Would change my mind: a stated need for a per-ruling audit trail, which would argue for keeping rulings.md as a table of open rows only.

### Q7.3

Position: Option A (add A23 "act v3 release"; merge A16 and A17 into S8 and A23) with v3 numbers taidl 3.0, taidl-to 3.0, act-backend 2.0, act 3.0, image devanshdvj/act:v3.0.

1. The act repo pins all three siblings and is the only place the README table and image tag live, so a release train without an act item cannot finish (AG14).
2. A16 and S8 both define the release order; one procedure in two places drifts, so one owner (S8) and one consumer (A23) is the minimum.
3. Decision 10 batches breaking changes into one major release; the matching generator release must bump the major too, since both generators pin taidl at the tag and will break on the renames.
4. act-backend sits at 1.16 today and act at 2.4, so "match the TAIDL major" is impossible without jumping numbers; the honest rule is "each repo bumps its own major in the same train", giving act-backend 2.0 and act 3.0.
5. Option D (independent versioning) is what exists today and is what makes the table of eight rows necessary for a newcomer to decode; keeping one visible rule is cheaper.
6. The README sentence "ACT vX moves the ecosystem to ..." is checked by tests/check_versions.py, so A23 must also update that sentence or the check fails.

Evidence: /home/user/act/README.md:66-80 version table and sentences; /home/user/act/pyproject.toml:15-17 pins; plans/act/plan/versions-release.md A15-A18; plans/reviews/act-glossary/review.md AG14, AG17; plans/minutes/owner-decisions.md rows 10, 11.
Confidence: medium on the numbers, high on the item.
Would change my mind: a semver statement from the owner that generator majors follow the TAIDL major, which would make act-backend 3.0.

### Q7.4

Position: Option A (release gate: fresh-venv install of the tag on the stock runner, then the quickstart), with Docker documented as the path for the GPU variant.

1. The README's first install command is `pip install git+...@v2.4`, and CI only builds the Docker image, so the advertised path is never exercised.
2. A newcomer from PyTorch or JAX expects `pip install` to work on a stock machine; an untested pip path that fails on a missing Rust toolchain or ANTLR is the first and last thing they see.
3. Option B (Docker only) is consistent but hides the product behind a 2-image build; MLIR and StableHLO both document a from-source build on a stock host and test it in CI.
4. Option C (scheduled test, A18) finds a broken tag a week after it ships, which is after the newcomer hit it.
5. Option D adds a packaging dependency (wheel build for Rust and ANTLR) and needs owner approval under the dependency rule; it is the right end state but not a v3 gate.
6. The gate costs one CI job per tag, within Decision 11 capacity.

Evidence: /home/user/act/README.md:24-30; /home/user/act/.github/workflows/test.yml:23-33; plans/reviews/act-glossary/review.md AG15; plans/act/plan/versions-release.md A18; .claude/engineering.md dependency rule.
Confidence: high.
Would change my mind: a measured install time over 20 minutes on the stock runner, which would push the test to a schedule.

### Q7.5

Position: Option A (a complete, runnable quickstart in the README, run by A20, linking spec, coverage tables, and the v3 tutorial), with A3 reduced to the newest "moves the ecosystem" line.

1. The usage snippet has the comment `# ... define ISA spec ...` in place of code, so it cannot run and teaches no vocabulary.
2. StableHLO, MLIR, and PyTorch READMEs each open with one runnable example or one link to a getting-started page; a list of repos and versions (Option C) is what their `CONTRIBUTING` pages hold, not the front page.
3. A20 already plans to test the snippet; a snippet that is only imported proves nothing, while a 15-line accelerator proves install, front end, and one generator.
4. Option B links a tutorial as the first example, but Decision 6 says no tutorial is current until the v3 one exists, so the README would link to nothing for two quarters.
5. Option D documents the present but leaves the newcomer problem (AG16) unsolved.
6. On A3: check_versions.py requires the sentence below the table, so one line stays; the four older lines are history and the docs rule deletes them.

Evidence: /home/user/act/README.md:36-45, 77-81; /home/user/act/tests/check_versions.py; plans/act/plan/terminology-docs.md A1-A3; plans/act/plan/tests-docker.md A20; plans/reviews/act-glossary/review.md AG16; plans/minutes/owner-decisions.md row 6.
Confidence: high.
Would change my mind: evidence that the smallest runnable accelerator exceeds about 25 lines, which would move the quickstart to docs/ with a README link.

### Q7.6

Position: Option A method names, Option B singular: `add_buffer(name, access_dims, base_element_dims, element_type)` and `add_state(name, value)`; one entry is "a state" or "state entry"; "state variable" stays banned.

1. Decision 19 fixes the nouns "buffers" and "state"; `add_buffer` and `add_state` are the only method names that spell those nouns.
2. Today `add_data_model(name, dimensions, unit_dim, var_type)` names the set when adding one unit, which is the AG1 "data model for one unit" confusion in code.
3. The `State` dataclass already exists with `name` and `value`, so `add_state` maps to it without a type rename; `act/act/taidl/__init__.py` re-exports `DataModel` and `State`, so only `DataModel` -> `Buffer` is a consumer rename.
4. "State variable" reads as mutable program state in MLIR and PyTorch; the ISA concept is a named architectural register value, so the hardware term "state" with a plain count noun ("one state", "each state") is clearer and matches `@s.` (Decision 21).
5. Options C and D add an object or a kind argument that no caller needs today; the engineering rule adds abstraction at the second use site.
6. On the sub-question: define state as "a named value of an integer or declared kind" now, because R14 lands typed kinds in v3 and the glossary ships with v3.

Evidence: taidl/taidl/accelerator.py:33-46; taidl/taidl/data_model.py:28-33; act/act/taidl/__init__.py:3-5; plans/minutes/owner-decisions.md rows 19, 21; plans/minutes/rulings.md R14; plans/reviews/act-glossary/review.md AG3, AG8; https://mlir.llvm.org/docs/ (MLIR uses "state" for architectural state in ArmSME and AMX dialect docs).
Confidence: high on names, medium on the singular.
Would change my mind: the owner naming a singular; the papers using "state variable" as a precise term, which under Decision 14 would unban it.

### Q7.7

Position: Option A: the term is "hbm", the built-in buffer `d0`; ban "global memory", "off-chip memory", "DRAM", "main memory".

1. Decision 19 keeps `hbm` as the identifier; the kernel parameter is `hbm` and the Rust variant is `Buffer::HBM`, so any prose synonym forces the reader to map two names for one thing.
2. NKI, the closest industry vocabulary for a tensor accelerator, names device memory `hbm` (alias of `private_hbm`) and on-chip memory `sbuf` and `psum`; a newcomer from Neuron reads "hbm" without translation.
3. "Global memory" is the MLIR gpu and CUDA term, but in MLIR it is an address space (`#gpu.address_space<global>`), not a named buffer; using it for `d0` in prose while the code says `hbm` invites the address-space reading.
4. Option B keeps the papers' term, but Decision 14 prefers the papers only when precise; the papers use HBM as an alias, so "hbm" is also the papers' word.
5. Option D (three synonyms with contexts) is the current state and is what produced AG4.
6. GPU sub-question: in the tensor-core scope, device memory is `hbm`, and shared memory and register fragments are buffers, which keeps one data-model rule for every accelerator.

Evidence: plans/minutes/owner-decisions.md rows 14, 15, 19; act-oracle/taidl_to/generic/decorator.py:142; act-backend/act_backend/instruction_info.py:71; plans/glossary/terms.md row "global memory"; plans/reviews/act-glossary/review.md AG4 and override table; https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.language.html (`hbm`, `sbuf`, `psum`); https://mlir.llvm.org/docs/Dialects/GPU/ (`gpu.address_space` values global, workgroup, private, constant).
Confidence: high.
Would change my mind: the owner stating that the prose term for `d0` must stay "global memory" for continuity with the papers.

### Q7.8

Position: Option A (add "fault"; "reject" only at generation and kernel compile time; delete "undefined behavior" and "implementation-defined"; ban "UB"), and rename "refus" to "reject" in prose and new code only, keeping `test_refusals.py` until a test-suite item renames it.

1. Decision 16 forbids undefined behavior in v3, so a glossary entry defining it describes nothing that exists; a banned word is the right place for it.
2. R55 defines "fault" as the one word for a defined run-time error; three stages with three words (reject, reject, fault) is one word too many, so "reject" covers the two static stages and "fault" the dynamic one.
3. StableHLO uses "implementation-defined" for memory layout and "the behavior is undefined" for cholesky preconditions; both are open cases, which Decision 16 does not allow, so importing the StableHLO term imports the open case.
4. Option B keeps "implementation-defined" for coverage status, but coverage is "supported or rejected", not a semantic choice; the word would mean something different from its StableHLO meaning for the same readers.
5. Option D ("error" for both) loses the static-versus-dynamic distinction that a compiler reader expects (verifier failure versus runtime trap).
6. The "refus" word appears 29, 214, and 85 times across the repos and the docstring at instruction.py:453 reads "one bare string is refused", so AG11's claim that :453 lacks the word is itself off by the current file; a bulk rename is a test-suite item, not a glossary item.

Evidence: plans/minutes/owner-decisions.md row 16; plans/minutes/rulings.md R55; plans/glossary/terms.md rows "reject", "undefined behavior", "implementation-defined"; taidl/taidl/instruction.py:453; act-oracle/tests/test_refusals.py; plans/reviews/act-glossary/review.md AG6, AG11; https://openxla.org/stablehlo/spec (bitcast_convert "implementation-defined"; cholesky "the behavior is undefined").
Confidence: high on the words, medium on the rename scope.
Would change my mind: a generator that must leave a case open for a hardware reason, which would reopen "implementation-defined" with a narrow definition.

### Q7.9

Position: Option C (keep the in-place row marked pending on R5 and RA2; delete the placement-mark row now).

1. R58 removes `set_apart` and `set_no_partial_overlap` in v3 and no example calls them, so the placement-mark row describes something v3 does not have.
2. Decision 17 defines what the mark means but not whether it is spelled; R5 says it is derivable from identical region texts, and the audit (RA2) leaves equal regions spelled differently and partial overlap open.
3. mini_trn's softmax spells `set_inplace([True])` with input and output region `['d2', ['@a.addr'], ['@c.n']]`, so text identity holds in the one example, but one example does not settle RA2.
4. A glossary row that decides R5 would pre-empt the semantics panels; a row marked pending states what is true today.
5. Option A and Option B each pick a side of R5; Option D keeps a dead row.
6. For the MLIR reader, the nearest concept is the `linalg` destination-passing style where the output operand names its storage; whichever side wins, the definition should say "the output reuses the marked input's storage" in those words.

Evidence: plans/minutes/owner-decisions.md row 17; plans/minutes/rulings.md R5, R58; plans/reviews/rulings/audit.md RA2; taidl/taidl/instruction.py:163-180; act-backend/examples/mini_trn/mini_trn.py:97-99; plans/glossary/terms.md rows "in-place mark", "placement mark"; https://mlir.llvm.org/docs/ (linalg destination-passing style).
Confidence: medium.
Would change my mind: a semantics-panel decision on R5, which would resolve the row to Option A or B.

### Q7.10

Position: Option A (rename all glossary identifiers in v3; keep `update`, `update_asts`, `update_text`; drop the five control-register renames), with the sub-question answered as a generated per-repo table plus a CI grep.

1. Decision 10 allows breaking renames batched into v3, and v3 is the only such window; a rename deferred (Option C) waits for v4.
2. Option B keeps grammar rule `instruction` and token `OPERATION` while the glossary says "operation", so the error messages a newcomer reads name the wrong term; grammar rule names leak into ANTLR error text.
3. `access_dims`, `base_element_dims`, and `element_type` replace `dimensions`, `unit_dim`, and `var_type`, which are the three parameter names a newcomer types first; `var_type` for an element type is a StableHLO and PyTorch mismatch (both say element type or dtype).
4. AG5 is right that `update` -> `assignments` had no reason once "control register" fell; "state update" and `update` agree.
5. The identifier columns cite front-end lines only, and the review found 12 of 14 cites exact; a grep-generated table of old name, new name, and consumer files is reproducible, and a CI grep that old names return nothing is the only check that survives a merge.
6. Option D leaves `add_data_model` naming one buffer and `var_type` naming an element type into v3, which fails Decision 19 in the code itself.

Evidence: plans/minutes/owner-decisions.md rows 10, 19-21; taidl/taidl/antlr4/IDLV2.g4:6-8; taidl/taidl/accelerator.py:33-35; plans/terminology/conflicts.md #6, #7; plans/reviews/act-glossary/review.md AG5, AG10, AG11; plans/reviews/rulings/audit.md R2; https://openxla.org/stablehlo/spec (tensor types are written as element type plus shape).
Confidence: high.
Would change my mind: a measured consumer count showing a rename touches external user code the owner wants preserved, which would move that one name to Option B.

### Q7.11

Position: Option A (prose "test oracle generator" and "compiler backend generator"; keep repo, distribution, module, and function names; one glossary row per name; short form "kernel" only for `@kernel`), and keep `act-ecosystem` per R52.

1. The README spells the oracle generator three ways and the backend two ways; one prose name per generator is the fix, and the glossary already has "generator", "test oracle", "compiler backend" rows to hang it on.
2. Repo `act-oracle`, distribution `taidl-to`, module `taidl_to`, and function `generate_oracle` are what every pin, import, and tutorial names; Option B renames them and breaks every user and every pin for no semantic gain (Decision 10 makes it a breaking release).
3. A newcomer needs the mapping between the prose name and the four code names in one table; Option C gives no row, so the mapping lives in nobody's head.
4. "Kernel" is overloaded between `@kernel` (an oracle test program) and "tiled kernel" (an XLA sub-graph); for an XLA reader "kernel" means the latter, so prose must say "kernel function" and "tiled kernel" and leave the bare word to the decorator (R36).
5. `targets/<Name>/` uses a banned word, but the path is generated output that users' scripts read; renaming it is a v3 breaking change that fits the batch, so `accelerators/<name>/` goes into the rename table, not an exemption (against Option D).
6. `act-ecosystem` versus import `act` is the PyTorch pattern (distribution `torch` versus project PyTorch) and costs nothing once the README sentence is fixed.

Evidence: /home/user/act/README.md:33 and generator names; /home/user/act/pyproject.toml:6; act-oracle/taidl_to/generator.py:21; act-backend/act_backend/generator.py:79; act-oracle/taidl_to/generic/decorator.py:142; plans/glossary/terms.md rows "generator", "test oracle", "kernel function", "tiled kernel"; plans/minutes/rulings.md R36, R52, R53; plans/reviews/act-glossary/review.md AG9, AG12, AG21; https://openxla.org/xla/operation_semantics (XLA uses "kernel" for a fused computation on the device).
Confidence: high.
Would change my mind: a count of external scripts reading `targets/` that the owner wants kept working, which would move the path to Option D.

### Q7.12

Position: Option A (a rejected v3 construct reports its replacement with the clause id) plus the migration note from A23 (Option C's list); no converter and no deprecated aliases.

1. R44 already puts a clause id suffix on every rejection; adding "use X instead" to the same message is one string per removed construct and costs nothing at run time.
2. A converter (Option B) must parse v2 ISA descriptions, which are Python programs calling the API, so it is either a Python rewriting tool or a hand-maintained regex set; both exceed Decision 11 capacity for a one-time migration.
3. Deprecated aliases (Option D) keep two names per concept in the code for a release, which is the AG1 problem moved from docs into code, and leaves the old names visible to newcomers who copy them.
4. A migration note alone (Option C) makes the reader hunt; the error message is where the reader already is.
5. Both together give the PyTorch-style experience: the error tells you the new name, and the release note lists every change.
6. The tutorials already use the removed `qkv.generate_oracle()` form, so the first users of the message are the archived tutorial readers.

Evidence: plans/reviews/rulings/audit.md RA11; plans/minutes/rulings.md R2, R5, R44, R58, R60; plans/reviews/act-glossary/review.md AG14; tutorials/micro25/exercise1/solution/QKV.py:108; plans/minutes/owner-decisions.md rows 10, 11.
Confidence: high.
Would change my mind: more than a handful of external v2 ISA descriptions in use, which would justify the converter's cost.
