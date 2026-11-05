# Panel 7, round 1: Sonnet (lens: new user, an architecture researcher)

Assumptions used throughout:

- The newcomer reaches the project through the act README, then pastes a snippet and expects it to run.
- The newcomer reads the two papers before the code, so paper vocabulary is the vocabulary they arrive with.
- Decisions 6, 10, 11, 14, 19, 20, 21 bind; Decision 5 means every position is a plan change, not an edit.
- Paper text was read from the PDFs at the owner-paper URLs in process.md (pdftotext, line numbers below are of that dump).

### Q7.1

Position: Option A, with R54-b (archive on tags, drop from master once the v3 tutorial lands).

1. Decision 6 says no tutorial is current, so any plan item that says "the current tutorial" (A9, A10, A12) has no referent (AG13).
2. The newest tutorial still pins the v1.2 image, so a newcomer who runs it gets a toolchain that predates every current release (tutorials/pldi26/docker.sh:10-12).
3. The micro25 solution calls `qkv.generate_oracle()` as a method, while the API is the function `generate_oracle(acc)` (tutorials/micro25/exercise1/solution/QKV.py:108 against act/tests/test_end_to_end.py:11,71).
4. v3 renames break every pre-v3 tutorial again, so refreshing one (Option B) buys one release of life (R63, Decision 10).
5. Options C and D reorganize files whose code does not run on one release, so they move breakage around without removing it.
6. Option E leaves three trees that a newcomer cannot tell apart from live material; the README table links release tags such as v1.2 already, so the history is reachable without master (README.md Version Compatibility table).
7. For R54: a banner README per archived tree is cheap now; moving the trees to tags (R54-b) removes the ASCII exemption problem and the stale-tree problem together; R54-c changes an owner rule for a transient case.
8. The new tutorial is one tree built from the end-to-end test accelerator (act/tests/test_end_to_end.py:30-66), which is already tested code.

Evidence: plans/act/plan/tutorials.md; AG13, AG20; R54 audit row; the lines above.
Confidence: medium-high on A, medium on R54-b over R54-a.
Would change my mind: an owner statement that old events must stay browsable on master, which favors R54-a.

### Q7.2

Position: Option A (apply overrides, delete the superseded files), which contains Option C.

1. A newcomer who greps the plans for "buffer" finds two rulings per concept today, so the glossary fails its one job (AG1; plans/glossary/rulings.md rows "tensor buffer", "control register", "d0").
2. Option B adds "supersedes" lines, which makes a reader apply a two-step rule; Option E leaves the contradicting files in the tree, and a tool or a debater will still cite them.
3. Decision 14 settles precedence: papers first when precise, else StableHLO and MLIR; Decisions 19-21 override the rows they touch, so the table in the review is mechanical to apply.
4. The state of the art in compiler projects is one glossary generated into others, for example one table feeding docs; keeping a second folder as "history" is what git is for.
5. Option D (merge folders) is the same end state as A with more file moves; the saving is nil.
6. Pointers to /home/user/work/ must go, as that tree already differs from plans/ (AG19).

Evidence: plans/glossary/README.md lines 5-11; plans/glossary/rulings.md; plans/glossary/words-to-avoid.md; AG1, AG2, AG19, AG20.
External: the owner papers define "tensor buffers" and "control registers" (TAIDL Sec. 3.1, dump line 700-706); Decision 19 overrides both, so the glossary must say why papers differ.
Confidence: high.
Would change my mind: a reason to keep the terminology/ comparison as a reviewed artifact, which would justify marking, not deleting.

### Q7.3

Position: Option A; numbers: taidl 3.0, act-oracle 3.0, act-backend 2.0, act 3.0.

1. Decision 10 requires one coordinated release; S8 owns the tags of the siblings, and only act re-pins them, so act needs its own item (AG14).
2. check_versions.py already enforces that README rows, pins, image tags, and the newest summary line agree (tests/check_versions.py:36-60), so the release item is mostly filling the table the script checks.
3. The README already promises "an even faster simulator in act-oracle v3" (README.md:4), so act-oracle 3.0 is the number the owner has said in public.
4. act-backend is at 1.16, so its next major is 2.0; forcing 3.0 would skip a number with no meaning for users; the compatibility table is the single map of which versions go together.
5. Option D (independent numbering) is already the practice (act 2.4 pins taidl 2.15); the coordinated release is a table row, not a shared number.
6. Merging A16 and A17 into S8 and A23 removes duplicate owners of the release order and the script (AG17).

Evidence: plans/act/plan/versions-release.md; AG14, AG17; README.md:4 and table; act/pyproject.toml:6-7.
Confidence: medium (numbers are a value judgment; the item itself is high).
Would change my mind: the owner wanting the generators to read "3" everywhere for marketing.

### Q7.4

Position: Option B for v3 (Docker is the supported path), with the quickstart run inside the image as the release gate.

1. A plain-host install needs Python 3.14, cargo with crates.io access, CMake, a C++17 compiler, OR-Tools, nlohmann/json headers (act-backend/README.md:28-34), and the ANTLR tool in the image build (act/docker/Dockerfile.amd64:92-98).
2. `generate_backend` runs cargo at generation time (act-backend/README.md:30,52), so prebuilt wheels (Option D) cannot remove the Rust requirement; the compiler is built per accelerator.
3. The ecosystem image installs the act package from the pinned components, so the pip-from-pin path is already exercised inside the image build (docker/Dockerfile.amd64:118-126).
4. The README leads with a bare `pip install` (README.md:23-27), which a newcomer on a stock laptop will run and fail; Docker first removes that failure.
5. Option A (fresh-venv gate on a stock runner) would need the whole toolchain installed on the runner, which is the image again.
6. Option C costs a scheduled job per quarter of capacity for a path few will use; Decision 11 sizes the team at 3-4 engineers.
7. The gate: run the quickstart (Q7.5) in the built image, in addition to the existing end-to-end test (.github/workflows/test.yml:23-33).

Evidence: lines cited; AG15.
Confidence: medium.
Would change my mind: evidence that target users cannot run Docker (HPC clusters), which would make C worth its cost.

### Q7.5

Position: Option A, with the snippet sized by need (not fixed at 15 lines) and run by A20.

1. The current snippet has "# ... define ISA spec ..." so nothing in the README runs (README.md:36-45).
2. The only complete accelerator in act is the 37-line function in the end-to-end test, one buffer and three instructions (act/tests/test_end_to_end.py:30-66); a README quickstart is that function, shortened to load, gemm, store.
3. A snippet that CI imports but does not run can rot unseen; A20 must execute it (AG16).
4. Option C sends the newcomer to another repo before they see anything; Option B links a tutorial, which Decision 6 forbids presenting as current.
5. Option D leaves the newcomer with an outline, the exact failure AG16 names.
6. Docker and testing detail move below the quickstart or into docs/, because they serve maintainers (README.md:57-81).
7. Sub-question A3: delete the news note and old lines; keep the newest line, which check_versions.py requires below the table (tests/check_versions.py reads the table section and summary line).
8. Roadmap teasers (act-allo, act-e2e, README.md:4) are not facts about this release; a newcomer cannot use them.

Evidence: README.md:36-45, 77-81; test_end_to_end.py; AG16; R51.
Confidence: high on A, medium on the exact length.
Would change my mind: a quickstart that needs more than about 30 lines to be honest, which suggests a prepared example file in the repo linked from the README.

### Q7.6

Position: Option B: `add_buffer(...)` and `add_state(name, value)`; the singular is "state entry"; "state variable" stays banned.

1. Decision 19 fixes the nouns (buffers, state) and the glossary fixes no method names; today the methods are `add_data_model` and `add_initial_state` on a flat Accelerator (taidl/taidl/accelerator.py:33,44).
2. Options C and D add an object layer or a kind switch for two things that differ in arguments (dims and type against a value), so the flat names are simpler to teach.
3. "Variable" collides with tensor variables in the backend and with `var_name`/`var_type` (the ACT paper allocates "tensor variables", ACT dump line 1165; data_model.py:28,33).
4. The paper says "control registers"; Decision 19 replaced that noun, so a short sentence "a state entry is one named value" is the bridge for readers of the papers.
5. R14 adds typed kinds (descriptors, async groups), so the definition should read "a named value of a declared kind; an integer today", not "named integer" (AG8).
6. R27 should say "state" (AG8).

Evidence: AG3, AG8; accelerator.py:33,44,46; TAIDL Sec. 3.1 (dump line 733).
Confidence: medium (the singular is naming taste).
Would change my mind: a count showing "state variable" in the owner's own writing.

### Q7.7

Position: Option B/D hybrid: identifier `hbm`; prose defines "global memory (hbm), the built-in buffer d0" once; thereafter write `hbm`.

1. Both owner papers call d0 "global memory" and use HBM as the alias: "We refer to this global memory as HBM ... hereafter" (TAIDL Sec. 3.1, dump line 773-775); "global memory (HBM) ... denoted d0" (ACT Sec. 3.2, dump line 607).
2. Decision 14 puts the papers first when precise, and Decision 19 keeps the identifier `hbm`; together they say the prose may carry both words with a defined relation.
3. Option A bans "global memory", which the papers define, so a newcomer reading a paper then the spec meets a banned term; Option C substitutes a third word the papers never use.
4. MLIR gpu names the same thing `global` address space: "memory that resides in the GPU's global memory" (https://mlir.llvm.org/docs/Dialects/GPU/), so the GPU scope (Decision 15) already says global.
5. Ban "off-chip memory", "DRAM", "main memory": none is a paper term and each names something else on some hosts.
6. GPU sub-question: device memory is `hbm`; shared memory and register fragments are buffers; the MLIR `workgroup` space is the analogue of the former.
7. Drop the A4 rewrite of "HBM" to "off-chip memory" (AG4).

Evidence: act-oracle/taidl_to/generic/decorator.py:142; act-backend/act_backend/instruction_info.py:71; AG4.
Confidence: medium.
Would change my mind: owner intent that "HBM" must never appear as prose, which makes Option A the only consistent one.

### Q7.8

Position: Option A: add "fault"; "reject" only for generation and kernel compile time; delete "undefined behavior" and "implementation-defined"; rename "refus" to "reject" in public names and messages.

1. Decision 16 forbids undefined behavior, so the term can only appear as a banned word (AG6).
2. A newcomer needs two stage words: one error before anything runs, one during a run; "fault" is the established word for a run-time trap in CPU architecture, and R55 defines it with the clause id.
3. Option D ("error" for both) drops that stage distinction; Option C leaves the stage to the sentence, which is the current ambiguity.
4. StableHLO leaves cases open ("the behavior is undefined" for cholesky; "implementation-defined" for tensor storage order, https://openxla.org/stablehlo/spec), so a reader arriving from StableHLO will look for what v3 does instead; the glossary must say "fault" there.
5. Decision 16 allows no open case, so "implementation-defined" has no legal use; coverage status (`supported`, `partial`, `rejected`) needs no such word (Option B is unnecessary).
6. The same mismatch exists in code: the docstring says "refused" while the glossary says "reject" (act-oracle/taidl_to/generator.py:23-24); the batched major release is the cheap time to rename.
7. Rename the file `test_refusals.py` and the messages; counts of 29, 214, 85 occurrences are mechanical with a grep list (AG6).

Evidence: R55; taidl/taidl/instruction.py:116,145,451; AG6.
Confidence: medium-high.
Would change my mind: an owner preference for "error" because "fault" suggests hardware.

### Q7.9

Position: Option A: keep `set_inplace` and define it per Decision 17; delete the placement-mark row.

1. Decision 17's text is "the output reuses the marked input's storage", which presupposes a mark; R5 removes the mark, so R5 conflicts with a binding decision (AG7).
2. A user states the intent once with `instr.set_inplace([True])`, as in the softmax example (act-backend/examples/mini_trn/mini_trn.py:97-99), and can read it back.
3. Option B derives aliasing from equal region text, so two spellings of one region silently change behavior and partial overlap stays open (audit R5, RA2).
4. A rule that depends on spelling is hard to explain, which Decision 16 asks to avoid.
5. The existing validation (`set_inplace needs one flag per input`, taidl/taidl/instruction.py:163-170) shows the mark is already checked at the edge.
6. Option C is acceptable only if the owner wants to keep R5 alive; it delays a row that Decision 17 already decides.
7. The placement mark row goes either way because R58 removes the marks.

Evidence: Decision 17; R5, R58; AG7, AG18; test_extraction.py:507.
Confidence: medium.
Would change my mind: the semantics panel showing that text identity is complete after RA2.

### Q7.10

Position: Option B: rename the names a user types; leave grammar rule and token names.

1. A newcomer types `add_instruction(name, computation_attr, addressing_attr, update=, constraints=)` and `add_data_model(name, dimensions, unit_dim, var_type)` (taidl/taidl/accelerator.py:33-35,60-65); these must match the docs word for word.
2. Grammar rule and token names (`instruction`, `OPERATION`) are not visible in an ISA description, so renaming them gives no newcomer benefit (taidl/taidl/antlr4/IDLV2.g4:6,8,28,45; IDLV2Expr.g4:8,11).
3. Rename: `add_buffer`, `add_state`, `computational_attrs`, `addressing_attrs`, `element_type`, `access_dims`, `base_element_dims`.
4. Keep `update`, `update_asts`, `update_text`: the only reason to rename them was "control register", which Decision 19 removed (AG5).
5. `Region` stays and `@s.` stays (Decisions 20, 21), so the list for them is empty.
6. Sub-question: yes to a generated table of old name, new name, and consumer files, per repo, and a CI grep that old names return nothing; cite symbols, not line numbers, since line cites drift (AG10, AG11).
7. Consumers outside the front end include act's umbrella and tests (act/act/taidl/__init__.py, act/tests/test_end_to_end.py:29) which a front-end-only list misses.

Evidence: AG5, AG10, AG11; audit R2 row.
Confidence: medium.
Would change my mind: finding that ANTLR error messages print rule names to users, which would justify renaming them.

### Q7.11

Position: Option A: prose "test oracle generator" and "compiler backend generator"; keep repo, distribution, and function names; keep `act-ecosystem` and fix the README sentence; rename `targets/` to `accelerators/` in the v3 batch.

1. The newcomer's code uses only the umbrella: `from act.generators import generate_backend, generate_oracle` and `from act.taidl import Accelerator` (act/tests/test_end_to_end.py:11-12), so inner names `taidl_to` and `act-oracle` stay hidden and renaming them (Option B) buys little.
2. The README uses at least five names for two tools: "TAIDL-TO (Test Oracle) Generator", "Test Oracle generator", "ACT-Backend Generator", "Compiler Backend Generator", plus the repo names (README.md:15-20); one canonical prose pair ends this (AG12).
3. The TAIDL paper defines "test oracle" and uses it throughout (title; dump lines 258-259 cite Sail and software testing), so the prose term is paper-backed.
4. The distribution is `act-ecosystem` (act/pyproject.toml:6) while the import name is `act`; a rename to `act` breaks every pin for no user gain, so keep and fix the sentence (R52).
5. `targets/<Name>/` holds the oracle and the backend crate of one accelerator (act-oracle/taidl_to/generator.py:22,31; act-backend README.md:48), so `accelerators/<name>/` states what the path is; the cost is one path change in the batched release (low confidence on this sub-point).
6. "kernel" alone means the `@kernel` decorator (decorator.py:142); prose says "kernel function" and "tiled kernel" (AG9, AG21).

Evidence: AG9, AG12, AG21; R36, R52, R53.
Confidence: medium (low on targets/).
Would change my mind: a user-study or support evidence that people confuse taidl-to with act-oracle at install time.

### Q7.12

Position: Option A plus a short migration note; renamed v2 methods raise a stub error naming the replacement; no converter, no working aliases.

1. ISA descriptions are Python scripts that call `add_*` methods, so a failing call is the first thing a v2 user sees; the error text is the migration path they read (R44).
2. A rejected v3 construct reports its replacement and the clause id (RA11 proposal), which costs a message table, not a tool.
3. A converter (Option B) must rewrite arbitrary Python; the tutorials and examples show kwargs and positional calls mixed, so an AST rewrite is a project of its own, beyond a 3-4 engineer team that is also shipping v3 (Decision 11).
4. Working aliases (Option D) keep two vocabularies alive and contradict the batched breaking release (Decision 10).
5. A stub that only raises "add_data_model was renamed add_buffer" is not an alias; it costs a line per rename and removes the Python `AttributeError` dead end.
6. The migration note (Option C) is the list that A23 owns (Q7.3), generated from the Q7.10 table so it cannot drift.
7. The tutorials already show a version break: `from taidl import Accelerator` against `from act.taidl import Accelerator` (plans/act/findings.md tutorials rows; act/tests/test_end_to_end.py:12).

Evidence: audit RA11; R2, R5, R44, R58, R60; AG14.
Confidence: medium.
Would change my mind: evidence of many external v2 ISA descriptions in the wild, which would justify a converter.
