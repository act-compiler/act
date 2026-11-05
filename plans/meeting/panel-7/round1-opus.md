# Panel 7 round 1: Opus (docs and terminology owner)
Assumptions:
- Owner decisions 6, 10, 14, 16, 17, 19, 20, 21 bind; every answer below sits inside them.
- "One term per concept" means one prose term, one identifier spelling, and one defining file per concept.

### Q7.1
Position: Option A for content plus Option C for layout; sub-question R54-b.
1. Decision 6 says no tutorial is current, so A9, A10, and A12 test or edit an object that does not exist.
2. The pldi26 tree pins image v1.2 and the micro25 solution calls `generate_oracle()` as a method current taidl lacks, so a refresh is a rewrite, not a check.
3. v3 renames (R2, R5, R58, R60) break any refreshed pre-v3 tutorial a second time, so Option B pays for the migration twice.
4. One tutorial written on the v3 release candidate absorbs A12 (terms), A14 (titles), and A13 (exercise 4) as one item.
5. Option C keeps event-specific setup on `tutorials/events/<event>.md` and leaves past trees on their tags, so master holds one current copy.
6. Option D needs all three exercise sets to run on one release, which item 2 shows they do not.
7. Option E leaves Decision 10's release without a newcomer path and keeps three stale trees on master.
8. R54-a and R54-c each amend the ASCII rule, which allows non-ASCII in binary assets only; R54-b needs no rule change once archived trees leave master at the v3 tutorial.
- Evidence, step 1: owner-decisions.md row 6; act/plan/tutorials.md:10,11,13.
- Evidence, step 2: act tutorials/pldi26/docker.sh:10-12; tutorials/micro25/exercise1/solution/QKV.py:108.
- Evidence, step 3: minutes/rulings.md R63 (IDLV2.g4:6 rejects the tutorial syntax); R2, R5, R58, R60.
- Evidence, step 5-6: act/findings.md:53-54.
- Evidence, step 8: engineering.md "ASCII only ... Binary assets are the only exception"; 57 files under act/tutorials contain non-ASCII (grep -P count).
- Confidence: high on A; medium on C over D.
- Would change my mind: evidence that one archived tutorial already runs on v2.4 with only factual edits.

### Q7.2
Position: Option A, with words-to-avoid generated from the terms.md banned column (Option C part).
1. A concept with two live rules has no term; the README override block leaves at least eight such concepts.
2. Authority must be one file: terms.md defines, owner decisions override it only by being written into it.
3. words-to-avoid.md duplicates the banned column, so it drifts; generating it removes the second source.
4. rulings.md records how terms were reached, which engineering.md forbids in a living doc; only reasons still in force stay.
5. terminology/, the spec-draft glossary, backend/naming.md, and oracle/plan/naming.md each contradict terms.md; each is deleted or reduced to a pointer.
6. Option B and Option E keep the contradictions and add a "supersedes" layer, which is history.
7. Option D is a reshuffle that keeps two rules if not applied first.
8. Pointers to /home/user/work/ name a second tree; every pointer goes to plans/glossary/terms.md.
9. A CI grep of the banned column over docs keeps the single vocabulary true after v3.
- Evidence, step 1: glossary/README.md override list; reviews/act-glossary/review.md AG1.
- Evidence, step 3: review.md AG20; glossary/words-to-avoid.md.
- Evidence, step 4: engineering.md Docs discipline ("A doc records no history").
- Evidence, step 5: review.md AG2.
- Evidence, step 8: review.md AG19.
- Evidence, external: StableHLO keeps one normative spec page for its terms (openxla.org/stablehlo/spec), not a set of competing glossaries.
- Confidence: high.
- Would change my mind: the owner wanting rulings.md kept as a design record outside the living docs.

### Q7.3
Position: Option A; v3 numbers taidl 3.0, taidl-to 3.0, act-backend 3.0, act 3.0, image tag `v3.0`.
1. S8 depends on act steps no A item owns, so the release train has an unowned leg; A23 owns it.
2. A16 and S8 both define the release order, and A17 and S8 both extend check_versions.py; one owner per step means merging A16 and A17 into S8 and A23.
3. Decision 10 says "TAIDL v3 and matching generator releases"; "matching" reads most simply as the same major number.
4. Today ACT v2.4 maps to TAIDL v2.15, TAIDL-TO v2.11, ACT-Backend v1.16, so a reader needs the compatibility table to say what "v2" means.
5. With one major number, "v3" names one concept across four repos and papers, and the table becomes a check rather than a lookup.
6. Option D keeps four numbering schemes, which contradicts the plain reading of "matching" in step 3.
- Evidence, step 1: review.md AG14; act/plan/versions-release.md A15-A18.
- Evidence, step 2: review.md AG17.
- Evidence, step 3: owner-decisions.md row 10.
- Evidence, step 4: act README.md:77; pyproject.toml dependencies (taidl v2.15, act-oracle v2.11, act-backend v1.16).
- Evidence, external: torchvision numbers its releases apart from torch and its README carries a torch-to-torchvision table (github.com/pytorch/vision), the lookup step 5 removes.
- Confidence: high on A23; medium on the numbers.
- Would change my mind: an owner reading of "matching" as compatible-but-independent, or a published consumer pinning act-backend by major number.

### Q7.4
Position: Option B for v3 (Docker is the supported path, tested in CI), with the plain-host requirements documented as unsupported; Option D deferred to the owner.
1. "Supported" must mean "tested"; otherwise the word names two concepts.
2. CI builds and tests only the Docker image.
3. The README's first install line is `pip install git+...@v2.4`, an untested path.
4. The pip path needs Python 3.14 and git-pinned sources that build Rust and ANTLR parts on the host.
5. Option A adds a release gate whose toolchain set must be maintained per runner; Decision 11 caps capacity.
6. Option D needs wheel builds, which is a new build dependency and so needs the owner's approval.
7. Option C tests on a schedule, which leaves the release itself unverified.
- Evidence, step 2: act .github/workflows/test.yml:23-33.
- Evidence, step 3: act README.md:29.
- Evidence, step 4: act pyproject.toml:9 (`requires-python = ">=3.14"`), :15-19 git dependencies; review.md AG15.
- Evidence, step 6: engineering.md "Don't add a dependency without asking".
- Confidence: medium.
- Would change my mind: evidence that most users install with pip on a plain host, which would make Option A worth its cost.

### Q7.5
Position: Option A without the tutorial link until the v3 tutorial exists; sub-question A3 reduces the history lines to the newest one.
1. A newcomer needs one runnable example; the current snippet is an outline that does not run.
2. A complete minimal accelerator run by A20 makes the README a tested artifact, not prose.
3. The README must link the spec and the coverage tables, the two documents a user reads next.
4. Decision 6 forbids linking a tutorial as current, so the tutorial link waits for the Q7.1 tutorial.
5. Option C sends a newcomer to a second repo before the first success.
6. check_versions.py requires a sentence below the table for the newest release, so A3 keeps exactly that line and deletes the rest.
7. The older "moves the ecosystem" lines are release history, which the README must not carry.
- Evidence, step 1: act README.md:36-45 (`# ... define ISA spec ...`).
- Evidence, step 2: act/plan/tests-docker.md A20; review.md AG16.
- Evidence, step 6: act tests/check_versions.py:61-65.
- Evidence, step 7: act README.md:77-81; engineering.md Docs discipline.
- Evidence, external: NKI docs open with a runnable kernel before the API reference (awsdocs-neuron nki.language pages).
- Confidence: high.
- Would change my mind: a 15-line accelerator proving impossible without hiding necessary parts.

### Q7.6
Position: Option A: `add_buffer(name, access_dims, base_element_dims, element_type)` and `add_state(name, value)`; singular "state variable"; define state for v3 to cover declared kinds.
1. Decision 19 names the units "buffers" and "state", so the methods are `add_buffer` and `add_state`.
2. PyTorch uses the same pair: `register_buffer` adds a buffer, and "state" is the collective noun.
3. "State" is a mass noun, so prose needs a count noun for one entry.
4. The code already says "machine state variable" for one entry, so "state variable" is the term in use, not a new one.
5. "State entry" (Option B) is a new word with no source.
6. Option C and Option D add object structure no task needs.
7. R14 puts typed state kinds in v3, so the v3 definition reads "a named value of a declared kind, an integer by default".
- Evidence, step 1: owner-decisions.md row 19; taidl/taidl/accelerator.py:33,46.
- Evidence, step 2: docs.pytorch.org torch.nn.Module.register_buffer ("Add a buffer to the module").
- Evidence, step 4: taidl/taidl/accelerator.py:47 docstring.
- Evidence, step 7: minutes/rulings.md R14; review.md AG8.
- Confidence: high on names; medium on the singular.
- Would change my mind: the owner saying "variable" misleads because state is read-only in some instruction forms.

### Q7.7
Position: Option A: the term is `hbm`, the built-in buffer `d0`; uppercase HBM only for a concrete chip's memory; ban "global memory", "off-chip memory", "DRAM", "main memory".
1. Decision 19 keeps the identifier `hbm`, and the kernel parameter and Rust variant already spell it.
2. Decision 14 puts papers first only when precise; the ACT paper calls `d0` both "global memory (HBM)" and "the HBM (d0)", so it is not precise.
3. With no precise paper term, one identifier and one prose term spelled the same removes a mapping step.
4. NKI names the same level `hbm` in its API, so the term matches an accelerator vendor's API.
5. MLIR gpu and CUDA say "global memory" for GPU memory, so in GPU scope device memory is `hbm`, and shared memory and register fragments are buffers.
6. Option C uses "off-chip memory", which nothing in the code or the papers uses for `d0`.
- Evidence, step 1: act-oracle/taidl_to/generic/decorator.py:142; act-backend/act_backend/instruction_info.py:69-71.
- Evidence, step 2: ACT paper text, "1-dimensional global memory (HBM) ... denoted d0" and "the HBM (d0)".
- Evidence, step 4: nki.language memory hierarchy, "hbm: HBM - Alias of private_hbm".
- Evidence, step 5: mlir.llvm.org/docs/Dialects/GPU, "global memory is memory that resides in the GPU's global memory".
- Confidence: medium.
- Would change my mind: the owner reading "global memory" as the papers' precise term under Decision 14.

### Q7.8
Position: Option A: "fault" at run time, "reject" at generation and kernel compile time, no "undefined behavior", no "implementation-defined"; rename "refus" to "reject" in identifiers, messages, and test file names in v3.
1. Decision 16 forbids undefined behavior, so the glossary row defines a concept that cannot occur.
2. R55 names the defined run-time error a "fault", so run time needs no second word.
3. StableHLO uses "the behavior is undefined" and "implementation-defined"; v3 departs from it on purpose, so both words go to the banned column.
4. Coverage status is a different concept (a generator not supporting an op), so Option B would reuse one word for two concepts; coverage says "unsupported".
5. "Refuse" and "reject" name one concept; one term means one spelling in code and messages too.
6. Messages are user-visible, so a prose-only rename leaves users reading the banned word.
7. v3 batches breaking renames, so renaming `test_refusals.py` costs nothing extra.
- Evidence, step 1: owner-decisions.md row 16.
- Evidence, step 2: minutes/rulings.md R55.
- Evidence, step 3: openxla.org/stablehlo/spec (cholesky "the behavior is undefined"; bitcast_convert "implementation-defined").
- Evidence, step 5: glossary/rulings.md:63; "refus" counts 27, 215, 73 in taidl, act-oracle, act-backend (.py, .md, .rs); taidl/taidl/instruction.py:116,145,451.
- Evidence, step 7: owner-decisions.md row 10; act-oracle and act-backend tests/test_refusals.py.
- Confidence: high.
- Would change my mind: evidence that "fault" collides with a hardware fault term in the papers.

### Q7.9
Position: Option C: define "in-place mark" by Decision 17 now and leave the identifier column pending R5 and RA2; delete the placement-mark row now.
1. Decision 17 fixes the meaning: the output reuses the marked input's storage.
2. Whether `set_inplace` exists is a language question for panels 1 and 2, not a vocabulary one.
3. If R5 stands, nothing is marked, so the term becomes "in-place alias" and "mark" is wrong; the row cannot be final first.
4. R58 removes `set_apart` and `set_no_partial_overlap`, so the placement-mark row names no construct.
5. Option D keeps a row for removed constructs, which states what is not.
- Evidence, step 1: owner-decisions.md row 17.
- Evidence, step 3: minutes/rulings.md R5; reviews/rulings/audit.md R5, RA2; act-backend/tests/test_extraction.py:507.
- Evidence, step 4: minutes/rulings.md R58.
- Evidence, external: mlir.llvm.org/docs/Bufferization, an op runs "in-place" by "reusing the buffer allocated for this destination input", the meaning Decision 17 takes.
- Confidence: medium.
- Would change my mind: a panel 1 or 2 ruling on R5 before this panel decides.

### Q7.10
Position: Option A (R2 with AG5), specified as a generated per-repo rename table citing symbols, checked by a CI grep for old names.
1. Each listed rename aligns an identifier with a term already ruled (`access_dims`, `base_element_dims`, `element_type`, `computational_attrs`, `addressing_attrs`).
2. The `update` to `assignments` rename existed only for "control register", which Decision 19 removes, so `update` stays.
3. Grammar rule and token names surface in parser classes and in the spec's grammar listing, so Option B leaves two names for one construct.
4. Option D leaves `add_data_model`, `unit_dim`, `var_type` in v3, a second breaking release later against Decision 10.
5. Line cites already drift; a table generated by grep over every consumer repo, citing symbols, stays true.
6. A CI grep for old names after v3 is the check that the vocabulary holds.
- Evidence, step 1: taidl/taidl/accelerator.py:33-35,62-65; glossary/terms.md identifier column.
- Evidence, step 2: review.md AG5; IDLV2Expr.g4:8,11.
- Evidence, step 3: IDLV2.g4:6,8,28,45; terminology/conflicts.md #6, #7.
- Evidence, step 5: review.md AG10, AG11.
- Evidence, external: the StableHLO spec spells each op as its dialect does (`add` and `stablehlo.add`, openxla.org/stablehlo/spec), one name per construct.
- Confidence: medium.
- Would change my mind: a measured cost of the grammar rename in generated parser consumers above the plan's capacity.

### Q7.11
Position: Option A; keep `act-ecosystem` (R52); rename `targets/<name>/` to `accelerators/<name>/` in v3.
1. The README gives the oracle generator three names and the backend two; prose needs one each.
2. The ACT paper calls ACT "a compiler backend generator", and the TAIDL paper title uses "test oracles", so the prose terms come from the papers per Decision 14.
3. Repo, distribution, module, and function names are spellings users type; renaming them breaks pins for no new meaning.
4. One glossary row per name maps each spelling to its concept, so the spellings stop looking like rival terms.
5. "kernel" short form only for `@kernel` and its call separates it from "tiled kernel".
6. The glossary bans "target", and the generated path is user-visible; v3 batches breaking changes, so the path moves with them.
- Evidence, step 1: act README.md:18-20.
- Evidence, step 2: ACT paper text, "we introduce ACT, a compiler backend generator"; TAIDL paper title (2025-micro-taidl.pdf).
- Evidence, step 3: act pyproject.toml:6,15-19; act-oracle/taidl_to/generator.py:21; minutes/rulings.md R52.
- Evidence, step 5: review.md AG9; minutes/rulings.md R36.
- Evidence, step 6: act-oracle/taidl_to/generator.py:22,31; act-backend/act_backend/generator.py:95; glossary/rulings.md.
- Confidence: high on prose; medium on the `targets/` rename.
- Would change my mind: external scripts depending on `targets/` paths.

### Q7.12
Position: Option A plus a migration note generated from the Q7.10 rename table; no converter, no aliases.
1. A rejection that names the replacement and clause id teaches at the point of failure.
2. The migration note lists the same rows, so one table feeds both and they cannot disagree.
3. Deprecated aliases keep two names per concept for a release, against one term per concept and against one batched break.
4. Python's deprecation policy needs warnings across releases; v3 is a single declared break, so that path does not apply.
5. A converter is a new tool to maintain for a small user base; Decision 11 caps capacity.
- Evidence, step 1: minutes/rulings.md R44; reviews/rulings/audit.md RA11.
- Evidence, step 2: review.md AG14.
- Evidence, step 4: peps.python.org/pep-0387 (warn, wait two minor versions, then remove).
- Evidence, step 5: owner-decisions.md row 11.
- Confidence: high.
- Would change my mind: a count of external v2 ISA descriptions large enough that hand migration costs more than a converter.
