# Backend review: architect

Independent view of act-backend, formed from the code before reading the plan, then a step-by-step challenge of `work/backend/plan/`.
Code paths are relative to `act-backend/act_backend/`; `generic/src/` is shortened to `src/`; plan paths are relative to `work/backend/plan/`.

## What the code is today

- Four stages in one process: HLO reader (`src/isel/initializer/lib.rs:36`), egg saturation (`src/pipeline.rs:491-514`), two extractors (`src/isel/extractor/lib.rs:30,41`), then the C++ allocator and the C cost model per candidate (`src/pipeline.rs:251-263`).
- Three in-process text round trips: candidate to `<n>.pii` and back (`src/ir/pii.rs:75-93`, `cpp/malloc/src/act_malloc.cc:52-60`), kernel facts to `metadata.json` and back (`src/pipeline.rs:457-469`, `act_malloc.cc:147-153`), assembly to `<n>.py` and back into the cost library (`src/cost/c.rs:7-15`).
- The two parsers are not peers: the Rust reader parses a real input language (HLO); `cpp/malloc/src/parser.cc:19` parses only the text Rust printed one call earlier.
- `parser.cc` is not only parsing: `hold` and `hold_tensors` (`parser.cc:146,335`) check that each tensor fits its storage, which is real allocator logic.
- Payloads are strings: `OpDot(String, ..)` and friends (`src/ir/egraph.rs:10-28`) are re-split at every precondition by `payload_ints` (`egraph.rs:63-94`).
- `Language::matches` compares only the discriminant (`egraph.rs:163-165`), so a pattern payload such as `convert_s32` is ignored by egg; rule 6 is enforced only by `precond_6` (`src/isel/rewrites/ir2ir_rewrites.rs:514-526`).
- Rewrites are text compiled in with `include_str!` (`src/isel/rewrites/lib.rs:7-11`), parsed at start-up, and bound to three functions by a string key (`src/isel/rewrites/applier.rs:15-56`); every name is known at generation time.
- The custom applier re-matches the LHS to recover e-nodes egg does not bind (`applier.rs:106-113,168-215`), with a cartesian product per class (`applier.rs:203`).
- Shape is not e-class analysis: `make` returns an empty shape (`egraph.rs:396-401`); the reader sets shapes by hand (17 `set_analysis_data` calls in `initializer/nodes.rs`) and every rewrite carries a `set_shapes_N` (`applier.rs:43-56,127`).
- `merge` treats an empty shape as unset and never copies a set shape into an unset one (`egraph.rs:403-415`).
- Four IR-to-IR rules have a bare-variable LHS and fire on every class every iteration (`src/isel/rewrites/ir2ir_rewrites.txt:3,4,5,8`); growth is bounded only by `NODE_LIMIT` 10000 and a 1000 ms `TIME_LIMIT` (`src/lib.rs:18-22`).
- Each extraction pass clones the whole e-graph (`src/pipeline.rs:234-240`) because both extractors add `DetectedConst` nodes (`src/isel/extractor/slow.rs:21-24`), twice per pass (fast then slow, `extractor/lib.rs:30,41`).
- The fast extractor is a cost-blind first-feasible DFS that clones its whole memo per tried e-node (`src/isel/extractor/fast.rs:88,109`).
- The slow extractor enumerates every tree up to `limit` nodes by cartesian product (`slow.rs:117`), then value-numbers; `limit` climbs from 10 to 15 per pass (`src/lib.rs:27-28`, `src/pipeline.rs:502`).
- The default cost is the count of `api.` lines in the assembly (`cpp/cost/count.cc:1-22`), so a node-count proxy in the extractor would rank candidates almost as the cost model does.
- The search is wall-clock bounded and stops on "no improvement for 10x the time of the last best" (`src/pipeline.rs:155-180`, `src/lib.rs:30`), so the chosen candidate can depend on machine speed.
- The goldens are the simulated outputs of the compiled kernel, not its assembly (`README.md:109`), so equal goldens do not mean an equal selection.
- Generators re-walk the TAIDL ANTLR tree in each emitter (`isel_ir2isa_rewrites_txt_generator.py:4,56-65`, `isel_ir2isa_rewrites_rs_generator.py:5-7`, `malloc_instructions_h_generator.py:6-7`), and the front end checks no shapes (`taidl/taidl/operations.py:3-4`).

## Good parts to keep

- Buffer-aware selection: an e-class is chosen per data model, with a child buffer rule (`fast.rs:50-54,99-106`, `fast.rs:200-208`); any new extractor must keep the (class, buffer) state.
- Drop-with-reason everywhere: allocator reasons (`act_malloc.cc:62-63`, `src/malloc/cpp.rs:43-48`) and the no-assembly message naming the first drop (`src/pipeline.rs:207-219`).
- `merge` as a loud consistency check (`egraph.rs:406-411`).
- The OR-Tools allocator and emitter in C++, and the one-function cost ABI (`src/cost/c.rs:7`).
- Precise generation-time rejections before anything is written (`generator.py:124-131`) and checked 64-bit arithmetic (`operand_sizes.py`, `expression_writer.py`).
- The reader's named errors, pinned by `tests/test_reader.py`.

## Target architecture (formed before reading the plan)

- One typed IR: `TensorOp` variants carry typed attributes (dims as `Vec<i64>` or `SmallVec`, `Dtype`), never strings; text exists only in `Display` for `--log`.
- `Analysis::make` infers shape and element type from the e-node and its children, with shape as `Option`; `merge` asserts agreement; constant-ness is analysis data too.
- The shape rules live once, per operation, in Rust; the reader only checks the HLO annotation against the inferred shape; the TAIDL front end checks semantics lines against the same rule table (one spec, two implementations tested against each other).
- Rewrites are Rust values built at start-up from generated constructors: IR-to-IR rules are hand-written, named Rust; IR-to-ISA rules are one generic applier over a generated per-instruction table (pattern, payload conditions as small expression trees, result attributes).
- Rule scheduling: IR-to-IR rules with a bare-variable LHS run under egg's `BackoffScheduler` or are made conditional, so saturation is not decided by the node limit.
- One extractor over `&EGraph`: a cost-guided k-best enumeration on (class, buffer) with a node-count proxy, feeding the allocator; the real cost model stays the final judge.
- One typed FFI call per candidate: a flat `#[repr(C)]` node array plus kernel facts; C++ keeps tensor creation, `hold_tensors`, solver, emitter; files are written only under `--log`.
- `search` as a library function returning `Result`, with a deterministic mode bounded by iterations and candidates rather than wall clock; `main.rs` only parses flags.
- Generators consume a typed semantics IR from the TAIDL front end, not ANTLR contexts; Rust output is mostly data, so few templates remain and `rustfmt` formats the rest.
- Ruling 16 (no undefined behavior): spec-level shape and type errors are front-end errors, so the backend trusts the semantics IR and drops re-checks; rewrites may assume defined run-time errors never occur.
- Ruling 17 (in-place is true aliasing): an in-place output and its marked input get one storage, as an equality constraint in the solver, with liveness guaranteed by the extractor.
- Today the allocator only lets the two overlap, skipping their interference edge (`cpp/malloc/src/constraint.cc:214-230`), and the extractor patches liveness after the fact (`src/isel/extractor/inplace_fix.rs:9`).

## Plan step by step

| Plan step | Verdict | Reason | Proposed change |
| --- | --- | --- | --- |
| 1: B3 golden harness (`overhaul.md:47`, `architecture.md:89-97`) | agree, insufficient | Goldens are kernel outputs (`README.md:109`); selection is wall-clock bounded (`src/pipeline.rs:155-180`), so "goldens identical" neither proves an unchanged selection nor is stable under refactors that change speed (B9, B10). | Add a deterministic search mode (iteration and candidate budgets, no clock) and gate structural steps on identical chosen assembly and `--log` candidate files, plus goldens. |
| 2: B1, B28 moves and renames (`overhaul.md:48`) | agree | Cheap, one commit, best done before parallel work starts. | Keep first; land before engineers split. |
| 3: B13 instruction model (`overhaul.md:49`, `generators.md:24-32`) | agree, re-scope | The model is built from ANTLR contexts today (`isel_ir2isa_rewrites_txt_generator.py:56-65`); building it again from ANTLR keeps the coupling to the grammar. | Build the model from a typed semantics IR exported by the TAIDL front end; joint item with the language track. |
| 4: B6, B8 typed rewrites (`overhaul.md:50`, `isel.md:45-53`) | agree on goal, disagree on order | B6 before B7 regenerates `set_shapes_N` in a new form only to delete it in step 8; B6 keeps string payloads, so `payload_ints` (`egraph.rs:63-94`) survives. | Typed payloads first, then B7, then B6 with only a condition and a result-attribute function per rule. |
| 4: B6 caveat on `dot_?` wildcards (`isel.md:50`) | disagree | Every op word already matches any payload because `matches` compares discriminants (`egraph.rs:163-165`); the gap is the reverse, a concrete payload is silently ignored (rule 6, `ir2ir_rewrites.txt:7`). | State the semantics: payloads never match in patterns; conditions carry them; add a test for rule 6. |
| 5: B2, B4, B25 typed FFI (`overhaul.md:51`, `architecture.md:79-87`) | agree, incomplete | Covers `.pii` only; `metadata.json` is a second text round trip (`act_malloc.cc:147-153`); `hold_tensors` (`parser.cc:335`) is allocator logic, not parsing. | Pass kernel facts in the same call; move `hold`/`hold_tensors` beside `create_instruction`, not into the deleted parser. |
| 6: B10 one extractor module (`overhaul.md:52`, `extractor.md:95-103`) | partial | Merging is right, but "preserve candidate order" freezes a cost-blind DFS (`fast.rs:88-128`) and an exponential enumeration (`slow.rs:117`); the default cost is an instruction count (`count.cc:12-22`), so a proxy cost exists. | Merge (S), then replace both strategies with a cost-guided k-best extractor on (class, buffer) as a measured behavioral step. |
| 7: B12 formatter instead of twins (`overhaul.md:53`, `generators.md:14-22`) | over-engineered now | It reformats Rust emitters that a data-driven applier deletes; doing B6-as-code, then B12, then the Q2 interpreter experiment (`decisions.md:102-111`) rewrites the same generator three times. | Decide data-vs-code now (recommend data for IR-to-ISA rules); apply B12 only to emitters that remain. |
| 7: B14 checks build the model (`generators.md:34-42`) | agree | Removes the check-emitter import cycle (`spec_checks.py:9-18`). | Keep, after the re-scoped B13. |
| 8: B7, B9, B26 shapes in the analysis (`overhaul.md:54`, `isel.md:55-63`) | agree, move to Q1 | It is the change that removes most of what the owner finds too complicated (`set_shapes_N`, reader `set_analysis_data`); `merge` makes failures loud (`egraph.rs:406-411`); with a deterministic gate its risk is measurable. | Do right after typed payloads; one engineer owns typed payloads, B7, B9 in sequence. |
| 8: B9 no e-graph clone (`isel.md:75-83`) | agree | Constant-ness belongs in `make`; `is_const` already exists in `TensorInfo` (`egraph.rs:370-374`). | Keep, immediately after B7, S-sized. |
| 9: B21, B22, B24, B27 rejections (`overhaul.md:55`) | agree | Named generation-time rejections are the defined errors ruling 16 asks for. | Keep; the shape rules B23 needs come from the B7 table (row below). |
| 9: B23 line agreement (`defects.md:38-47`) | re-home | Under ruling 16 an ill-typed semantics is a defined front-end error, not a backend refusal; today it is silently never selected (`bugs.md` D3). | Move B23 to the language track; the backend asserts the semantics IR is typed. |
| 10: B11 in-place copies (`extractor.md:105-113`) | agree, upgrade | Ruling 17 makes aliasing a hard constraint, but `constraint.cc:214-230` only permits overlap, so an output may still land elsewhere. | Q1 behavioral item: equality constraint on offsets in the solver, plus B11 copies so the marked input is dead after the instruction. |
| 10: B15 to B20 coverage (`overhaul.md:56`) | agree | B17 and B18 cost little once B7 and data-driven rules exist. | Keep in Q2 after the core steps. |
| B5 kernel reader split (`parser.md:22-30`) | over-engineered | A working, tested reader split into three layers buys little; after B7 the reader only checks annotations, which shrinks `nodes.rs` by itself. | Replace by "reader checks, analysis infers" after B7; S-M. |
| Missing: typed payloads | missing | String payloads underlie D10, the FFI shape of B2, B6 conditions and B7 rules (`egraph.rs:10-28,63-94`). | New first behavioral-neutral step, M. |
| Missing: rule scheduling | missing | Bare-variable rules (`ir2ir_rewrites.txt:3,4,5,8`) make saturation end at `NODE_LIMIT` (`src/lib.rs:20`). | Measure node growth per rule; add `BackoffScheduler` or conditions. |
| Missing: staffing | missing | The plan is sized for two engineers (plan `README.md:3`); the roadmap has 3-4. | Re-cut into two parallel lanes (Rust core, FFI and generators) with the sizes below. |
| Decision: keep C++ allocator (`decisions.md:113-118`) | agree | OR-Tools solver (`cpp/malloc/include/solver.h`) is the solid part. | Keep; typed FFI only. |
| Decision: rewrite from scratch rejected (`decisions.md:98-100`) | agree, except two pieces | The applier dispatch and both extractors have no design worth keeping beyond the (class, buffer) state and the drop reasons. | Replace those two; reshape the rest. |

## What the plan would lose

- B10 "preserve candidate order" keeps the current tie-breaking but also keeps cost-blind selection; nothing good is lost by dropping it if the deterministic gate pins the new order.
- B4 as written deletes `parser.cc` down to `create_instruction` (`parser.md:14-20`) and risks deleting `hold`/`hold_tensors` with it.
- B2 drops the readable `.pii` text; the plan keeps a Rust printer for `--log` (`architecture.md:84`), which must stay byte-compatible for debugging diffs.

## Revised order

Sizes in engineer-weeks (ew) for the backend lanes; Q1 is weeks 1-13.

| # | Step | Lane | Size | Gate |
| --- | --- | --- | --- | --- |
| 1 | B1, B28 moves and renames | A | 1 ew | build, tests |
| 2 | Deterministic search mode, `search` returning `Result`, harness (B3 extended) | A | 3 ew | identical asm and candidates across two runs and machines |
| 3 | Typed payloads in `TensorOp` | A | 3 ew | identical asm |
| 4 | B7 shapes and types in `make`, shape as `Option` (D8) | A | 5 ew | identical asm; `merge` silent |
| 5 | B9 constants in the analysis, `extract(&EGraph)` | A | 1 ew | identical asm |
| 6 | B2, B4, B25 typed FFI incl. kernel facts; keep `hold_tensors` | B | 4 ew | identical asm; no file read in C++ without `--log` |
| 7 | B13 model from the front end's typed IR (with language track) | B | 4 ew | identical generated crate |
| 8 | Data-vs-code decision; B6 as generic applier plus per-instruction table; B8 named IR-to-IR rules | A | 5 ew | identical asm; text rules deleted |
| 9 | B14 checks build the model; B21, B22, B24 | B | 3 ew | `test_refusals.py` |
| 10 | B12 formatter for the remaining emitters | B | 2 ew | `tests/lint.sh` |
| 11 | B10 merge, then cost-guided k-best extractor; rule scheduling | A | 5 ew | equal or lower cost on all examples; time per kernel |
| 12 | Reader checks annotations against `make` (replaces B5) | B | 2 ew | `test_reader.py` |
| 13 | In-place aliasing as a solver equality constraint, B11 copies at candidate building (ruling 17) | B | 3 ew | oracle test with a shared overwritten input; output offset equals input offset |
| 14 | B26, B27; B23 handed to the language track with the B7 rule table (ruling 16) | B | 2 ew | new rejection tests |
| 15 | B15 to B20 coverage | A and B | 10-14 ew | per-row tests flip |

- Steps 1-5 and 6-7 run in parallel in Q1; steps 8-14 close Q1 into early Q2; step 15 fills Q2.
- Total about 53-57 ew: two backend engineers for two quarters, leaving the third and fourth engineer to the language and oracle tracks.
