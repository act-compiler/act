# Panel 12, round 1: Sonnet (MLIR and torch-mlir integration lens)

Scope: Q12.4, Q12.1, Q12.2, Q12.3.
Owner decisions 23, 28, 33, 39 to 45 are taken as binding.
New item IDs below were free in a grep of plans/ (B47 to B58, T39 to T42, O34, O35, A33, A34); an author re-checks before use.

### Q12.4: How do these items join the roadmap?

Position: the work is waves 8 to 10 of the existing tracks, after M5, with the torch path in wave 9 and the fsim path in wave 10.
The new items stay in their tracks: backend owns the importer, dialect, library, and torch adapter; oracle owns fsim; tests owns gates; act owns pins and the user-facing package.

1. Waves 1 to 7 end at M5 with act 3.0 and the tutorial (plans/roadmap/waves.md, Waves table); the platform must not move M1 to M5.
2. Wave 7 is "follow-ons with no scheduled dependent" (plans/roadmap/wave-4-7.md, Wave 7); these items have real dependents, so they open new waves instead of joining wave 7.
3. Dependencies are real: the importer needs B42/B43 (input specification), the dialect needs U10 and B13 (generators read the `Model`), the library needs B31, B7, B9, B29 (typed payloads, analysis, no clone, deterministic mode).
4. The oracle half joins through O29 and O3: the fsim emitter reuses the generic emitter calls (plans/oracle/plan/architecture.md, O29 and O3 rows).

| ID | Item | Track | Depends on | Size | Wave |
| -- | ---- | ----- | ---------- | ---- | ---- |
| B47 | `act-import`: C++ MLIR tool, StableHLO (text or portable artifact) to the kernel text the backend reads | backend | B42, B43, U10 | M | 8 |
| A33 | Toolchain pin: one LLVM and StableHLO commit per release, build image, CI cache | act | A16, B47 | M | 8 |
| T39 | StableHLO corpus: examples lowered once from jax, differential against the HLO path | tests | B47, T13 | S | 8 |
| B48 | ISA dialect generator: Python emits ODS and C++ (ops, verifiers from constraints, semantics region) from the `Model` | backend | B13, U10, B47, A33 | L | 8 |
| B50 | E-graph library core: arena, hash-cons, union-find, typed class analysis, rebuild, builder for rules | backend | B31, B7, B9, B29 | L | 8 |
| B51 | E-matcher and the differential harness against egg (canonical dumps) | backend | B50, B6, B3 | L | 9 |
| B52 | Backend runs on the library through FFI; egg removed | backend | B51, B10, B34, P3 | M | 9 |
| B54 | MLIR bridge: operations to e-nodes and extracted candidates to ISA dialect operations | backend | B50, B48 | M | 9 |
| B49 | Allocator as an MLIR pass over the ISA dialect (replaces the B2 crossing) | backend | B48, B2, B30 | M | 9 |
| B55 | Torch adapter: `ExportedProgram` to versioned StableHLO artifact, then B47 | backend | B47, B42 | M | 9 |
| B56 | `torch.compile` backend: registration, strict mode, cache, run on the oracle | act | B55, A23 | M | 9 |
| T40 | Torch tests: eager PyTorch against ACT plus oracle per operator, with tolerances | tests | B56, T29 | M | 9 |
| O34 | fsim-lib emitter: second target of the oracle emitter, instruction semantics as StableHLO | oracle | O29, O3, B48, A33 | L | 9 |
| B53 | Selection and extraction as C++ passes; rules generated as C++; Rust driver removed | backend | B52, B54, B49 | L | 10 |
| O35 | Oracle executes through fsim-lib; jax, jaxlib, nanobind leave the run | oracle | O34, T13 | L | 10 |
| T41 | Oracle-on-fsim gate: every golden and norm row holds | tests | O35 | M | 10 |

5. PrivateUse1 eager kernels (B57) get a "later" row with no wave and no dependent, the same rule as wave 7 (plans/roadmap/wave-4-7.md, Wave 7 intro).
6. What earlier items must do now, each a text change to an existing item:
   - B29: `search` returns one record (stop reason, candidates with costs, chosen assembly) so a second engine compares records, not log text (plans/backend/plan/search.md, B29 What).
   - B31 and B7: the typed `TensorOp` fields and `TensorInfo` (shape, dtype, constant value) are written as the e-node and class-data schema of B50, with plain integers and a dtype enum (act-backend/act_backend/generic/src/ir/egraph.rs:5-30, :385-410).
   - B2: the flat node array is plain-old-data with a size and offset test, owned by the C++ side from the start; cbindgen is transitional (plans/backend/plan/architecture.md, B2 What).
   - B6: the rule record is language-neutral (name, pattern, guard expression tree, right-hand side), generated from the `Model`, with a Rust emitter now and a C++ emitter in B53 (plans/backend/plan/isel.md, B6 What).
   - U2: the `Model` stays a Python object model; generators emit C++ source, so no serialized form is needed (plans/shared/plan/U2.md, Caveats "second non-Python consumer").
   - B42 and B43: the input specification names StableHLO as the one MLIR input and states the StableHLO version range; B43's Rust StableHLO reader becomes B47 (plans/backend/plan/failures.md:56-66).
   - A16: the release checklist records the LLVM and StableHLO pin (plans/roadmap/waves.md, Release order).
7. The fsim join is one shared pin and one shared lowering: fsim-lib already links static MLIR and StableHLO at 1.20.0 (plans/oracle/fsim/architecture.md:25, :68), and the ISA dialect's semantics region is the StableHLO that both the backend patterns and the fsim emitter read.

Confidence: 75 percent on the wave split, 60 percent on the exact item boundaries.
Would change my mind: an owner schedule that wants torch before M5, or a measurement that B47 cannot reach the Rust reader's coverage.

### Q12.1: How is the stack ported to C++ on MLIR infrastructure, and in what order?

Position: option A in strangler form: importer first, then the dialect, then the library, then the passes, with the front end left in Python.

1. The front end is the authoring surface and the check function (owner decisions 41, 42); its audience writes Python, so a C++ front end buys no MLIR input (plans/shared/plan/U2.md, What).
2. The backend is already half C++: the allocator is C++ with OR-Tools (act-backend/act_backend/generic/cpp/malloc/CMakeLists.txt), and the Rust crate is 5,348 lines with generated parts (wc over generic/src).
3. Option C breaks the B3 gate, which needs a working pipeline at every step (plans/backend/plan/architecture.md, B3 What); a one-shot port has no oracle-differential at the midpoint.
4. Step 1 is B47, a standalone C++ tool that parses StableHLO and prints the kernel text the Rust reader accepts.
   - It delivers StableHLO input before the port, because the reader today takes XLA HLO text only (act-backend/act_backend/generic/src/isel/initializer/parser.rs:21-35).
   - MLIR parses `func` and `stablehlo` natively, as fsim already does (plans/oracle/fsim/integration.md:20).
   - IREE took the same route: its input conversion ingests StableHLO as an in-tree input plugin (https://opensecura.googlesource.com/3p/openxla/iree/+/ebb5b7d4a3c59b8dac9c5a4fe4ba78f0c736dc7f).
   - The printer must map StableHLO to the reader's forms: `reduce` with a region becomes a helper computation plus `to_apply` (act-backend/examples/mini_trn/kernels/attention.hlo:4-8), `dot_general` becomes `dot` with the four dimension lists (expression.rs:173-178), and element types `si32` become `s32`.
   - The StableHLO spec lists 15 float types and sub-byte integers (https://openxla.org/stablehlo/spec); the reader's dtype table decides which are in the input specification, and the rest stop with a coded error (decision 40).
5. Step 2 is B48, the ISA dialect generated from the `Model` by the Python generator, in the same style as today's generators that emit Rust and C++ from templates (act-backend/act_backend/templates/).
   - Each instruction becomes an ODS op; attributes carry the width layer type, default `s64` (plans/shared/plan/U2.md, What on declared types).
   - The verifier is generated C++ with checked 64-bit arithmetic, the same discipline as `CheckedWriter` (plans/backend/plan/isel.md, B6 Preserves).
   - ODS declares typed attributes, regions, and `hasVerifier` hooks (https://mlir.llvm.org/docs/DefiningDialects/Operations/), so the semantics region and the generated constraint check fit; mapping `s<w>` and `u<w>` onto integer attributes is a B48 spike.
   - IRDL could define the dialect as data without a C++ build, but `irdl.c_pred` constraints cannot register at runtime and the page calls IRDL not comfortable to write by hand (https://mlir.llvm.org/docs/Dialects/IRDL/); generated ODS keeps the checked guards.
6. Step 3 is the library and its FFI switch (Q12.3), which comes before the selection pass because MLIR's greedy driver requires a pattern to update, replace, or erase its root (https://mlir.llvm.org/docs/PatternRewriter/), and ACT needs the alternatives that saturation keeps (act-backend/act_backend/generic/src/pipeline.rs:488-495).
7. Step 4 is B49 and B53: the allocator becomes a pass, selection and extraction become passes, and the Rust crate is deleted.
8. The pins are the integration risk.
   - StableHLO pins one LLVM commit (https://raw.githubusercontent.com/openxla/stablehlo/main/build_tools/llvm_version.txt gives c744694c6ae1d12bbf7d0447b4bc91ee94a1904c at fetch time).
   - torch-mlir pins its own `llvm-project` and `stablehlo` submodules and builds against nightly PyTorch (https://raw.githubusercontent.com/llvm/torch-mlir/main/docs/development.md).
   - StableHLO portable artifacts carry 5 years backward and 2 years forward compatibility, and only the official serialize and deserialize APIs carry it (https://openxla.org/stablehlo/compatibility).
   - So ACT anchors on StableHLO's pin and consumes torch-mlir or any other producer only as a source of serialized portable artifacts across a file boundary, never as a linked library (A33, B55).
9. The gate per step: structural steps keep chosen assembly and sorted candidate sets identical; engine swaps compare saturated e-graph dumps where the kernel saturates, and cost plus oracle result where a node limit stops it (act-backend/act_backend/generic/src/pipeline.rs:520-530).
10. LLVM, MLIR, and StableHLO are not on the owner decision 28 list; owner decisions 43 and 44 name them, so I read them as covered when the item lands.

Confidence: 80 percent on A over B and C; 60 percent on the importer-as-text-bridge in step 4.
Would change my mind: a measurement that kernel text loses information StableHLO carries (for example dynamic shapes), which would force a typed handoff in B47.

### Q12.2: How is a portable PyTorch backend built?

Position: option C with only the compile path scheduled; the adapter turns a captured graph into a serialized StableHLO artifact behind one interface, with torch-mlir the default producer and `torch_xla` the cross-check producer.

1. A PrivateUse1 device needs a C++ kernel per ATen operator plus a CPU fallback (https://pytorch.org/tutorials/advanced/privateuseone.html); an ACT accelerator has no eager runtime, so each eager call would be a one-operator search plus oracle run, with no fusion.
2. The compile path hands the whole graph to the existing pipeline and needs no fork of PyTorch.
   - Precedent: iree-turbine registers a custom `torch.compile` backend, sends Dynamo's graph through torch-mlir to IREE, and says it needs "compatible versions of IREE, torch-mlir, and PyTorch" (https://iree.dev/guides/ml-frameworks/pytorch/).
   - The same page says its initial support is CPU-only, so the integration cost is real even for a mature project.
3. The input to the producer is an `ExportedProgram`: torch-mlir's `fx.export_and_import` takes one and, with `output_type="stablehlo"`, runs `torch-backend-to-stablehlo-backend-pipeline` (https://raw.githubusercontent.com/llvm/torch-mlir/main/python/torch_mlir/fx.py, compiler_utils.py in the same directory).
   - Whether Dynamo's graph is re-exported with `torch.export` or taken from `aot_autograd` with decompositions is a B55 spike; the two custom-backend and export doc pages did not load for me, so I do not rely on their guarantees here.
4. The two producers differ on coverage and coupling.
   - torch-mlir's StableHLO lowering is a set of per-operator patterns, some of which need static shapes (`NativeLayerNorm`, `Diagonal`, `Trilu`; https://raw.githubusercontent.com/llvm/torch-mlir/main/lib/Conversion/TorchToStablehlo/Basic.cpp, first 100,000 characters read).
   - Its roadmap says each backend is implemented independently with inconsistent coverage, and the FX path is "not fully functional yet" (https://raw.githubusercontent.com/llvm/torch-mlir/main/docs/roadmap.md; the text may be stale).
   - It lives in the LLVM incubator, outside any LLVM release (https://github.com/llvm/torch-mlir), and builds against nightly PyTorch (docs/development.md).
   - `torch_xla.stablehlo` needs `torch_xla` matched to `torch` and has limited dynamic shapes (https://openxla.org/stablehlo/tutorials/pytorch-export).
   - The target is Core ATen, 193 operators and "still under active development" (https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_ir.html), against the 18 operations the backend handles.
5. So the interface is `ExportedProgram -> serialized StableHLO artifact`, run in a separate Python environment so the torch-mlir pins never meet the ACT build; both producers run in T40 and a coverage table over the backend's 18 operations decides the default.
   - This keeps the plans' staged torch-mlir reader (plans/backend/plan/failures.md:65) and drops the third torch.fx reader (:66), since the exported graph replaces raw FX nodes.
6. Decision 40: an operator outside the input specification stops with a coded error naming it; the adapter does not fall back silently, which the engineering rules forbid for research code (/home/user/.claude/.claude/engineering.md, "Blanket exception handlers or silent fallbacks").
7. Portability comes from the ISA specification only: the adapter holds no accelerator knowledge, and the registered backend name takes the generated compiler as a parameter.
8. Testability: run the produced assembly on the generated oracle and compare with eager PyTorch within the per-dtype tolerance (plans/minutes/rulings.md R8, R48), using the oracle's `kernel(...)` entry (R36).
9. B57 (PrivateUse1) is recorded as later; nothing depends on it.

Confidence: 70 percent for strict mode and the compile path first; 50 percent that torch-mlir is the right default producer.
Would change my mind: a coverage table where `torch_xla` lowers more of the needed operators with stable wheels, which makes it the default; or a PyTorch release that ships a StableHLO export in core.

### Q12.3: What is the C++ e-graph library, and how does the backend migrate to it?

Position: option B, a standalone C++ library with a compact term representation, typed class analysis, and an MLIR bridge; the MLIR-native option is rejected for the engine.

1. No fit exists to adopt: a search found no maintained C++ e-graph, only an inactive 2023 egg port with no stated e-matching or tensor use (https://gitee.com/klose28/egraph); option C is empty.
2. DialEgg drives egglog (Rust) from MLIR by translating text, which is the Rust FFI that owner decision 45 leaves (https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog).
3. The `eqsat` dialect holds e-graphs as IR, with `eqsat.eclass` and `eqsat.egraph` operations (https://arxiv.org/html/2505.09363).
   - It matches with PDL lowered to `pdl_interp` and a custom backtracking interpreter, and restores congruence by running MLIR CSE over the whole e-graph, which the authors call less efficient than incremental rebuilding.
   - PDL cannot match whole regions, the paper reports no benchmark numbers, and the implementation is a prototype in xDSL (Python), not in C++ MLIR.
   - So it supplies neither speed nor a C++ artifact to adopt; its idea of non-destructive `replace` into an e-class is the semantics B54 needs at the bridge.
4. IR-resident e-nodes pay MLIR uniquing and region mutation per merge, while ACT's graphs are small tensor DAGs with shapes as class data (act-backend/act_backend/generic/src/ir/egraph.rs:385-410).
   - egg's central speed device is deferred, amortized rebuilding (https://arxiv.org/abs/2004.03082), which the library copies: apply all matches of an iteration, then rebuild once.
5. The representation: an e-node is an op id, a small integer-attribute array, a dtype, and child ids in an arena; the class analysis is `{shape, dtype, constant value}` with `make` and `merge` hooks, and a `merge` conflict is recorded as the coded shape fault, not a panic (egraph.rs:403-409, plans/meeting/resolutions.md, Q3.6 row).
6. E-matching: start with egg-style backtracking over a per-operator index, because the ACT rule patterns are small (ten IR-to-IR rules, none over five nodes; act-backend/act_backend/generic/src/isel/rewrites/ir2ir_rewrites.txt).
7. Add generic join (relational e-matching) if the harness shows pattern cost dominating: it gives a worst-case bound on run time and was reported orders of magnitude faster in practice on large e-graphs (https://arxiv.org/abs/2108.02290); the matcher interface is "yield substitutions", so the swap is local.
8. Preconditions: guards are C++ predicates over the typed substitution, run as soon as their variables bind, which prunes inside the join; the guard body is the generated checked-integer expression tree (plans/backend/plan/isel.md, B6 Preserves).
9. Complex rewrites with little code: a rule is `rule(name, pattern, guard, builder)` where the builder returns typed nodes, replacing the three number-keyed Rust functions per rule (act-backend/act_backend/generic/src/isel/rewrites/lib.rs:6-19).
10. Extraction: the library offers bottom-up cost extraction and a k-best enumerator; the (class, buffer) policy of B10 stays in the backend (plans/roadmap/wave-3.md, B10 row).
11. No proof machinery: no explanations or proof forest; stop reasons and rule counters stay (plans/priorart/egg-egglog.md, borrow list).
12. Migration, in order:
    - B50 builds the core; B51 builds the matcher and a harness that runs egg and the library on the same corpus under B29's deterministic mode.
    - The compare set is the saturated class dump for kernels that saturate and cost plus oracle result for kernels that hit `NODE_LIMIT`, since egg's default `BackoffScheduler` shapes what a limit cuts (plans/backend/plan/search.md, intro).
    - B52 switches the Rust driver to the library through FFI and deletes egg; B3 goldens and chosen assembly must hold.
    - B53 moves the driver into C++ passes.
13. Rejected sub-option: porting egg's runner semantics exactly; the library matches stop-reason names and limits, not the ban schedule.

Confidence: 70 percent on B over A; 55 percent that backtracking suffices without a join.
Would change my mind: an upstream C++ `eqsat` with incremental rebuild, class analyses, and a measured match rate within 2x of the library on the corpus, or a rule set needing patterns of eight or more nodes.

## Unverified

- The PyTorch custom-backend and `torch.export` doc pages returned no text; claims on `aot_autograd` and Dynamo graph breaks are left to the B55 spike.
- The torch-mlir roadmap text predates the current FX importer and may be stale; `Basic.cpp` was read only to character 100,000.
- The StableHLO LLVM pin was read once; it moves weekly.

## Escalation to the owner

None.
