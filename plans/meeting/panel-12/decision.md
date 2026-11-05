# Panel 12 decision: long-term platform

| Question | Decision | Confidence |
| -------- | -------- | ---------- |
| Q12.1 | A in strangler form: StableHLO importer as a text bridge, then dialect and in-house e-graph, then the engine swap through the Rust FFI, then one C++ driver; front end stays Python; one generic instruction op by default, a spike may overturn it | high on order, medium on op form |
| Q12.2 | C with the compile path scheduled: `ExportedProgram` to a StableHLO portable artifact; torch-mlir is the default producer when a released torch-mlir runs on a released torch, else an ACT-owned walker; strict by default; partition opt-in in wave 10; eager device and PJRT plugin later | high on structure, medium on producer |
| Q12.3 | B: standalone in-house C++ library, no MLIR header; backtracking matcher on per-operator storage; extraction stays in the backend; a three-tier swap gate | high on B, medium-high on gate |
| Q12.4 | 19 items in waves 8 to 10 of tracks U, B, O, A, T; weights 23, 24, 18; nine text changes to earlier items now | medium-high |

Decision 33 holds as written: a release pins a released StableHLO version, and the LLVM commit that release fixes is transitive; no owner question.
Code paths are relative to /home/user; plan paths to /home/user/act/plans; brief names to panel-12/research/.

## Rule set

- The front end, `check`, and every generator stay Python; generators emit C++ as they emit Rust today; no `Model` is serialized.
- MLIR code lives only in the importer, the dialect, and the bridge; the e-graph library and the allocator include no MLIR header.
- MLIR rewrite drivers never do selection; the allocator stays a C++ library behind the B2 flat struct.
- The supported MLIR input is a StableHLO portable artifact; text input serves tests within one pinned version.
- XLA HLO text stays an input (decision 40) and gets a C++ reader before the Rust crate is deleted.
- Public releases pin one released StableHLO version (U15); public build files derive its LLVM commit from that tag and never write one.
- Producers of StableHLO (torch-mlir, jax) run in their own package and process and hand over a portable artifact at U15's version; their MLIR never links into ACT.
- The torch package is strict by default under decision 40, sets `dynamic=False`, and never falls back silently.
- The e-graph library keeps: arena e-nodes with the typed payload in the hash-cons key; union-find with path compression and union by size; deferred rebuild; one class-data record with `make` and a `merge` that is a coded fault and reports change correctly; preconditions evaluated as soon as their variables bind; matches that yield e-node ids; an applied-substitution memo; a read-only class view; insertion-ordered containers; three limits with stop reasons.
- The library drops: explanations, a run-time rule language, Datalog functions, threading, `BackoffScheduler`, and the `modify` hook.
- Generic join, an MLIR allocator pass, an eager device, and a PJRT plugin each wait for their named trigger.

## Q12.1: port order

Decision: option A in five gated steps, each under B3 and B29.
1. B47 reads StableHLO portable artifacts and prints the kernel text today's reader accepts; no Rust changes, so B3 goldens hold.
2. B48 spike, then B50 dialect generator; B49 library core in parallel.
3. B52 swaps egg for the library through the B2 C ABI, decision 45 read literally; egg is deleted.
4. B54 typed import replaces the text bridge; B55 ports the HLO text reader.
5. B56 is one C++ driver; the Rust crate is deleted.

Reasoning and evidence:
- The reader takes XLA HLO text only (failures.md B43 section; isel/initializer/lib.rs:36), and MLIR parses StableHLO but not HLO text (mlir-ingest.md 1.1, 2.1), so a C++ HLO reader is a real item, not an omission to absorb (round3-opus.md B55).
- Only portable artifacts carry the 5-year backward and 2-year forward guarantees (mlir-ingest.md 2.2).
- "LLVM does not have a stable API" (mlir-ingest.md 1.7), which confines MLIR to the edges.
- Selection keeps saturation's alternatives (act-backend/act_backend/generic/src/pipeline.rs:486-495), while the greedy driver applies the highest-benefit pattern (mlir-ingest.md 1.4).
- Option B duplicates `check`, which decision 42 defines once; option C has no intermediate for B3 to compare.
- ACT is not an IREE plugin: dynamic plugins need the host's exact IREE and LLVM revisions (mlir-ingest.md 4.2).
- Op form: no consumer needs per-ISA op classes, since selection and allocation run outside MLIR.
- A generic op reads a table the backend generator writes, like today's generated Rust, and its verifier reuses the library's guard evaluator, so checked guards have one implementation.
- Per-ISA ODS puts `mlir-tblgen` and MLIR headers into every ISA build; no brief measured that cost (mlir-ingest.md section 7), so B48 measures it.
- Per-ISA ODS replaces the generic op only if it builds from U15's prebuilt package with no further install on the architect's machine.
- The table is generator output, not the `Model`, so U2's "second non-Python consumer" caveat (shared/plan/U2.md:15) stays closed.

Dissent: Fable keeps per-ISA ODS as the default; Sonnet first proposed the allocator as an MLIR pass, withdrawn in round 3.

## Q12.2: PyTorch backend

Decision: option C, compile path only scheduled.
- B53 is one adapter: `torch.export` with core ATen decompositions, then a producer that writes a portable artifact at U15's version for B47.
- B53's first step is the pairing check: does the latest released torch-mlir import an `ExportedProgram` from the latest released torch and cover the B42 input specification?
- Pass: torch-mlir is the producer, and torch-mlir input files (decision 43) enter through the same package.
- Fail: an ACT-owned walker maps each core ATen node in the input specification to StableHLO and stops with a coded error naming any other node; it is also B43's torch.fx reader.
- Vendoring torch-mlir is out: it puts torch-mlir's LLVM submodule and a torch nightly into ACT's public build, which decision 33 forbids.
- torch-mlir is a new dependency, asked when B53 lands (decision 28 and the engineering rules), not now.
- A29 wraps B53 for `torch.compile` through `aot_autograd`; A30 adds partition as an opt-in that prints every undelegated node.

Reasoning and evidence:
- Decision 33 does not bar torch-mlir: PyPI release 20261101 is a final version whose `requires_dist` names only numpy, packaging, and an onnx extra (verified at pypi.org/pypi/torch-mlir/json).
- Its own CI pins `torch==2.15.0.dev20260917` (verified in llvm/torch-mlir pytorch-requirements.txt), so the pairing with a released torch is untested upstream; that is the empirical question B53 settles first.
- torch-mlir is the default because decision 43 needs it anyway for torch-mlir input, and its per-operator lowering already does broadcasting and type promotion that a walker must rewrite.
- The walker is the fallback, not a second build: one producer ships, behind one interface.
- Risks recorded: torch-mlir's PyPI history jumps from 2022 to 20261001 (verified), so its cadence is young; FX import of one large model took over 20 minutes (mlir-ingest.md section 3).
- PyTorch/XLA is deprecated and its repo archives on 15 Nov 2026 (verified in pytorch/xla README.md), so `torch_xla` is neither producer nor cross-check.
- Strict default: TorchNeuron's compile path fails by default on an op it cannot lower (torch-backends.md, existing accelerators); Dynamo graph-breaks only without `fullgraph=True` (torch-backends.md, torch.compile).
- Partition is scheduled, not later: the backend handles 18 operations against 193 core ATen ops (torch-backends.md, torch.export), so strict alone runs few real models, and decision 44 asks for a usable backend.
- Partition stays legal under decision 40 because the compiler only sees supported subgraphs, and the report keeps it from being silent.
- The `aot_autograd` hook is private and minors ship every two months (torch-backends.md, torch.compile), so each act release pins one released torch with a CI row per supported minor.
- PrivateUse1 needs a device runtime ACT does not own (torch-backends.md, PrivateUse1); a PJRT plugin reaches PyTorch only through PyTorch/XLA; both are later rows.

Dissent: Opus preferred the walker as default, but agreed in round 3 that the pairing check decides; Fable kept partition as a later row.

## Q12.3: C++ e-graph library and migration

Decision: option B as the rule set states; option A fails on evidence, option C is closed by decision 46.
- Option A: the eqsat dialect keeps congruence by whole-graph CSE, reports no benchmarks, and PDL cannot match regions (egraphs.md section 4).
- Deferred rebuild is the one egg algorithm with measured value, 88x on congruence and 21x on saturation (egraphs.md section 1); an algorithm is not a dependency.
- Payloads join the key because `matches` compares only the variant today (egraphs.md section 8).
- `merge` must report change: today it panics on disagreement and always returns `b` unchanged (act-backend/act_backend/generic/src/ir/egraph.rs:403-414, verified).
- Matcher: backtracking over a per-operator index first; generic join gains are measured at 1e5 to 3e6 e-nodes, and dense patterns gain little (egraphs.md section 3), while ACT caps at 10,000 (generic/src/lib.rs:20, verified).
- The waste of rules 2, 3, 4, and 7 (ir2ir_rewrites.txt lines 3, 4, 5, 8, verified) is re-application, which B32's guards and the memo remove, not matching.
- Generic join comes forward only when a B34 profile after B32 puts matching above one third of search time; five percent buys too little for an L item.
- Extraction stays in the backend: candidates are ranked by an external C cost model and keyed by (class, buffer) (egraphs.md section 8), so no local cost function exists.
- `modify` is dropped: B32's guards and the memo give once-per-class firing without adding e-nodes inside rebuild.

Swap gate, three tiers:
1. Library property tests (congruence, hash-cons, analysis conflict) without egg.
2. Kernels that saturate or stop at the iteration bound: equal canonical e-graph dumps per iteration, then identical chosen assembly and `--log` files, given a no-ban schedule and canonical extraction order.
3. Kernels that still stop at `NODE_LIMIT`: chosen cost no worse than egg and an equal oracle result.
- Tier 2 needs egg's `SimpleScheduler`, which runs every rewrite every iteration, instead of the default `BackoffScheduler` in force today (pipeline.rs:490-495, verified); B32 makes the switch once its guards end node-limit stops, with one golden re-record in its own commit.

Dissent: none remains on the gate; all three round-3 files compose it the same way.

## Q12.4: joining the roadmap

Decision: later waves 8 to 10 of the existing tracks, after M5, with no new track or plan.
- Wave 7 holds only follow-ons with no scheduled dependent (roadmap/wave-4-7.md, Wave 7), and these items have dependents; no date is set before the M1 rebaseline (roadmap/waves.md, Rules).
- fsim joins through O29 and native-path steps 1 to 4 (oracle/fsim/native-path.md, minimal steps); step 5, the interpreter, stays out under decision 28.
- The oracle's per-instruction StableHLO comes from O3's generic emitter alone; O34 and B50 both read it, so the oracle never waits on the backend dialect (risk K9).
- The pin set is shared-track, beside U9, which owns docker and CI scripts; A16 records it.

Decision 33 and StableHLO, ruled:
- StableHLO cuts a release weekly with a version bump (stablehlo docs/compatibility.md, verified), and tag v1.20.0 exists with `build_tools/llvm_version.txt` fixing LLVM commit ab547095 (verified).
- Pinning StableHLO 1.20.0 pins a released version; the LLVM commit is a property of that release, as the released `jaxlib==0.11.2` already fixes an XLA commit (act-oracle/pyproject.toml:19).
- This applies decision 33 as written and does not change the owner's policy, so it is not marked for the owner.
- Public users receive a prebuilt importer; if a public source build of the image must fetch LLVM, it reads the commit from the StableHLO tag, and if even that is unwanted the source build stays in the private repos.

Text changes to earlier items, now:

| Item | Change |
| ---- | ------ |
| B29 | `search` returns one record: stop reason, candidates with costs, chosen assembly |
| B31, B2 | Typed payload is plain integers and a dtype enum with a size and offset test; it becomes B49's e-node payload and key |
| B7 | Class data is (shape, dtype, constant value); `merge` reports change per egg's contract |
| B6 | A rule is a language-neutral record (name, pattern, guard tree over bound names, right-hand side); no engine type in any signature; Rust emitter now |
| B10 | Extraction orders choices by canonical term key, not e-node id |
| B32 | Once guards end node-limit stops, both modes run egg's `SimpleScheduler`, unless cost rises on an example; goldens re-recorded once in its commit |
| B42 | Input specification names StableHLO portable artifacts with a version window, ranked static-shape tensors, and a dtype list |
| B43 | Narrows to detection, including portable artifacts; its readers move to B47 (StableHLO) and B53 (torch-mlir, torch.fx) |
| O3, O29, A16 | O3 is the one source of per-instruction StableHLO; O29 plans O34 to O36 on U15; A16 records U15 and each supported torch minor |

## Final roadmap items

IDs are the next free per track (highest today B46, O33, T38, U14, A28); an author re-checks before use.
Sizes S = 1, M = 3, L = 6; within a wave, a dependency sits in an earlier row.

| ID | Title | Depends on | Size | Wave |
| -- | ----- | ---------- | ---- | ---- |
| U15 | Toolchain pin set: one released StableHLO, LLVM from its tag, MLIR and StableHLO static libraries in one build image, CI cache, prebuilt importer | U9 | M | 8 |
| B47 | StableHLO importer: C++ tool reading portable artifacts, printing today's kernel text, coded errors | B42, B43, U15 | M | 8 |
| B48 | Dialect spike: generic instruction op against per-ISA ODS, per-ISA build cost measured | B13, U15 | S | 8 |
| B49 | E-graph library core: arena, typed key, union-find, deferred rebuild, class data, read-only view, property tests | B29, B31, B7, B9, B26 | L | 8 |
| B50 | ISA dialect generator per B48: typed attributes, checked verifier, semantics region from O3 | B48, U10, O3 | M | 9 |
| B51 | Matcher and rules: backtracking per operator, early preconditions, matched ids, applied memo, C++ rule emitter | B49, B6, B8 | L | 9 |
| B52 | Dual-engine harness and swap through the B2 C ABI, three-tier gate, egg deleted | B51, B2, B3, B10, B32, B34, B35 | M | 9 |
| B53 | Torch adapter: `ExportedProgram` to portable artifact; pairing check picks torch-mlir or the walker; strict | B47, B42 | M | 9 |
| B54 | Typed import: importer builds library terms, chosen program as dialect ops, text bridge deleted | B47, B50, B52 | M | 10 |
| B55 | XLA HLO text reader in C++, holding every message of tests/test_reader.py | B54 | M | 10 |
| B56 | One C++ driver: importers, engine, extraction, allocator, cost, exclusion report; Rust crate deleted | B54, B55, B46 | L | 10 |
| O34 | fsim step 1: C++ emitter per instruction from O3's generic emitter | O29, O3, U15 | L | 8 |
| O35 | fsim step 2: `Pipeline::submit`, `end_trace`, `sync` replace `compile_and_load` and `execute` | O34 | M | 8 |
| O36 | fsim steps 3 and 4: debug print out of `custom_call`, PJRT pinned by header and plugin hash; every golden and norm row holds | O35, U15 | M | 9 |
| A29 | `torch.compile` and `torch.export` entry points, artifact-hash cache, one released torch per release, CI row per minor | B53, A23 | M | 9 |
| A30 | Partition opt-in: supported subgraphs to ACT, every undelegated node printed | A29, B46 | M | 10 |
| T39 | StableHLO corpus exported once from jax, differential against the HLO path | B47, T13 | S | 8 |
| T40 | Torch tests: eager PyTorch against ACT on the oracle per operation and dtype, coverage table | A29, T29 | M | 9 |
| T41 | Platform gate: B3 on the C++ driver, oracle on fsim | B56, O36 | M | 10 |

Wave weights: wave 8 is 23 (7 items), wave 9 is 24 (7 items), wave 10 is 18 (5 items).
Later rows, no wave, no dependent: eager PrivateUse1 device over the oracle; PJRT plugin as the JAX door; generic join in B51 (trigger above); allocator as an MLIR pass at a second consumer.

## Escalation to the owner

None; no debater raised one, and the decision 33 reading above applies the policy without changing it.
