# Panel 12, round 3: Sonnet (MLIR and torch-mlir integration lens)

Read: all six round files, the three briefs, owner decisions 23, 28, 33, 39 to 46, and plans/backend/plan/search.md.
Briefs are cited as egraphs.md, mlir-ingest.md, torch-backends.md.
New IDs were free in a grep of plans/ outside meeting/ (B47 to B58, U15 to U17, T39 to T42, O34 to O37, A29 to A31); an author re-checks.

## Round-2 concessions I carry into the final positions

- The allocator stays a C++ library on the B2 flat struct; Sonnet B49 (allocator as an MLIR pass) is withdrawn.
- The extractor stays outside the e-graph library.
- The text bridge is step one only; a typed handoff replaces it when the library lands.
- Relational matching is not built first.

## Q12.1: C++ on MLIR, and in what order

Where the panel now agrees:
- Option A in strangler form; the front end stays Python and keeps `check` once (decision 42).
- Order: importer, dialect, library core, typed handoff, engine swap, then one C++ driver with the Rust crate deleted.
- Step one is a C++ tool that reads a StableHLO portable artifact and prints the kernel text the Rust reader accepts; Opus and Fable conceded it over a C++ `main` around egg.
- Portable artifacts are the supported input, since only they carry the 5-year and 2-year guarantees (mlir-ingest.md:72-75); text is a test path inside one pinned version.
- HLO text stays on the Rust reader until the examples are re-exported (mlir-ingest.md:11, 63).
- MLIR C++ has no API stability (mlir-ingest.md:58), so MLIR code is confined to the importer, the dialect, and the bridge; the library includes no MLIR header.
- Selection is not an MLIR pattern pass: the greedy driver applies one rewrite destructively (mlir-ingest.md:35) and selection needs saturation's alternatives (pipeline.rs:488-495).
- ACT borrows IREE's layering and is not an IREE plugin, since dynamic plugins need the host's exact IREE and LLVM revisions (mlir-ingest.md:4.2 bullets).
- IRDL is out for the dialect because `irdl.c_pred` cannot register at run time.

What still divides, one question: where ISA knowledge enters the MLIR code.
- Per-ISA generated ODS (Opus, Fable): the generate-then-build model already exists (act-backend/README.md:94), and typed ops carry a generated verifier and semantics region.
- Generic instruction op (me): an ISA-specific C++ build needs MLIR headers and `mlir-tblgen` on the architect's machine, about 147 MB installed, and a loaded plugin needs exact revisions (mlir-ingest.md:130, s4.2); selection uses no MLIR patterns, so typed ops add only the verifier.

Final position: option A, strangler order as above.
- The dialect is one generic instruction op plus a semantics region, driven by a data table the Python generator writes at generation time.
- That table is generator output like today's generated Rust, so U2's "second non-Python consumer" caveat (plans/shared/plan/U2.md:15) stays closed and no `Model` is serialized.
- The first task of the dialect item is a spike: measure per-ISA ODS build time and architect-side install cost; per-ISA ODS wins only if the install is a wheel or a prebuilt MLIR the architect already has.
- The spike result lands before the fsim emitter and the bridge start, since both read the semantics region.
- Confidence: high on A and the order; medium on generic op over per-ISA ODS (no brief measured a stock MLIR build, mlir-ingest.md:171).
- What would change my mind: a spike showing generated ODS builds in under a few minutes against a prebuilt MLIR wheel.

## Q12.2: Portable PyTorch backend

Where the panel now agrees:
- Option C with the compile path first; no fork of PyTorch; a `torch.compile` backend registered through `torch_dynamo_backends` under `aot_autograd` with core ATen (torch-backends.md:20-26).
- The handoff is a StableHLO portable artifact across a file boundary, into the same importer as Q12.1.
- Strict by default (decision 40): an op outside the input specification stops with a coded error; `dynamic=False`, as TorchNeuron does (torch-backends.md:121); a cache keyed by artifact hash and ACT version.
- Partition to eager is an opt-in with a report of undelegated ops, never silent.
- PrivateUse1 eager device and a PJRT plugin are later rows with no wave: the minimum eager set plus runtime pieces is real work (torch-backends.md:58-65) and PyTorch/XLA, the only PyTorch door to PJRT, is archived on 15 Nov 2026 (torch-backends.md:94).
- `torch_xla` is not a producer or a CI dependency.
- The package supports the latest two torch minors and is released per minor, because `torch._dynamo` is private and minors ship every two months (torch-backends.md:34-35).
- Every torch test is an oracle-differential test against eager PyTorch.

What still divides: the producer from `ExportedProgram` to StableHLO.
- For torch-mlir (Fable, me): AWS Neuron ships StableHLO from a vendored torch_mlir in production (torch-backends.md:110), and its per-operator lowering handles broadcasting and type promotion that a walker must reimplement.
- For an ACT-owned walker (Opus): torch-mlir pins `torch==2.15.0.dev20260917`, a nightly (torch-backends.md:89), and decision 33 forbids pre-release pins.

Final position: option C, compile path first; the adapter interface is `ExportedProgram -> portable artifact`, and the producer is decided by a pairing spike, not by preference.
- Spike: does the torch-mlir release 20261101 (mlir-ingest.md:101) import an `ExportedProgram` from released torch 2.15.0 and cover the operations in the input specification?
- Pass: torch-mlir is the producer, in its own package and environment, run as a subprocess.
- Fail: an ACT-owned walker over core ATen maps each node in the input specification to StableHLO and stops with a coded error on any other node; this is also the shape of B43's torch.fx reader.
- The producer serializes with `serialize_portable_artifact(module, target_version)` set to ACT's pinned StableHLO version (mlir-ingest.md:81-82), so torch-mlir's LLVM submodule and ACT's LLVM commit never meet in one process.
- Opus's own change-of-mind condition (a torch-mlir release on a released torch) is exactly the spike's pass case, so the panel agrees on the decider.
- Confidence: high on the structure and on strict; medium (about 50 percent) on which producer ships.

## Q12.3: C++ e-graph library and migration

Where the panel now agrees (option B, in-house per decision 46):
- A standalone library with no MLIR header; an MLIR bridge at import and at the chosen program only.
- Option A (eqsat dialect) is rejected on evidence: whole-graph CSE for congruence, no benchmarks, no analysis or extraction, PDL cannot match regions (egraphs.md s4).
- Kept: arena of hash-consed e-nodes with typed payloads in the key, union-find with path compression and union by size, deferred rebuild (88x and 21x, egraphs.md:11), class data with `make` and a `merge` that is a coded fault and reports change correctly.
- Kept: preconditions as predicates over bound class data, run as soon as variables bind; a matcher that returns matched e-node ids; an applied-substitution set per rule; the three limits with stop reasons; a read-only class view.
- Dropped: explanations, a run-time rule language, Datalog functions, threading, and `BackoffScheduler`.
- Matcher: backtracking over a per-operator index first, behind a "yield substitutions" interface, so generic join is a local swap.
- Extraction stays in the backend: the (class, buffer) policy and the external cost model on emitted assembly (egraphs.md:167-170); the library exposes members, class data, and canonical ids.
- B9 and B32 land before the swap: B9 makes constant-ness class data so extraction clones nothing (pipeline.rs:235); B32 guards remove the bare-variable fan-out (search.md:4).
- Migration runs egg and the library side by side behind the B2 C ABI; goldens are re-recorded at most once, in a commit of their own.

What still divides:
- Gate strength for kernels that stop at `NODE_LIMIT`: Fable and Opus want identical chosen assembly and `--log` files; I want cost no worse than egg and an equal oracle result.
  - Evidence for identical: B29's verification already demands identical assembly and `--log` on two machines (search.md:23).
  - Evidence for cost plus oracle: an independent engine cannot reproduce a cut that depends on egg's rule order (egraphs.md:19).
- The generic-join trigger: Opus 5 percent of compile time, then a third; I accepted 5 percent in round 2.

Final position: option B as above, with one gate in three tiers.
- Tier 1: library property tests (congruence, hash-cons, analysis conflicts) with no egg.
- Tier 2: kernels that saturate give equal sorted canonical term sets per class, and identical chosen assembly and `--log` once B29 numbers candidates in canonical term order and B35 breaks ties by declaration index.
- Tier 3: kernels that still stop at the node limit give cost no worse than egg and an equal oracle result.
- B29's deterministic mode runs egg's no-ban `SimpleScheduler` (docs.rs egg SimpleScheduler, cited by Opus and Fable); I concede it: it is a defined schedule, cheap to reproduce, and needs no copy of the ban logic.
- Generic join is brought forward when a B34 profile after B32 shows matching above a third of search time; at 5 percent a join pays little for an L item, so I withdraw 5 percent.
- Confidence: high on B over A; medium-high on backtracking first; medium on the tier split.
- What would change my mind: a rule set needing patterns of eight or more nodes.

## Q12.4: How the items join the roadmap

Where the panel now agrees:
- No new track and no new plan: items join backend (B), shared (U), tests (T), oracle (O), act (A), as waves 8 to 10 after M5, since wave 7 holds only items with no dependent (plans/roadmap/wave-4-7.md:118).
- Wave 8: pin set, importer, dialect, library core, fsim step 1, corpus.
- Wave 9: matcher and dual-engine gate, engine swap, bridge, torch adapter and backend, fsim steps 2 to 4 with their gate.
- Wave 10: C++ driver with the Rust crate deleted, and the platform gate.
- No M date is set for waves 8 to 10 before the M1 rebaseline (plans/roadmap/waves.md:15).
- Pins are released versions (decision 33): StableHLO 1.20.0 as fsim-lib already uses (plans/oracle/fsim/native-path.md:25); its LLVM commit is inherited from that tag.
- A33-style "commit pins" wording is dropped.
- No serialized `Model` item (U15 is withdrawn by Opus, Fable, and me) because the dialect table is generator output.

What still divides:
- Track of the pin set: shared U (Fable, tied to U9 docker and CI scripts), tests T (Opus), act A (my round 1).
  - Evidence for U: U9 owns the docker and CI scripts (plans/roadmap/wave-4-7.md:120) and both backend and oracle read the pin.
  - Evidence for A: A16 owns the release checklist (plans/roadmap/waves.md, Release order).
- Whether partition opt-in gets a wave (Opus, me: wave 10) or stays a later row (Fable).

Final position: pin set in U with A16 recording it; partition opt-in a later row.
Confidence: medium-high on ownership and dependencies; medium on the wave 8 load, which carries three L items (dialect, library core, fsim step 1).
If wave 8 must shrink, fsim step 1 moves to wave 9 first, since nothing in the backend path waits on it.

## Roadmap items for everything the panel agrees on

Sizes use S = 1, M = 3, L = 6.

### U (shared)

| ID | Title | Depends on | Size | Wave |
| -- | ----- | ---------- | ---- | ---- |
| U16 | Toolchain pin set: released StableHLO version, build image, CI cache, prebuilt importer package | U9 | M | 8 |

### B (backend)

| ID | Title | Depends on | Size | Wave |
| -- | ----- | ---------- | ---- | ---- |
| B47 | `act-import`: C++ tool, StableHLO portable artifact to kernel text, coded errors by decision 40 | B42, B43, U16 | M | 8 |
| B48 | ISA dialect from the checked model: instruction op, semantics region, generated table and verifier; first task a spike on generic op versus per-ISA ODS | B13, U10, U16 | L | 8 |
| B49 | E-graph library core: arena, hash-cons, union-find, deferred rebuild, class data, read-only view, property tests without egg | B29, B31, B7, B9 | L | 8 |
| B50 | Matcher and rule builder: backtracking by operator, early preconditions, applied-substitution set | B49, B6 | M | 9 |
| B51 | Dual-engine harness and three-tier gate: egg no-ban schedule against the library | B49, B50, B32, B35, B3 | M | 9 |
| B52 | Engine swap: library behind the B2 C ABI, egg deleted, goldens re-recorded at most once | B51, B2, B10, B34 | M | 9 |
| B53 | MLIR bridge: portable artifact to library terms (typed handoff replaces the text bridge), chosen program to dialect ops | B47, B48, B49 | M | 9 |
| B54 | Torch adapter package: `ExportedProgram` to portable artifact; pairing spike picks torch-mlir or the core ATen walker | B47, B42 | M | 9 |
| B55 | C++ driver: reader, engine, extraction, allocator, cost in one process; Rust crate deleted | B52, B53 | L | 10 |

### A (act)

| ID | Title | Depends on | Size | Wave |
| -- | ----- | ---------- | ---- | ---- |
| A29 | `torch.compile` backend package: registration, strict mode, `dynamic=False`, cache, run on the oracle, release per torch minor | B54, A23 | M | 9 |

### O (oracle)

| ID | Title | Depends on | Size | Wave |
| -- | ----- | ---------- | ---- | ---- |
| O34 | fsim-lib step 1: C++ emitter per instruction from the dialect semantics region, no Python trace | O29, O3, B48, U16 | L | 8 |
| O35 | fsim-lib steps 2 to 4: `Pipeline::submit` and `sync`, debug print off `custom_call`, PJRT pinned by header and plugin hash; jax leaves the run | O34 | L | 9 |

### T (tests)

| ID | Title | Depends on | Size | Wave |
| -- | ----- | ---------- | ---- | ---- |
| T39 | StableHLO corpus exported once from jax; differential against the HLO path on identical chosen assembly | B47, T13 | S | 8 |
| T40 | Torch tests: eager PyTorch against ACT plus oracle per operator, tolerances per dtype, coverage table over the input specification | A29, T29 | M | 9 |
| T41 | Oracle-on-fsim gate: every golden and norm row holds | O35 | M | 9 |
| T42 | Platform gate: B3 on the C++ driver, oracle on fsim | B55, T41 | M | 10 |

### Later rows, no wave, no dependent

- Partition to eager as an opt-in with an undelegated-op report (after A29 and B46).
- Eager PrivateUse1 test device over the oracle, revived only for the upstream ACCELERATOR suite as a conformance run (torch-backends.md:63).
- PJRT plugin for JAX over the compiler and the C++ oracle.
- Allocator as an MLIR pass, at a second use.
- fsim-lib step 5 (own interpreter), out under decision 28.

## Text changes to earlier items, now

| Item | Change |
| ---- | ------ |
| B29 | Deterministic mode runs the no-ban schedule, numbers candidates in canonical term order, and `search` returns one record (stop reason, candidates with costs, chosen assembly); one golden re-record covers all three |
| B31, B2 | Typed payload is plain data with a size and offset test, owned as the e-node schema of B49 |
| B7, B9 | Class data (shape, dtype, constant value) is the schema B49 copies; constant marks live in class data, not as extractor mutations (extractor/utils.rs:44-60) |
| B6 | Rule record is language-neutral; preconditions take bound names and class data, with no egg type in a signature |
| B35, B8 | Ties break by declaration index; tests assert terms, not class ids |
| B42 | Input specification names StableHLO as the MLIR input, its version window, ranked static-shape tensors, and the dtype list; the rest stops with a coded error |
| B43 | Detection reads portable artifacts; the staged readers become B47 and B54; the torch.fx reader is B54's walker case |
| O29 | The study plans O34 and O35 and adopts the state-to-state `@main` shape (plans/oracle/fsim/integration.md:43-49) |
| A16 | Release checklist records the U16 pin set and each torch minor the package supports |

## Escalation to the owner

None.
