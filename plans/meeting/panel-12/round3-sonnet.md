# Panel 12, round 3: Sonnet (MLIR and torch-mlir integration lens)

Read: all six round files, the three briefs (egraphs.md, mlir-ingest.md, torch-backends.md), decisions 23, 28, 33, 39 to 46, plans/backend/plan/search.md.
New IDs were free in a grep of plans/ outside meeting/ (B47 to B58, U15 to U17, T39 to T42, O34 to O37, A29 to A31); an author re-checks.
Concessions carried in: the allocator stays a C++ library on the B2 flat struct (my MLIR-pass item is withdrawn); extraction stays outside the e-graph library; the text bridge is step one only; relational matching is not built first.

## Q12.1: C++ on MLIR, and in what order

Agreed:
- Option A in strangler form; the front end stays Python and keeps `check` once (decision 42).
- Order: importer, dialect, library core, typed handoff, engine swap, one C++ driver with the Rust crate deleted.
- Step one is a C++ tool that reads a StableHLO portable artifact and prints the kernel text the Rust reader accepts; Opus and Fable conceded it over a C++ `main` around egg.
- Portable artifacts are the supported input, since only they carry the 5-year and 2-year guarantees (mlir-ingest.md:72-75); text is a test path, and HLO text stays on the Rust reader until re-exported (mlir-ingest.md:11, 63).
- MLIR C++ has no API stability (mlir-ingest.md:58): MLIR code is confined to the importer, dialect, and bridge, and the library includes no MLIR header.
- Selection is not an MLIR pattern pass: the greedy driver rewrites destructively (mlir-ingest.md:35) and selection needs saturation's alternatives (pipeline.rs:488-495).
- ACT is not an IREE plugin (dynamic plugins need the host's exact IREE and LLVM revisions, mlir-ingest.md s4.2); IRDL is out because `irdl.c_pred` cannot register at run time.

Divides, one point: how ISA knowledge enters the MLIR code.
- Per-ISA generated ODS (Opus, Fable): generate-then-build already exists (act-backend/README.md:94) and typed ops carry a generated verifier and semantics region.
- Generic instruction op (me): per-ISA ODS needs MLIR headers and `mlir-tblgen` on the architect's machine, about 147 MB installed (mlir-ingest.md s5), and selection uses no MLIR patterns, so typed ops add only the verifier.

Final position: option A, order as above.
- The dialect is one generic instruction op plus a semantics region, driven by a data table the Python generator writes, like today's generated Rust.
- A generator output is not the `Model`, so U2's "second non-Python consumer" caveat (plans/shared/plan/U2.md:15) stays closed and no `Model` is serialized.
- The dialect item opens with a spike: per-ISA ODS wins only if its build and install cost is small against a prebuilt MLIR; the result lands before the bridge and the fsim emitter start.
- Confidence: high on A and the order; medium on generic op over per-ISA ODS, since no brief measured a stock MLIR build (mlir-ingest.md:171).

## Q12.2: Portable PyTorch backend

Agreed:
- Option C with the compile path first; no fork; a `torch.compile` backend registered through `torch_dynamo_backends` under `aot_autograd` with core ATen (torch-backends.md:20-26).
- The handoff is a StableHLO portable artifact across a file boundary, into the Q12.1 importer.
- Strict by default (decision 40): an op outside the input specification stops with a coded error; `dynamic=False` as TorchNeuron does (torch-backends.md:121); a cache keyed by artifact hash and ACT version.
- Partition to eager is a later opt-in with a report of undelegated ops, never silent.
- The PrivateUse1 device and a PJRT plugin are later rows with no wave: the eager minimum needs runtime pieces (torch-backends.md:58-65) and PyTorch/XLA, the only PyTorch door to PJRT, is archived 15 Nov 2026 (torch-backends.md:94).
- `torch_xla` is neither a producer nor a CI dependency.
- The package supports the latest two torch minors and ships per minor, since `torch._dynamo` is private and minors arrive every two months (torch-backends.md:34-35).
- Every torch test is an oracle-differential test against eager PyTorch.

Divides: the producer from `ExportedProgram` to StableHLO.
- For torch-mlir (Fable, me): AWS Neuron ships StableHLO from a vendored torch_mlir (torch-backends.md:110), and its per-operator lowering already handles broadcasting and type promotion that a walker must reimplement.
- For an ACT-owned walker (Opus): torch-mlir pins `torch==2.15.0.dev20260917`, a nightly (torch-backends.md:89), and decision 33 forbids pre-release pins.

Final position: option C, compile path first; the adapter interface is `ExportedProgram -> portable artifact` and a pairing spike picks the producer.
- Spike: does torch-mlir release 20261101 (mlir-ingest.md:101) import an `ExportedProgram` from released torch 2.15.0 and cover the input specification?
- Pass: torch-mlir is the producer, in its own package and environment, run as a subprocess.
- Fail: an ACT-owned core ATen walker maps each input-specification node to StableHLO and stops with a coded error on any other; it is also B43's torch.fx reader.
- The producer calls `serialize_portable_artifact(module, target_version)` with ACT's pinned StableHLO version (mlir-ingest.md:81-82), so torch-mlir's LLVM submodule and ACT's LLVM commit never meet in one process.
- Opus's change-of-mind condition is the spike's pass case, so the panel agrees on the decider.
- Confidence: high on structure and strict; about 50 percent on which producer ships.

## Q12.3: C++ e-graph library and migration

Agreed (option B, in-house per decision 46):
- A standalone library, no MLIR header; the bridge touches MLIR at import and at the chosen program only.
- Option A (eqsat dialect) fails on evidence: whole-graph CSE for congruence, no benchmarks, no analysis or extraction, PDL cannot match regions (egraphs.md s4).
- Kept: arena of hash-consed e-nodes with typed payloads in the key; union-find with path compression and union by size; deferred rebuild (88x and 21x, egraphs.md:11).
- Kept: class data with `make` and a `merge` that is a coded fault and reports change correctly; preconditions as predicates over bound class data, run as soon as variables bind.
- Kept: matches with e-node ids; an applied-substitution set per rule; the three limits with stop reasons; a read-only class view.
- Dropped: explanations, a run-time rule language, Datalog functions, threading, `BackoffScheduler`.
- Matcher: backtracking over a per-operator index, behind a "yield substitutions" interface so generic join is a local swap.
- Extraction stays in the backend with the (class, buffer) policy and the external cost model (egraphs.md:167-170).
- B9 and B32 land before the swap: B9 makes constant-ness class data so nothing clones the e-graph (pipeline.rs:235); B32 removes the bare-variable fan-out (search.md:4).
- Migration runs egg and the library side by side behind the B2 C ABI; goldens are re-recorded at most once, in their own commit.

Divides:
- Gate for kernels that stop at `NODE_LIMIT`: identical assembly and `--log` (Fable, Opus) against cost no worse plus equal oracle result (me).
  - For identical: B29 already demands identical assembly and `--log` across machines (search.md:23).
  - For cost plus oracle: a cut depends on egg's rule order, which an independent engine does not reproduce (egraphs.md:19).
- Generic-join trigger: Opus said 5 percent of compile time, then a third.

Final position: option B as above, with a three-tier gate.
- Tier 1: library property tests (congruence, hash-cons, analysis conflicts), no egg.
- Tier 2: kernels that saturate give equal sorted canonical term sets per class, and identical assembly and `--log` once B29 numbers candidates in canonical term order and B35 breaks ties by declaration index.
- Tier 3: kernels that still stop at the node limit give cost no worse than egg and an equal oracle result.
- B29's deterministic mode runs egg's no-ban `SimpleScheduler` (cited by Opus and Fable); I concede it, since it is a defined, cheap schedule with no ban logic to copy.
- Generic join comes forward when a B34 profile after B32 shows matching above a third of search time; I withdraw 5 percent, which buys little for an L item.
- Confidence: high on B over A; medium-high on backtracking first; medium on the tier split; a rule set with patterns of eight or more nodes would change my mind.

## Q12.4: How the items join the roadmap

Agreed:
- No new track or plan; items join B, U, T, O, A as waves 8 to 10 after M5, since wave 7 holds only items with no dependent (plans/roadmap/wave-4-7.md:118).
- Wave 8: pin set, importer, dialect, library core, fsim step 1, corpus.
- Wave 9: matcher, dual-engine gate, engine swap, bridge, torch adapter and backend, fsim steps 2 to 4 with their gate.
- Wave 10: C++ driver with the Rust crate deleted, and the platform gate.
- No M date for waves 8 to 10 before the M1 rebaseline (plans/roadmap/waves.md:15).
- Pins are released versions (decision 33): StableHLO 1.20.0 as fsim-lib uses (plans/oracle/fsim/native-path.md:25), with its LLVM commit inherited from the tag.
- No serialized `Model` item: Opus, Fable, and I withdrew it because the dialect table is generator output.

Divides:
- Track of the pin set: U (Fable), T (Opus), A (my round 1).
  - For U: U9 owns the docker and CI scripts (plans/roadmap/wave-4-7.md:120), and backend and oracle both read the pin.
  - For A: A16 owns the release checklist (plans/roadmap/waves.md, Release order).
- Partition opt-in: wave 10 (Opus, Sonnet in round 2) or a later row (Fable).

Final position: pin set in U with A16 recording it; partition opt-in a later row.
- Confidence: medium-high on ownership and dependencies; medium on wave 8, which carries three L items; fsim step 1 moves to wave 9 first if it must shrink, since no backend item waits on it.

## Roadmap items the panel agrees on (S = 1, M = 3, L = 6)
Sizes: S = 1, M = 3, L = 6.
| ID | Title | Depends on | Size | Wave |
| -- | ----- | ---------- | ---- | ---- |
| U16 | Toolchain pin set: released StableHLO, build image, CI cache, prebuilt importer package | U9 | M | 8 |
| B47 | `act-import`: C++ tool, StableHLO portable artifact to kernel text, coded errors (decision 40) | B42, B43, U16 | M | 8 |
| B48 | ISA dialect from the checked model: instruction op, semantics region, generated table and verifier; opens with the ODS spike | B13, U10, U16 | L | 8 |
| B49 | E-graph library core: arena, hash-cons, union-find, deferred rebuild, class data, read-only view, tests without egg | B29, B31, B7, B9 | L | 8 |
| B50 | Matcher and rule builder: backtracking by operator, early preconditions, applied-substitution set | B49, B6 | M | 9 |
| B51 | Dual-engine harness and three-tier gate against egg's no-ban schedule | B49, B50, B32, B35, B3 | M | 9 |
| B52 | Engine swap: library behind the B2 C ABI, egg deleted, one golden re-record | B51, B2, B10, B34 | M | 9 |
| B53 | MLIR bridge: artifact to library terms (typed handoff replaces the text bridge), chosen program to dialect ops | B47, B48, B49 | M | 9 |
| B54 | Torch adapter package: `ExportedProgram` to portable artifact; pairing spike picks the producer | B47, B42 | M | 9 |
| B55 | C++ driver: reader, engine, extraction, allocator, cost in one process; Rust crate deleted | B52, B53 | L | 10 |
| A29 | `torch.compile` backend: registration, strict mode, `dynamic=False`, cache, run on the oracle, release per torch minor | B54, A23 | M | 9 |
| O34 | fsim-lib step 1: C++ emitter per instruction from the dialect semantics region | O29, O3, B48, U16 | L | 8 |
| O35 | fsim-lib steps 2 to 4: `Pipeline::submit` and `sync`, debug print off `custom_call`, PJRT pinned by header and plugin hash; jax leaves the run | O34 | L | 9 |
| T39 | StableHLO corpus exported once from jax; differential against the HLO path | B47, T13 | S | 8 |
| T40 | Torch tests: eager against ACT plus oracle per operator, per-dtype tolerances, coverage table over the input specification | A29, T29 | M | 9 |
| T41 | Oracle-on-fsim gate: every golden and norm row holds | O35 | M | 9 |
| T42 | Platform gate: B3 on the C++ driver, oracle on fsim | B55, T41 | M | 10 |
Later rows, no wave, no dependent:
- Partition to eager as an opt-in with an undelegated-op report (after A29 and B46).
- Eager PrivateUse1 device (only to run the upstream ACCELERATOR suite, torch-backends.md:63); PJRT plugin for JAX; allocator as an MLIR pass at a second use; fsim-lib step 5, out under decision 28.

## Text changes to earlier items, now
| Item | Change |
| ---- | ------ |
| B29 | Deterministic mode runs the no-ban schedule, numbers candidates in canonical term order, and `search` returns one record (stop reason, candidates with costs, chosen assembly); one golden re-record covers all |
| B31, B2 | Typed payload is plain data with a size and offset test, and is B49's e-node schema |
| B7, B9 | Class data (shape, dtype, constant value) is B49's schema; constant marks leave the extractor (extractor/utils.rs:44-60) |
| B6 | Rule record is language-neutral; preconditions take bound names and class data, no egg type in a signature |
| B35, B8 | Ties break by declaration index; tests assert terms, not class ids |
| B42 | Input specification names StableHLO as the MLIR input with its version window, ranked static-shape tensors, and a dtype list; the rest stops with a coded error |
| B43 | Detection reads portable artifacts; staged readers become B47 and B54; the torch.fx reader is B54's walker case |
| O29 | The study plans O34 and O35 and adopts the state-to-state `@main` shape (plans/oracle/fsim/integration.md:43-49) |
| A16 | Checklist records the U16 pin set and each torch minor supported |

## Escalation to the owner
None.
