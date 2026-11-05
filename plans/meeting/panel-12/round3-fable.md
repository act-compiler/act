# Panel 12, round 3, Fable

Lens: ML systems and PyTorch backend architect.
Read: the six round files and the three briefs; plan paths are relative to plans/, brief paths to plans/meeting/panel-12/research/.
New IDs are the next free in the roadmap tables (highest today: B46, O33, T38, U14, A28); an author re-checks before use.

### Q12.1

Where the panel agrees:
- Option A in strangler form; the front end stays Python; option B and C are out (all three round-2 files).
- Step one is a C++ importer over the pinned libStablehlo that reads portable artifacts and prints the kernel text today's reader accepts; text input serves tests only (round2-fable.md:19-21, round2-opus.md:43-44, round2-sonnet.md:57-58).
- The ISA dialect is generated from the `Model` by the Python generators; IRDL is out because `irdl.c_pred` cannot register at run time; PDLL is out because no MLIR pattern driver runs selection.
- MLIR stays at the importer, the dialect, and the bridge; the e-graph library includes no MLIR header, since the C++ API has no stability promise (mlir-ingest.md:57-58).
- ACT is not an IREE plugin; the public pin is a released StableHLO version and its LLVM commit is transitive (mlir-ingest.md:119, :127).
- Order: importer, then dialect and library core, then the engine swap, then one C++ driver with the Rust crate deleted.

What still divides:
- Per-ISA generated ODS (Fable, Opus) against one generic instruction op driven by an exported `Model` (Sonnet).
  For ODS: the verifier is generated checked 64-bit C++ from the constraints, the discipline `CheckedWriter` already enforces (backend/plan/isel.md:54), and ODS emits op class, getters, builders, and verifier from one record (mlir-ingest.md:18-20).
  For the generic op: per-ISA static linking puts MLIR headers and `mlir-tblgen` on the architect's machine, a 10-minute MLIR build and about 147 MB of libraries (mlir-ingest.md:128, :140).
- The allocator as an MLIR pass (Sonnet) against a C++ library the driver calls (Fable, Opus).
  For the pass: one IR end to end from import to chosen program.
  Against: candidates are ranked by an external C cost model on emitted assembly and placement is chosen inside extraction keyed by (class, buffer) (egraphs.md:167-170), so an MLIR op per candidate has no consumer.

Final position: option A as above; per-ISA generated ODS is the default and the B48 spike measures per-ISA build time against the generic op before the item commits; the allocator stays a library behind the B2 struct and MLIR appears at import and at the chosen program.
Confidence: high on A and the order; medium on the dialect form, which the spike decides.

### Q12.2

Where the panel agrees:
- The compile path is the one scheduled door; a PrivateUse1 eager device and a PJRT plugin are later rows with no wave (round2-fable.md:44, round2-opus.md:72, round2-sonnet.md:41).
- Strict by default under decision 40, as TorchNeuron's compile path does (torch-backends.md:121); partition is a later opt-in that prints every undelegated node like the B46 report.
- `dynamic=False`, symbolic dimensions rejected with a coded error naming the dimension, kernels cached by artifact hash (torch-backends.md:28, :121).
- The handoff is a StableHLO portable artifact across a file boundary into the importer, from a Python package with its own pins; torch-mlir's MLIR never links into the importer (mlir-ingest.md:102).
- `torch_xla` is dropped as producer or cross-check, since PyTorch/XLA is archived on 15 Nov 2026 (torch-backends.md:94).
- The `aot_autograd` hook is a private `torch._dynamo` import and minors ship every two months (torch-backends.md:34-35), so the package pins one released torch per act release with a CI row per supported minor.

What still divides:
- The producer: torch-mlir `fx.export_and_import` (Fable, Sonnet) against an ACT-owned FX walker from core ATen to StableHLO text (Opus).
  For torch-mlir: a dated release exists on PyPI (mlir-ingest.md:101) and TorchNeuron ships StableHLO through a vendored torch_mlir (torch-backends.md:110).
  For the walker: torch-mlir pins `torch==2.15.0.dev20260917`, a nightly (torch-backends.md:89), which decision 33 forbids in public, and the backend handles 18 operations (round1-sonnet.md:96).

Final position: one adapter item whose first step is the pairing check, torch-mlir release 20261101 importing an `ExportedProgram` from released torch 2.15.0; if it passes, torch-mlir is the producer; if not, the walker over the B42 input specification is, sized inside the item; vendoring torch-mlir is asked under decision 28 only if both fail.
Confidence: high on the compile path and strict default; medium on which producer the check selects.

### Q12.3

Where the panel agrees:
- Option B: a standalone in-house C++ library, MLIR at the bridge only (all three round-2 files; decision 46 closes adoption).
- Keep: arena of hash-consed e-nodes with the typed payload in the key, union-find with path compression and no explanations, deferred rebuild (egraphs.md:11), one class-data record with `make` and `merge` as a coded fault with a correct changed flag (egraphs.md:147-148), preconditions evaluated as soon as their variables bind (egraphs.md:112-113), a matcher that returns matched e-node ids with the substitution (egraphs.md:154), a per-op column store, an applied-substitution memo, a read-only view for extraction, insertion-ordered containers.
- Drop: explanations, a run-time rule language, Datalog functions, threading, the `BackoffScheduler`.
- Backtracking matcher first; generic join is a local addition behind the yield-substitutions interface when a B34 profile puts matching above 5 percent of compile time (round2-opus.md:95, round2-sonnet.md:110-111); the measured gains sit at 1e5 to 3e6 e-nodes (egraphs.md:53).
- Extraction stays in the backend, since no local cost exists (egraphs.md:167); Sonnet withdrew k-best in the library.
- B32 lands before the swap; B9 lands before any library work; goldens are re-recorded at most once in a commit of their own (backend/plan/search.md:17).

What still divides:
- The swap gate: identical chosen assembly and `--log` files under a no-ban schedule set in B29 (Fable, Opus) against canonical term sets for saturating kernels and cost plus oracle result for node-limit kernels (Sonnet).
  For the schedule switch: `SimpleScheduler` runs every rewrite every time, so after k iterations any engine that applies all matches then rebuilds yields the same e-graph (round2-opus.md:26).
  For the relaxed gate: egg's cut at a node limit depends on its rule order and ban schedule, which an independent engine cannot reproduce (egraphs.md:19).
- The `modify` hook: kept by Fable, dropped by Sonnet.

Final position: both gates, in order; B29's deterministic mode runs the no-ban schedule and orders candidate logs by canonical term text, so saturating and iteration-bounded kernels compare on canonical dumps per iteration plus identical assembly and logs; the few node-limit kernels left after B32 compare on cost no worse and an equal oracle result; `modify` is dropped, since B32's guards and the memo cover identities, and returns at a second need.
Confidence: high on B and the keep list; medium-high on the combined gate.

### Q12.4

Where the panel agrees:
- Waves 8 to 10 after M5, in the existing tracks, no new track; wave 7 keeps only items with no dependent (roadmap/wave-4-7.md:116).
- The library core starts in wave 8, since its inputs B29, B31, B7, B9, and B6 sit in waves 1 to 3 (roadmap/wave-1.md:24, wave-2.md:21-23, wave-3.md:12-13) and it gates the Rust deletion.
- The pin set is one released StableHLO version with a build image and CI cache; fsim-lib already pins 1.20.0 (oracle/fsim/native-path.md:25).
- fsim joins through O29 (roadmap/wave-3.md:28) and native-path steps 1 to 4 (oracle/fsim/native-path.md:32-41); step 5 stays out under decision 28; risk K9 already places the migration after M5 (roadmap/risks.md:26).
- Later rows with no wave: eager device, PJRT plugin, partition opt-in.
- Earlier items constrained now by text: B29 returns one record, runs the no-ban schedule in deterministic mode, orders candidates canonically; B31 and B2 define the plain-data typed payload that becomes the e-node (backend/plan/architecture.md:25-26); B6 writes rules as language-neutral records with a Rust emitter now; B9 precedes library work; B35 and B8 break ties by declaration index and assert terms; B32 precedes the swap; B43 detects portable artifacts and its staged readers become B47; U2 keeps the `Model` plain data; A16 records the pin set.

What still divides:
- The pin set's track: shared (Fable), tests (Opus), act (Sonnet).
  For shared: U9 owns the one copy of docker and CI scripts (roadmap/wave-4-7.md:120).
  For act: the A16 release checklist is where pins are recorded (roadmap/waves.md:49-57).
- A serialized `Model` item: Sonnet needs it for the generic op; Fable and Opus see no C++ reader of the `Model` (shared/plan/U2.md:15).
- Engine deletion in wave 9 (Fable, Sonnet) or wave 10 with the C++ passes (Opus).

Final position: the shared track owns the pin set and A16 records it; no `Model` export item until the B48 spike picks the generic op, in which case it enters wave 8 beside B48; the swap deletes egg in wave 9, because the swap and the deletion pass one gate, B3 on every example, and wave 10 deletes the Rust driver.
Confidence: medium-high on ownership and dependencies; medium on wave 8 load, three L items.

## Roadmap items the panel agrees on

Weights by the roadmap rule, S = 1, M = 3, L = 6: wave 8 is 25, wave 9 is 21, wave 10 is 12.
No date is set for waves 8 to 10 before the M1 rebaseline (roadmap/waves.md:16).

| ID | Title | Track | Depends on | Size | Wave |
| -- | ----- | ----- | ---------- | ---- | ---- |
| U15 | Toolchain pin set: released StableHLO version, MLIR and StableHLO static libraries in one build image, CI cache | U | U9 | M | 8 |
| B47 | StableHLO importer: C++ tool reading portable artifacts, printing the kernel text the reader accepts; text accepted for tests | B | B42, B43, U15 | M | 8 |
| B48 | ISA dialect generated from the `Model`: one op per instruction, typed attributes, generated verifier, StableHLO semantics region; spike measures per-ISA build time against one generic op | B | B13, U10, U15 | L | 8 |
| B49 | E-graph library core: arena, typed payload key, union-find, deferred rebuild, class data with coded-fault merge, per-op column store, read-only view, property tests | B | B29, B31, B7, B9, B6 | L | 8 |
| O34 | fsim native-path steps 1 and 2: C++ emitter per instruction reading the B48 semantics region; `Pipeline::submit` and `sync` replace `compile_and_load` and `execute` | O | O29, O3, B48, U15 | L | 8 |
| T39 | StableHLO corpus exported once from jax, differential against the HLO path on the same examples | T | B47, T13 | S | 8 |
| B50 | E-matcher with early preconditions and applied-substitution memo; dual-engine harness against egg under B29's deterministic mode | B | B49, B6, B8, B32, B3 | L | 9 |
| B51 | Engine swap: the Rust driver calls the library through the B2 C ABI; egg and its rule glue deleted; B3 holds on every example | B | B50, B2, B10, B35 | M | 9 |
| B52 | Torch adapter package: `ExportedProgram` with core ATen decompositions to a StableHLO portable artifact; strict; `dynamic=False`; pairing check picks torch-mlir or the walker; torch-mlir files enter the same way | B | B47, B42 | M | 9 |
| O35 | fsim native-path steps 3 and 4: debug print out of `custom_call`; PJRT pinned by header and plugin hash; jax, jaxlib, nanobind leave the run | O | O34 | M | 9 |
| A29 | `torch.compile` registration by entry point, kernel cache by artifact hash, one released torch per act release with a CI row per supported minor; A16 records the pin set | A | B52, A23 | M | 9 |
| T40 | Torch tests: eager PyTorch against ACT on the oracle per operator with per-dtype tolerance | T | A29, T29 | M | 9 |
| B53 | MLIR bridge: the importer builds library terms directly, retiring the text bridge; the chosen program is emitted as ISA dialect ops | B | B47, B48, B49 | M | 10 |
| B54 | C++ driver: importer, bridge, engine, extraction, allocator, cost in one process; Rust crate deleted | B | B51, B53, B46 | L | 10 |
| T41 | Platform gate: B3 on the C++ driver; every golden and norm row holds on fsim | T | B54, O35 | M | 10 |

Later rows, no wave, no dependent: eager device over the generated oracle; PJRT plugin as the JAX door beside fsim; partition opt-in with an undelegated-op report; generic join in B49's matcher.

## Escalation to the owner

None.
