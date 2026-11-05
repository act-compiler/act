# Panel 12, round 2, Fable

Lens: ML systems and PyTorch backend architect.
Code paths are relative to /home/user; plan paths are relative to /home/user/act; brief paths are relative to plans/meeting/panel-12/research/.

## Revisions to round 1 after the briefs and owner decision 46

- Q12.3 step 4 argued against Rust engines behind FFI as "option C"; owner decision 46 settles the source, so only the design of the in-house library and the migration remain open.
- Q12.1 step 4 put the instruction's region constraints into PDLL native constraints; with selection inside the in-house e-graph no MLIR pattern driver runs, and PDLL "is still under active development" (mlir-ingest.md:31), so constraints are ODS verifiers plus e-graph preconditions, and PDLL leaves the plan.
- Q12.1 step 2 had the C++ reader parse "StableHLO and HLO through MLIR"; MLIR parses stablehlo and func, not XLA HLO text (mlir-ingest.md:11, 63), so HLO text stays on the Rust reader until the examples are re-exported as StableHLO and that reader is deleted.
- Q12.4 step 2 asked for U15, a serialized `Model`; the dialect and the rule tables are emitted by the Python generators at generation time (round 1 Q12.1 step 5), so no C++ consumer reads the `Model` at run time, U15 is withdrawn, and U2's caveat stays deferred (plans/shared/plan/U2.md:15).
- Q12.2 step 11 scheduled a PJRT plugin in wave 9; the PJRT C API is at major version 0 (torch-backends.md:98) and its only PyTorch door is archived on 15 Nov 2026 (torch-backends.md:94), so the plugin becomes a later row with no wave.
- Q12.2 step 9 scheduled an eager device over the oracle; the runtime surface it needs and a cross-repository CI relay that works only at its silent level (torch-backends.md:58-65) make it a later row too.

### Q12.1

Strongest opposing argument: the first C++ step must change nothing inside the Rust pipeline; Sonnet's B47 is a C++ tool that prints the kernel text today's reader accepts (Sonnet Q12.1 step 4), and Opus's N3 is a C++ `main` that owns the allocator and drives egg through the B2 C ABI (Opus Q12.1 step 2).

1. I concede the text bridge as step one: it needs no FFI into egg, and B3 compares it on identical chosen assembly and `--log` files (plans/backend/plan/architecture.md:37).
2. The bridge loses nothing the backend reads: StableHLO shapes "cannot be unranked" (mlir-ingest.md:65) and the backend is static-shape, so Sonnet's own dynamic-shape caveat does not bite.
3. The bridge reads portable artifacts, since the guarantees exclude prettyprinted text and raw bytecode (mlir-ingest.md:75); text input stays only for hand-written tests.
4. Opus's N3 is premature: the allocator already runs behind FFI from the Rust process (act-backend/act_backend/generic/src/malloc/cpp.rs:15-24) and B2 types that handoff (architecture.md:25-26); a C++ `main` before the engine moves adds a second FFI direction with no gate benefit.
5. On the dialect form I join Opus's open point: per-ISA ODS against one generic op is decided by measured build time, and no brief page gives an MLIR build time for a stock tree (mlir-ingest.md:171), so the B48 spike measures it; generated ODS is the default because `irdl.c_pred` cannot register at run time (Sonnet Q12.1 step 5).
6. Opus's "one region as both the rule's left-hand side and the oracle body" (Opus Q12.1 step 5) holds for the oracle half only: fsim runs the region as StableHLO (plans/oracle/fsim/native-path.md:32-34), but a selection rule is a tensor-operator pattern plus a precondition (plans/backend/plan/isel.md:3, :51), and matching a whole region is what PDL cannot do (https://arxiv.org/html/2505.09363v2).
7. Sonnet's B49 and B53, the allocator, selection, and extraction as MLIR passes, pay MLIR op construction per candidate: candidates are ranked by an external C cost model on emitted assembly (egraphs.md:167) and placement is chosen inside extraction keyed by (class, buffer) (egraphs.md:169-170); the allocator keeps the B2 flat struct, and MLIR appears at import and at the chosen program (Opus Q12.3 step 10).

Final position: option A in four gated steps: the B47 text bridge over StableHLO portable artifacts; B48, the ISA dialect generated as ODS and C++ from the `Model` with its semantics region for fsim; the in-house e-graph behind the B2 C ABI (Q12.3); then one C++ driver owning reader, engine, extraction, allocator, and cost, with the Rust crate deleted; the front end stays Python.
Confidence: high on A and on the front end staying Python; medium-high on the text bridge as step one.
What still divides the panel: the text bridge against a C++ `main` as the first step; whether the allocator becomes an MLIR pass.

### Q12.2

Strongest opposing argument: the torch package partitions by the B46 exclusion report and the coverage table, sends supported subgraphs to ACT, leaves the rest to eager, and offers strict mode as an option (Opus Q12.2 step 5).

1. The default is strict, as Sonnet also holds (Sonnet Q12.2 step 6): decision 40 stops invalid input, and TorchNeuron's compile path fails by default on an op it cannot lower (torch-backends.md:121), while Gaudi's CPU fallback is an eager-mode behavior (torch-backends.md:122).
2. Partition becomes a later opt-in with a printed report of undelegated ops, which the engineering rules allow because it is not silent; ExecuTorch's `partition` plus `preprocess` is the form (torch-backends.md:71-75).
3. Dynamo graph-breaks on untraceable code before any backend sees the graph (torch-backends.md:30), so strict means the adapter documents `fullgraph=True` and errors with a coded message on any op outside the input specification.
4. Opus's fallback producer and Sonnet's cross-check producer, `torch_xla.stablehlo`, come from a project deprecated and archived on 15 Nov 2026 (torch-backends.md:94); a one-time corpus cross-check is fine, a CI dependency is not under decision 28.
5. torch-mlir pins a torch nightly (torch-backends.md:89) and publishes dated PyPI releases (mlir-ingest.md:101), so the adapter is its own Python package with its own pins that writes a portable artifact across a file boundary (Sonnet Q12.2 step 5); whether a released torch-mlir wheel runs on a released torch is the first check of the adapter item and decides whether it ships under decision 33.
6. The ahead-of-time door is primary and public: `torch.export` with `run_decompositions(decomp_table=None)` gives core ATen (torch-backends.md:43); the `torch.compile` door wraps it with the documented `aot_autograd` helper, which lives under the private `torch._dynamo` namespace (torch-backends.md:34) and is retested on each two-monthly minor release (torch-backends.md:35).
7. Shapes: automatic dynamic recompiles with dynamic dimensions (torch-backends.md:28) while the backend is static-shape, so the adapter sets `dynamic=False` as TorchNeuron does (torch-backends.md:121) and caches compiled kernels by artifact hash.
8. I concede the eager device to Opus and Sonnet as a later row with no wave; the one reason to revive it is the upstream ACCELERATOR test suite run through `PYTORCH_TESTING_DEVICE_FOR_CUSTOM` (torch-backends.md:63) as a conformance suite over the generated compiler and oracle.

Final position: option A scheduled, with the eager device and the PJRT plugin as later rows; the adapter is `ExportedProgram` to StableHLO portable artifact through torch-mlir in its own package; strict by default; partition as a later opt-in with a report.
Confidence: high on A over B; medium on torch-mlir shipping under decision 33 pins.
What still divides the panel: strict default against partition with a strict option; whether `torch_xla` stays as a cross-check producer.

### Q12.3

Strongest opposing argument: relational e-matching with generic join and semi-naive deltas from the first version, because today's bare-variable rules fire on every class every iteration, and extraction outside the library through a read-only class view (Opus Q12.3 steps 4 and 8).

1. The measured gains of generic join are at 1e5 to 3e6 e-nodes on complex, non-linear patterns with small outputs (egraphs.md:53-54); ACT stops at 10,000 e-nodes (egraphs.md:159) with patterns of depth 1 to 3 (Opus Q12.3 step 1).
2. ACT's heaviest rules are bare-variable identities (egraphs.md:155, plans/backend/plan/search.md:4), the dense, large-output case where the paper reports little or no speedup (egraphs.md:55).
3. A bare-variable pattern has one atom and no join; what removes its waste is firing identities once per class at creation, the hook egg calls `modify` (egraphs.md:12), or B32's guards (search.md:27), not the matcher.
4. Semi-naive needs the database representation (egraphs.md:37, :57); a backtracking matcher gets the same effect from a per-iteration list of new e-nodes indexed by operator over an insertion-ordered arena.
5. Two concessions to Opus and Sonnet: the matcher interface is "yield substitutions", so generic join is a local swap (Sonnet Q12.3 step 7); and preconditions run as soon as their variables bind, inside the match, which is egglog's advantage over egg's post-match guards (egraphs.md:112-113) and costs a backtracking matcher nothing.
6. B34 measures where compile time goes (search.md:38); if matching exceeds allocation plus cost on the examples, generic join is the library's second step, not a redesign.
7. Opus's read-only view is right and already planned: both extractors today add `DetectedConst` e-nodes (egraphs.md:166), and B9 moves constant-ness into class data so `extract` takes `&EGraph` (isel.md:41-47), which is why B9 precedes any library work.
8. Extraction stays outside the engine and inside the backend: repeat-free enumeration under a limit (egraphs.md:164-165) and the (class, buffer) key (egraphs.md:170) are selection policy; the library exposes class members, class data, and canonical ids; optimal extraction is NP-hard (egraphs.md:119) and ACT does not need it, since an external cost ranks candidates (egraphs.md:167).
9. Sonnet's deferred rebuild is kept: 88x on congruence and 21x on saturation (egraphs.md:11); the core is small, since Luau's header held union-find, `make`, `join`, and `rebuild` in about 250 lines (egraphs.md:95), so the cost sits in the matcher, the preconditions, and the goldens.
10. Payloads join the e-node key, since `matches` compares only the variant today (egraphs.md:143) and conditions re-parse strings (egraphs.md:142); B31's typed payload is that key (isel.md:20).
11. Gate fail point one: today's `merge` returns `b_merged = false` against egg's contract (egraphs.md:147-148), so egg may skip a re-propagation that a correct library performs; a differing example at the swap can be a correct change, re-recorded once in a commit of its own (search.md:17), never hidden.
12. Gate fail point two: the `BackoffScheduler` bans rules (egraphs.md:19) and shapes what a node-limit stop cuts; I propose B29's deterministic mode runs egg's no-ban `SimpleScheduler` (https://docs.rs/egg/latest/egg/struct.SimpleScheduler.html), since B29 already holds one mode-equivalence check with one golden re-record (search.md:17), so any engine can reproduce the schedule; B32 lands before the swap so node-limit stops are rare, and where they remain the gate compares cost and oracle result (Sonnet Q12.3 step 12).

Final position: option B, a standalone C++ library: arena of hash-consed e-nodes with typed payloads in the key, union-find with path compression and no explanations, deferred rebuild, class data with `make`, `merge` as a coded fault, and `modify`, a backtracking matcher indexed by operator with early preconditions, a read-only extraction API, and an MLIR bridge at import and at the chosen program; migration behind the B2 C ABI under B29's deterministic mode with egg and the library side by side.
Confidence: high on B; medium-high on backtracking first; medium on the no-ban scheduler inside B29.
What still divides the panel: generic join first or second; whether the deterministic mode may change scheduler before the swap.

### Q12.4

Strongest opposing argument: Sonnet's table starts the library core in wave 8 beside the importer, puts the torch path in wave 9 and fsim in wave 10, and places the pin set in the act track (Sonnet Q12.4 table); Opus adds a toolchain image in the tests track and a serialized `Model` with a C++ loader (Opus Q12.4 N1, N2).

1. I concede the library core to wave 8: its dependencies B29, B31, B7, B9, and B6 sit in waves 1 to 3 (plans/roadmap/wave-1.md:24, wave-2.md:21-23, wave-3.md:12-13), and under decision 46 it is the largest item with no vendor to shorten it, so it starts early with its own unit tests.
2. U15 is withdrawn (revisions above), and Opus's N2 with its nlohmann loader goes with it; nlohmann/json is in the build for the allocator only (architecture.md:42).
3. The pin set is one shared-track item beside U9, which owns the docker and CI scripts (plans/roadmap/wave-4-7.md:122); it pins released StableHLO and torch-mlir versions and inherits their LLVM commit (mlir-ingest.md:127), which satisfies decision 33, and A16's release checklist records it (Sonnet Q12.4 step 6).
4. B43's staged readers (wave-4-7.md:28) become the text bridge; all three round-1 files hold this.
5. fsim joins through O29 (wave-3.md:28) and the four native-path steps (native-path.md:32-41): steps 1 and 2 in wave 8 with the emitter reading the B48 semantics region, steps 3 and 4 in wave 9, and step 5 out under decision 28.
6. No platform item precedes M5; wave 7 holds only follow-ons with no dependent (wave-4-7.md:118), so these open waves 8 to 10, as both other files hold.

| Item | Track | Depends on | Wave |
| ---- | ----- | ---------- | ---- |
| Toolchain pin set: released StableHLO and torch-mlir, build image, CI cache | U | U9 | 8 |
| B47 text bridge: C++ reader of StableHLO portable artifacts printing the kernel text | B | B42, B43, pin set | 8 |
| StableHLO corpus exported once from jax, differential against the HLO path | T | B47, T13 | 8 |
| B48 ISA dialect generated as ODS and C++ from the `Model`, with semantics region | B | B13, U10, pin set | 8 |
| E-graph library core: arena, union-find, rebuild, class data, matcher, unit tests | B | B29, B31, B7, B9, B6 | 8 |
| fsim native-path steps 1 and 2 | O | O29, O3, B48, pin set | 8 |
| Engine swap: library behind the B2 C ABI, dual-engine gate, egg deleted | B | core, B2, B10, B32, B35, B3 | 9 |
| Torch adapter package: `ExportedProgram` to portable artifact through torch-mlir, strict | B | B47, B42 | 9 |
| `torch.compile` registration, cache, and torch tests on the oracle | A, T | adapter, A23, T29 | 9 |
| fsim native-path steps 3 and 4 | O | fsim steps 1 and 2 | 9 |
| C++ driver: reader, engine, extraction, allocator, cost in one process; Rust crate deleted | B | engine swap, B48 | 10 |
| Platform gate: B3 on the C++ driver, oracle on fsim | T | driver, fsim steps 3 and 4 | 10 |
| Later, no wave: eager device over the oracle, PJRT plugin, partition opt-in | -- | -- | -- |

7. Earlier items constrained now, each a text change: B29 returns one record and runs the no-ban scheduler in deterministic mode (search.md:15, :17); B31 and B2 define the typed plain-data payload that becomes the e-node (architecture.md:25); B6 prefers the data form of rules (isel.md:53); B9 lands before any library work (isel.md:41); B35 and B8 break ties by declaration index and assert terms (isel.md:66, :73); B43 detects portable artifacts; A16 records the pin set.

Final position: later waves of the backend, shared, oracle, act, and tests tracks, waves 8 to 10 after M5, with the table above and no new track.
Confidence: medium-high on ownership and dependencies; medium on wave 8 load, which carries three L items.
What still divides the panel: the track of the pin set; whether the engine swap is wave 9 or 10; nothing on the order of the torch path after M5.

## Escalation to the owner

None.
