# Panel 12, round 2, Opus (lens: e-graph and compiler architect)

Read: my round 1, round1-fable.md, round1-sonnet.md, the three briefs in research/, and owner decision 46.
Backend code paths are relative to act-backend/act_backend/generic/src/.

Round-1 claims I withdraw or revise, with the evidence that contradicts them:

- N2, a serialized `Model` with a C++ loader: withdrawn; generators stay Python and emit C++, so no C++ code reads the `Model` (round1-sonnet.md:42, plans/shared/interface.md:66).
- N3, a C++ driver around egg behind a C ABI: withdrawn; decision 45 runs the move "through a Rust FFI" into C++, and decision 46 deletes egg, so wrapping egg in C++ is throwaway.
- N6 waiting on N3: wrong; the in-house library needs no MLIR, so its core starts in wave 8 beside the importer.
- torch_xla as the fallback converter: withdrawn; PyTorch/XLA is deprecated and archived on 15 Nov 2026 (torch-backends.md:94).
- Q12.2 point 5, partition to eager by default: withdrawn; the default is strict, partition is an opt-in.
- Relational e-matching from the start: revised to a backtracking first version; the gains are measured at 1e5 to 3e6 e-nodes on complex patterns (egraphs.md:53-55), and ACT runs at 1e4 with depth 1 to 3.
- Relational matching as the fix for bare-variable rules: wrong; `(?x)` is cheap to match, the waste is re-applying, which B32's guards and an applied-substitution set remove.
- Option C (DialEgg or any external engine): moot under decision 46.

### Q12.4

Strongest opposing argument (round1-fable.md:75): two waves suffice, wave 8 for readers, dialect, and fsim, wave 9 for the engine, and a third wave only adds a gate.

1. Concession on the dependency graph: Sonnet's table is right that the library core depends only on B31, B7, B9, and B29 (round1-sonnet.md:23), since decision 46 makes it a standalone C++ library.
2. So wave 8 holds the importer, the toolchain pin, the library core, and fsim steps 1 and 2.
3. Wave 9 holds the matcher with its dual-engine harness, the dialect generator, the torch path, and fsim steps 3 and 4.
4. Wave 10 holds the C++ passes, the Rust crate's deletion, and the platform gate; the deletion needs the B3 gate on every example (plans/backend/plan/architecture.md:36-45), so a third wave is a real dependency, not a buffer.
5. Earlier items, revised constraints now:
   - B29: the deterministic mode also fixes the rule schedule; egg's `SimpleScheduler` "runs every rewrite every time" and is set by `with_scheduler` (docs.rs/egg/latest/egg/struct.SimpleScheduler.html), while `BackoffScheduler` bans rules by match counts (egraphs.md:19), which a second engine cannot reproduce without copying it.
   - B32: its verification, "stop reason is saturation or iteration limit, not node limit" (plans/backend/plan/search.md:31), is what makes the schedule above safe; B29 switches the schedule only once B32 holds.
   - B9: constant marks become class data, because extraction today clones the whole e-graph per pass (pipeline.rs:235) to add `DetectedConst` e-nodes (isel/extractor/utils.rs:44-60).
   - B35: ties break by declaration index, so engines with different class ids choose the same assembly (unchanged).
   - B6: rules are language-neutral records, with a Rust emitter now and a C++ emitter later (round1-sonnet.md:41); I adopt this wording over my "data form".
   - B43: keep its third reader, torch.fx (plans/backend/plan/failures.md, B43 table); Sonnet would drop it (round1-sonnet.md:98), and Q12.2 below needs it.
6. Pins and decision 33: StableHLO builds against one LLVM commit (mlir-ingest.md:127), and decision 33 forbids commit pins in public.
   The public pin is a released StableHLO version, fsim-lib's 1.20.0 (plans/oracle/fsim/native-path.md:25), and its LLVM commit is a transitive fact of that release; Sonnet's A33, "one LLVM and StableHLO commit per release" (round1-sonnet.md:20), names a commit and should be reworded.

Final position: waves 8 to 10 of the existing tracks, with the library core in wave 8.
Confidence: medium-high on dependencies; medium on whether the transitive LLVM commit satisfies decision 33.
What still divides: two waves or three; whether `Model` serialization is ever scheduled (I now say no item until a C++ reader of the `Model` exists).

### Q12.1

Strongest opposing argument (round1-sonnet.md:57-62): step 1 is a standalone C++ tool that parses StableHLO and prints the kernel text the Rust reader accepts; it delivers decision 43's input soonest, needs no ABI, and keeps MLIR out of the cargo build.

1. Concession: this beats my N3 as step 1; it touches neither search nor allocation, and T39 checks it against the HLO path on the same examples (round1-sonnet.md:21).
2. The tool reads portable artifacts as the documented input, since only they carry the 5-year and 2-year guarantees (mlir-ingest.md:72-75); text is accepted for tests within one act release (round1-fable.md:12).
3. The C++ API is not stable: "LLVM does not have a stable API", so Triton targets one LLVM revision per version (mlir-ingest.md:58).
   This argues for confining MLIR to the importer, the dialect, and the bridge, and keeping the e-graph engine free of MLIR types.
4. Order: importer and library core (wave 8); dialect generator and the Rust driver calling the library through FFI, egg deleted (wave 9); selection, extraction, and allocation as C++ passes, Rust driver deleted (wave 10).
5. That middle step is decision 45 read literally: Rust calls C++ (round1-sonnet.md:25), the reverse of my round-1 direction.
6. Rewrite drivers: MLIR's greedy driver applies the highest-benefit pattern destructively (mlir-ingest.md:35); selection needs the alternatives saturation keeps (pipeline.rs:488-495), so MLIR patterns serve canonicalization at the edges, never selection.
7. The dialect is generated ODS, one op per instruction, as all three round-1 files agree; IRDL fails because `irdl.c_pred` cannot register at run time (round1-fable.md:14).

Final position: option A, strangler order as above, front end stays Python.
Confidence: high on order; medium on per-ISA ODS versus one generic op, which no brief measured (mlir-ingest.md:171-172).
What still divides: whether the allocator becomes an MLIR pass (Sonnet's B49) or stays a C++ library the passes call; I see no need for MLIR there and would defer it to a second use.

### Q12.2

Strongest opposing argument (round1-fable.md:32, round1-sonnet.md:89): torch-mlir's `export_and_import` turns an `ExportedProgram` into StableHLO, reuses the StableHLO importer, and is the durable converter now that torch_xla is archived; iree-turbine and TorchNeuron ship this shape.

1. Rebuttal on coupling: torch-mlir pins `torch==2.15.0.dev20260917`, a nightly (torch-backends.md:89), and is outside any LLVM release (torch-backends.md:88).
   Decision 33 allows "no pre-release tags" in public releases, so a public act torch package cannot require it.
2. Its LLVM is its own submodule (mlir-ingest.md:102, :130), a second commit beside StableHLO's, so it cannot link into the importer.
3. Field evidence is mixed: Neuron vendors torch_mlir, Tenstorrent marks its torch-mlir-based TT-Torch deprecated (torch-backends.md:110, :124), and FX import of one model took over 20 minutes (mlir-ingest.md:100).
4. The needed coverage is small: the backend handles 18 operations (round1-sonnet.md:96), and core ATen is fully functional with no in-place or `_out` variants (torch-backends.md:46).
5. B43 already plans a torch.fx reader mapping graph nodes to the operation table (plans/backend/plan/failures.md, B43 table), and decision 40 names torch.fx as an input.
6. Final shape: a `torch.compile` backend under `aot_autograd` with core ATen decompositions (torch-backends.md:23-26) hands the graph to an ACT-owned Python FX reader.
   That reader maps each core ATen node in the input specification to StableHLO text for the importer, and stops with a coded error naming any other node.
   Its only dependency is a released torch.
7. torch-mlir stays the route for torch-mlir input files (B43's second reader), as a producer across a file boundary (round1-sonnet.md:74), asked under decision 28 when it lands.
8. Decision 40: strict by default, as TorchNeuron does (torch-backends.md:121) and as Fable argues (round1-fable.md:35); partition to eager is an opt-in that reports the undelegated ops like the B46 exclusion report.
9. Concession to Fable on the private API: the `aot_autograd` hook lives under `torch._dynamo` (torch-backends.md:34), and PyTorch ships every two months (:35); the package pins one released torch per act release and a CI row per supported torch.
10. PrivateUse1 and an oracle-backed eager device: a later row with no wave; the PJRT plugin is a JAX door and belongs with fsim, not Q12.2.

Final position: option C with the compile path first, through an ACT-owned FX reader; torch-mlir only for torch-mlir input.
Confidence: high on compile path and strict default; medium on the FX reader over torch-mlir, which is a dependency and coverage call.
What would change my mind: a torch-mlir release that pins a released torch, or a coverage table where the FX reader needs more than about 50 core ATen ops.
What still divides: torch-mlir as default producer (Fable, Sonnet) against an in-house reader (me).

### Q12.3

Strongest opposing argument for option A: one IR end to end, PDLL patterns with native C++ constraints (mlir-ingest.md:30), attributes as metadata as DialEgg shows (egraphs.md:87), and a persistent e-graph across IR levels (egraphs.md:84).

1. Rebuttal: the eqsat dialect reports no benchmarks, keeps congruence by whole-graph CSE, and backtracks in a `pdl_interp` interpreter (egraphs.md:82-83).
2. An engine on MLIR objects inherits the unstable C++ API (mlir-ingest.md:58); a standalone library confines that churn to the bridge.
3. Precedent for decision 46: Cranelift built its own e-graph at about 7-8 percent compile-time cost after an egg-based prototype began over 2x (egraphs.md:74-75).
4. The counter-precedent: Luau shipped a header-only C++ e-graph and removed it two tags later (egraphs.md:95-97); in-house means ACT owns its maintenance, so the core stays small.

What decision 46's "only what compiler rewriting needs" keeps:

- Hash-consed e-nodes in an arena: op id, interned typed payload, child class ids; payloads are part of the key and matchable, unlike today's variant-only `matches` (egraphs.md:143).
- Union-find with path compression and union by size; deferred rebuild once per iteration, an algorithm from the egg paper, not its code (egraphs.md:10-11).
- One typed class-data record (shape, dtype, constant value) with `make` and `merge`, where a disagreement is a coded fault and the changed-flag is correct; today `merge` panics and always reports `b` unchanged (ir/egraph.rs:404-415, egraphs.md:148).
- Preconditions as filters over bound class data, run as soon as their variables bind, as egglog places conditions in the query (egraphs.md:112-113); not egg's post-match guard.
- A matcher that returns the matched e-node ids with the substitution, which deletes the applier's second backtracking match (isel/rewrites/applier.rs:106-113, egraphs.md:154).
- Storage as per-op tables, so a generic-join planner can be added later without re-layout; backtracking over a per-op index first (round1-fable.md:54, round1-sonnet.md:120).
- An applied-substitution set per rule, so an old match is not re-applied each iteration.
- An iteration hook with a read-only view, so extraction neither mutates nor clones the e-graph (pipeline.rs:235).
- Insertion-ordered containers only, so runs are reproducible (round1-fable.md:58).

What it drops: explanations and proof forests (unused, egraphs.md:172), a rule language or interpreter at run time (rules are generated C++), Datalog functions, and parallel execution.

5. Extraction stays outside the library, against Sonnet's library extractors (round1-sonnet.md:124).
   Candidates are ranked by a C cost model on emitted assembly (egraphs.md:167), with state keyed by class and buffer (egraphs.md:170), so no local cost function exists for a generic extractor; add one at the second use.
6. Migration gate, revised: with B29's `SimpleScheduler` mode and no node-limit stop, both engines must give equal canonical e-graph dumps per iteration and identical chosen assembly and `--log` files.
   Where a node limit still stops a kernel, compare cost and oracle result instead, as Sonnet proposes (round1-sonnet.md:128).
   Goldens are re-recorded at most once, in their own commit, when B29 switches the schedule.

Final position: option B, standalone in-house C++ library, MLIR only at the bridge, backtracking matcher first on a relational-ready layout.
Confidence: high on B over A; medium-high on the keep and drop lists; medium on when generic join is needed.
What would change my mind: B34 showing matching over a third of compile time after B32, which brings generic join forward.
What still divides: extraction in or out of the library; whether the gate needs the schedule switch (me) or tolerates scheduler differences by comparing cost and oracle only (Sonnet).
