# Panel 12, round 2: Sonnet (MLIR and torch-mlir integration lens)

Read: the three briefs (cited as egraphs.md, mlir-ingest.md, torch-backends.md), decision 46, and the Opus and Fable round-1 files.

## Round-1 claims revised first

1. Decision 46 closes option C by decision, so my "upstream C++ eqsat would change my mind" trigger is void; an in-house option A stays open.
2. No maintained C++ reference exists for the in-house build (egraphs.md s5).
   - Luau's `EGraph.h` is at tag 0.701 and absent from 0.702 and master, with no e-matching in it; Z3's carries SMT justifications.
   - The build cost is therefore real: core L, matcher L, harness M.
3. B47 accepted "text or portable artifact" in round 1; the supported input is the portable artifact only.
   - StableHLO guarantees exclude prettyprinted text and raw bytecode (mlir-ingest.md s2.2); text is a development path inside the pinned version.
4. A33 pinned "one LLVM and StableHLO commit"; decision 33 forbids commit pins in public.
   - The pin is one released StableHLO version, which fsim already uses at 1.20.0 (plans/oracle/fsim/architecture.md:68).
   - The LLVM commit is inherited from that tag's `llvm_version.txt`, and public users get a prebuilt wheel, about 80 MB for comparable tools (mlir-ingest.md s5).
5. B48 generated per-ISA ODS in round 1; I move to one generic instruction op driven by the exported `Model`.
   - Dynamic compiler plugins are experimental and need the host's exact IREE and LLVM revisions (mlir-ingest.md s4.2), so a per-ISA dialect loaded into a prebuilt tool is the fragile route.
   - Static linking per ISA needs MLIR headers and `mlir-tblgen` on the architect's machine, and the briefs report a 10 minute MLIR build and 147 MB of installed libraries (mlir-ingest.md s5).
6. I claimed PrivateUse1 needs "a kernel per ATen operator"; the minimum is about eight factory, view, and copy ops plus `cpu_fallback` (torch-backends.md, PrivateUse1).
7. I proposed `torch_xla` as the cross-check producer; PyTorch/XLA is to be archived on 15 Nov 2026 (torch-backends.md, XLA), so it is dropped.
8. I proposed comparing saturated dumps "where the kernel saturates"; today that set is empty.
   - Rules 2, 3, 4, and 7 fire on every class and end at `NODE_LIMIT` (plans/backend/plan/search.md:4; ir2ir_rewrites.txt:3-5, 8), so B32 guards come first.
9. I gave the library a k-best enumerator; the brief shows no e-graph k-best method and a hard optimal problem (egraphs.md s7).

### Q12.4: How do these items join the roadmap?

Strongest opposing argument (Fable, Opus):
- The library belongs in wave 9, after the reader and the dialect.
- Its payload schema waits for the whole B31 to B35 chain and the dialect's op ids, and wave 8 is already full with the importer, dialect, pins, and fsim steps.
- The shared track also needs a serialized `Model` (U15 or N2) and one shared pin set (U16), which my table omitted.

Rebuttal and concessions:
1. Conceded: the serialized `Model`.
   - Revision 5 makes the verifier and bridge read the `Model` at run time, the "second non-Python consumer" of plans/shared/plan/U2.md:15, so my "no serialized form" claim is withdrawn; JSON through nlohmann/json adds no dependency (plans/backend/plan/architecture.md:42).
   - U15 sits in wave 8, not wave 7, because wave 7 holds items with no scheduled dependent (plans/roadmap/wave-4-7.md, Wave 7) and U15 has B48.
2. Held: the library core in wave 8.
   - Its inputs are B29 (wave-1.md:24), B31 and B7 (wave-2.md:21,23), and B9 (wave-3.md:12), all finished before wave 8.
   - The e-node op id comes from the exported `Model`, not from the dialect, so B50 does not wait for B48.
   - Under decision 46 it is the longest new code and gates deleting Rust, so starting it last puts it on wave 10's critical path.
3. Conceded: the matcher and the swap sit in wave 9, and B51 gains B32 as a dependency (revision 8).
4. A PJRT plugin for ACT stays a "later" row, as does PrivateUse1; the briefs give no ACT-specific evidence for either, and neither has a dependent.

Final position: waves 8 to 10 as in round 1, with these table edits.
- A33 pins a released StableHLO version; B47 reads portable artifacts; B48 is the generic op plus a data-driven verifier.
- U15 is added in wave 8; B51 depends on B32; B56 drops `torch_xla`; B9 keeps the class view free of `DetectedConst` mutation (extractor/utils.rs:44-60).

Confidence: 70 percent on the wave split, 60 percent on item boundaries.
Still divides the panel: library core in wave 8 (me) or wave 9 (Opus, Fable); whether the PJRT plugin is scheduled (Fable).

### Q12.1: How is the stack ported to C++ on MLIR infrastructure, and in what order?

Strongest opposing argument (Opus, Fable):
- A standalone importer that prints kernel text keeps a lossy text hop alive: `si32` to `s32`, regions to `to_apply` helpers, `dot_general` to `dot`.
- The typed handoff B2 already exists, so the C++ reader should hand over typed ops through the C ABI at once, and the printer is throwaway code feeding a Rust crate slated for deletion.

Rebuttal and concession:
1. Conceded in part: the text bridge is transitional, sized S, and is replaced by a typed handoff when the library lands (B54), at which point the importer builds library terms directly.
2. Held for step one, because a typed handoff into Rust means changing the 2,100-line Rust initializer the plan is deleting (plans/backend/plan/parser.md:4), while a printer changes no Rust and leaves B3 goldens identical.
3. The importer follows fsim's pass stage, static MLIR behind a C entry (plans/oracle/fsim/architecture.md:30), and runs as a subprocess, never inside a torch process that may hold torch-mlir's MLIR.
4. The MLIR C++ API has no stability promise; Triton targets one LLVM revision per release (mlir-ingest.md s1.7).
   - The MLIR-touching C++ is therefore kept small: the importer, the bridge, and the allocator pass; the e-graph library includes no MLIR header.
5. Agreed with Fable item 11: ACT is not an IREE plugin, for the revision-matching reason in revision 5; the front end stays Python.

Final position: option A in strangler form.
- Order: importer with text bridge, then Model export plus generic dialect, library core in parallel, typed handoff replaces the text bridge, matcher and swap, then allocator and selection as passes.

Confidence: 78 percent on A; 60 percent on the text bridge as step one.
Still divides the panel: text bridge or typed handoff first (Opus, Fable); generic op or per-ISA ODS (Fable keeps generated C++).

### Q12.2: How is a portable PyTorch backend built?

Strongest opposing argument (Opus, partly Fable):
- A torch.compile backend that rejects a whole graph on one unsupported operator is useless when the backend handles 18 operations.
- Graph breaks and eager fallback are the PyTorch norm, and Gaudi and Neuron eager fall back to CPU (torch-backends.md, existing accelerators).
- ExecuTorch partitions exactly this way, so the torch package should partition by the exclusion report and keep strict only as a mode.

Rebuttal and concession:
1. Decision 40 places strictness at the compiler's input, not at the user's model, so partitioning in the torch package is legal if the compiler only ever sees supported subgraphs.
2. Default stays strict.
   - TorchNeuron's compile path fails by default on an operator it cannot lower (torch-backends.md, existing accelerators).
3. Conceded: partition mode is scheduled as an explicit opt-in, one wave after strict mode, with the B46 exclusion report naming every undelegated node.
   - Each subgraph is its own search, so the cache is part of the item, keyed by shapes, dtypes, and ACT version.
4. Shapes: Dynamo's automatic dynamic mode produces symbolic dimensions after the first recompile (torch-backends.md, torch.compile), and StableHLO shapes are ranked (mlir-ingest.md s2.1).
   - B56 asks for `dynamic=False` as Neuron does and rejects symbolic dimensions with a coded error naming the dimension.
5. Producer: torch-mlir pins `torch==2.15.0.dev20260917`, a nightly, while 2.15 is dated 28 Oct 2026 (torch-backends.md, torch-mlir and releases).
   - Decision 33 forbids pre-release pins, so the B55 spike first tests that torch-mlir release 20261101 imports an `ExportedProgram` from released torch 2.15.0.
   - Fallbacks in order: a vendored torch-mlir, as AWS Neuron does (torch-backends.md, existing accelerators), then an in-house walker for the 18 operations.
6. Coupling: the documented AOTAutograd hook is a private `torch._dynamo` import and minors ship every two months, with 2.16 due 22 Dec 2026 (torch-backends.md, torch.compile).
   - The torch package supports the latest two minors and is released per minor, as Neuron's `torch-neuronx==2.14.*` is.
7. Eager device: the case against it is the runtime it needs and the lost whole-kernel view (Fable items 8, 10), not the operator count.

Final position: option C with the compile path first, strict by default, partition as an opt-in follow-up, eager and PJRT later.
Confidence: 70 percent on the structure; 45 percent that torch-mlir is the producer, until the pairing spike runs.
Still divides the panel: default strict versus default partition (Opus); whether the eager test device is scheduled (Fable).

### Q12.3: What is the C++ e-graph library, and how does the backend migrate to it?

Strongest opposing argument (Opus):
- Relational e-matching with semi-naive deltas should be built from the start.
- It cuts the waste of the bare-variable rules that fire on every class each iteration (search.md:4), gives multi-patterns for free, and has a worst-case bound (egraphs.md s3).

Rebuttal:
1. The measured speedups are at 1e5 to 3e6 e-nodes (egraphs.md s3), and ACT caps at 10,000 nodes (generic/src/lib.rs:20) with left-hand sides of at most two operators (ir2ir_rewrites.txt:1-10).
2. Dense patterns with large outputs gain little (egraphs.md s3), and a bare-variable rule is the densest pattern, so a join does not cut the waste Opus targets.
   - B32 guards stop the firing, and a per-rule applied-class memo stops the rematching.
3. A relational matcher means storing the graph as per-operator relations with index tries.
   - The prototype rebuilt its database per match and egglog avoided that by being the database (egraphs.md s3), and egglog 2.0 rewrote its backend as database-first and parallel (egraphs.md s2).
   - Doing that in-house rebuilds the Datalog core of systems that decision 46 says were built more for proofs than for compilers.
4. Compiled backtracking has production precedent in Z3's matching abstract machine and in egg's own VM (egraphs.md s1, s3).
5. Conceded: the node store is column-oriented per operator, which backtracking needs as its index anyway, so a join is a local addition behind the "yield substitutions" interface.
   - The trigger is a B34 profile where matching exceeds a stated share of compile time; I accept Opus's 5 percent as the threshold.

Option A in-house, restated and answered:
- Evidence against holds: per-merge MLIR uniquing (Opus), whole-graph CSE for congruence that its authors call "possibly less efficient", and no benchmarks (egraphs.md s4).
- New: an e-graph on the MLIR C++ API inherits a port at every LLVM bump (mlir-ingest.md s1.7).

What decision 46 keeps and drops:
- Keep: arena e-nodes with interned payloads, hash-consing, union-find with path compression and union by size, and deferred rebuild.
  - Deferred rebuild is the one egg idea with measured value, 88x on congruence closure and 21x on saturation (egraphs.md s1), and a published algorithm is not a dependency.
- Keep: class analysis `make` and `merge` with a coded fault, the three limits with stop reasons, and the read-only class view.
- Keep: guards as expression trees evaluated as soon as their variables bind, which gives egglog's in-query conditions without its planner (egraphs.md s6).
  - The same evaluator runs the dialect verifier of revision 5, so there is one guard implementation.
- Drop: explanations (unused, egraphs.md s8), the `modify`, `pre_union`, and `remake` hooks, egglog `:merge` functions and rule language, semi-naive tables, threading, and `BackoffScheduler`, whose job guards plus the applied-class memo take over.
- Cranelift's acyclic aegraph with no rebuild is rejected as a model: ACT's identity rules create cycles (ir2ir_rewrites.txt:8, `(?x) => (copy ?x)`), and its average class holds 1.13 nodes with no associativity (egraphs.md s4).

Migration gates, replacing round 1:
1. Library property tests (congruence, hash-cons, analysis conflicts) run without egg; egg is a test oracle in the harness only, not a design source.
2. After B32, kernels that saturate must give equal sorted canonical term sets per class; kernels that still stop at the node limit must give chosen cost no worse than egg's and an equal oracle result.
   - An independent engine cannot reproduce egg's cut, which depends on its rule order and ban schedule (egraphs.md s1).
3. Chosen assembly and `--log` files must be identical only after B29 orders candidates by canonical term text and B35 breaks ties by declaration index.
   - Fable's insertion-ordered containers would match egg only by copying egg's match order, which decision 46 rules out.

Final position: option B, standalone, no MLIR header, with the bridge in B54; backtracking matcher first.
Confidence: 75 percent on B over A; 65 percent on backtracking first.
Still divides the panel: relational matching from the start (Opus); whether the library owns k-best extraction (my round 1, now withdrawn); the gate for node-limit kernels (Fable wants identical logs).

## Escalation to the owner

None.
