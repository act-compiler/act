# Panel 12, round 1, Opus (lens: e-graph and compiler architect)

New items below carry local labels N1 to N11; each owning track assigns its own ID.

### Q12.4

Position: the platform items join as waves 8 to 10 after M5, owned by the existing tracks, with six earlier items asked to keep one property each now.

| Label | Item | Track | Depends on | Wave |
| ----- | ---- | ----- | ---------- | ---- |
| N1 | One pinned LLVM, MLIR, and StableHLO toolchain image for backend and oracle | T | U9 | 8 |
| N2 | Serialized checked model with `CONTRACT_VERSION`, and a C++ loader | U | U2, U3, U10 | 8 |
| N3 | C++ driver: MLIR StableHLO reader, egg behind a C ABI, allocator in process | B | N1, B2, B3, B29, B42 | 8 |
| N4 | fsim-lib migration, native-path steps 1 and 2 | O | O29, N1 | 8 |
| N5 | TAIDL ISA dialect generated from the checked model | B | N2, B13, B39 | 9 |
| N6 | C++ e-graph library core, differential against egg | B | B31, B7, B6, B35, N3 | 9 |
| N7 | torch-mlir reader into the same StableHLO path | B | N3, B42 | 9 |
| N8 | fsim-lib migration, native-path steps 3 and 4, on N5's semantics | O | N4, N5 | 9 |
| N9 | Engine swap: N6 replaces egg; Rust crate deleted | B | N6, N5, B10 | 10 |
| N10 | torch.compile backend package | A | N7, B46, N9 | 10 |
| N11 | Platform gate: B3 on the C++ pipeline, oracle on fsim | T | N9, N8 | 10 |

1. Wave 7 already holds the post-M5 follow-ons and U9, one copy of docker and CI scripts (`plans/roadmap/wave-4-7.md:115-122`), so a shared toolchain image (N1) builds on U9 and starts wave 8.
2. fsim-lib already links MLIR and StableHLO static libraries pinned at StableHLO 1.20.0 (`plans/oracle/fsim/native-path.md:25`), so the backend's MLIR pipeline and the oracle share one pin, not two.
3. O29 is a study in wave 3 that "reorders nothing on the current path" (`plans/oracle/plan/architecture.md:107-117`); its output becomes N4 and N8, and risk K9 already schedules the migration after M5 (`plans/roadmap/risks.md:26`).
4. U2 states that "a serialized form is added at the second non-Python consumer" (`plans/shared/plan/U2.md:15`); the C++ backend is that consumer, so N2 lands with N3, not earlier, and the v3 contract is not reopened.
5. N2 should be JSON parsed by nlohmann/json, which the backend build already needs (`plans/backend/plan/architecture.md:42`), so the export adds no dependency.
6. What earlier items must do now, one property each:
   - U2: the `Model` stays plain data with no Python callables, so N2 is a dump, not a redesign (`plans/shared/plan/U2.md:3-4`).
   - B29: the deterministic mode bounds by iteration and candidate counts (`plans/backend/plan/search.md:15`) and records its rule schedule, so a second engine can reproduce it; egg's `BackoffScheduler` (`pipeline.rs:491-495`, cited at `search.md:5`) is engine-specific.
   - B31: the typed payload is the one record the B2 C ABI carries (`plans/backend/plan/architecture.md:25-26`), so it becomes N6's e-node payload unchanged.
   - B6: the spike's data form of rules (`plans/backend/plan/isel.md:53`) is preferred, because data rules port to N6 and generated Rust does not.
   - B35 and B8: ties break by declaration index and tests assert terms, not ids (`isel.md:66,73`), so N9's gate compares engines whose class ids differ.
   - B43: the Q2 per-format readers (`plans/backend/plan/failures.md:69`) wait for N3 and N7, so StableHLO and torch-mlir are not parsed twice; the Q1 detection stays.
7. Owner decision 33 forbids commit pins in public; torch-mlir and StableHLO track LLVM by commit, so N1 pins released StableHLO and torch-mlir versions and inherits their LLVM pin (https://github.com/openxla/stablehlo).

Confidence: medium-high on ownership and dependencies; medium on wave sizes, since decision 11 sizes capacity for two quarters ending near M5.
What would change my mind: O29 finding that fsim needs an MLIR pin incompatible with any released torch-mlir, which would split N1 in two.

### Q12.1

Position: option A, staged inside one running pipeline: C++ driver around today's engine first, then the ISA dialect, then selection, then the engine swap; the front end stays Python and exports the checked model.

1. Decision 43 says "starting with the backend, so it reads StableHLO, MLIR, and torch-mlir inputs natively"; today the reader is HLO text only (`act-backend/act_backend/generic/src/isel/initializer/lib.rs:36`, per `plans/backend/plan/failures.md:58`).
2. The allocator is already C++ behind FFI (`act-backend/act_backend/generic/src/malloc/cpp.rs:15-24`, `generic/build.rs:10-37`), so the port's first step is a C++ `main` that owns the allocator in process and calls the search through a C ABI, not a rewrite of the allocator.
3. B2 already makes that C ABI with cbindgen from a `#[repr(C)]` schema (`plans/backend/plan/architecture.md:25-26`), and decision 45 itself routes the move "through a Rust FFI".
4. The input stage uses MLIR's own parser and the StableHLO dialect (https://openxla.org/stablehlo/spec); verifier failures become decision 40's reader errors, which ends the hand-written-parser panics of D22 (`plans/backend/plan/bugs.md:30`).
5. The ISA dialect: one op per instruction, attributes typed by the width layer, generated from the checked model, each op carrying its semantics as a StableHLO region.
   That one region is both the selection rule's left-hand side (today `ir2isa_rewrites.txt`, `generic/src/isel/rewrites/lib.rs:6-19`) and the body the oracle executes, which fsim already runs as StableHLO (`plans/oracle/fsim/native-path.md:32-34`).
6. Option B (front end to C++ first) delays decision 43's stated goal and reopens the v3 authoring surface (decisions 41, 42); option C has no gate, while B3 and B29 give a staged port an exact one (`plans/backend/plan/search.md:23`).
7. IREE shows the shape: an MLIR compiler with target plugins over StableHLO input (https://iree.dev); ACT needs its input conversion and dialect conventions, not its runtime.

Confidence: high on order (backend first, front end stays Python); medium on per-ISA ODS generation versus one generic `taidl.instr` op keyed by the model, which N5 should decide on build time.
What would change my mind: a measured per-ISA MLIR build time that makes generated ODS impractical, which favors the generic op.

### Q12.2

Position: option A now; PrivateUse1 is out of scope until a TAIDL accelerator has a real device runtime.

1. A torch.compile backend has the contract `(gm: torch.fx.GraphModule, example_inputs) -> Callable` and, through AOTAutograd, receives canonical ATen graphs, so one backend covers every TAIDL accelerator (https://pytorch.org/docs/stable/torch.compiler_custom_backends).
2. It is registered by the `torch_dynamo_backends` entry point (same URL), so no PyTorch fork is needed.
3. torch-mlir lowers FX to its torch dialect and then to StableHLO (https://github.com/llvm/torch-mlir), which feeds N3's reader; decision 43 names torch-mlir as a native input.
4. The returned callable runs the selected program on the oracle, so every torch test is an oracle-differential test against eager PyTorch, which satisfies the testability criterion.
5. Decision 40: the ACT compiler stays strict, stopping on an invalid or out-of-specification graph; the torch package partitions by the B46 exclusion report and the coverage table, sends only supported subgraphs, and leaves the rest to eager, with a strict mode that errors instead.
6. PrivateUse1 needs one eager kernel per ATen op per accelerator and a device allocator, which is per-target work and fails the portability criterion (https://pytorch.org/tutorials/advanced/privateuseone.html).
7. torch-mlir is a new dependency; per engineering rules and decision 28 it is asked when N7 lands; torch.export with `exported_program_to_stablehlo` from torch_xla is the fallback (https://docs.pytorch.org/xla/master/features/stablehlo.html).

Confidence: high on A over B; medium on torch-mlir over torch_xla, which is a dependency choice for the owner.
What would change my mind: a TAIDL target shipping hardware with a runtime, which makes a PrivateUse1 device worth a later item.

### Q12.3

Position: option B, a standalone C++ library with its own compact hash-consed e-nodes, relational e-matching with semi-naive deltas, a class-data analysis, and an MLIR bridge at import and export only; migration runs egg and the library side by side under B29 and B3.

1. Today's e-graph is small and shallow: `NODE_LIMIT` 10000 and `ITERATION_LIMIT` 30 (`generic/src/lib.rs:20-22`), with left-hand sides of depth 1 to 3 (`ir2ir_rewrites.txt:1-10`).
   An MLIR `Operation` per e-node costs operands, use lists, and attribute dictionaries; a compact node (op id, interned payload id, child class ids) keeps 10000 nodes cache-resident.
2. Option A exists only as research: the eqsat dialect is an xDSL prototype that rebuilds congruence by whole-graph CSE, which its authors call less efficient than incremental rebuilding, and it gives no analysis mechanism or extraction (https://arxiv.org/abs/2505.09363, sections 4 and 6).
   Decision 45 requires native metadata with preconditions, which eqsat lacks.
3. Option C: DialEgg bridges MLIR to egglog, a Rust engine, by translation (https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog); I found no maintained C++ e-graph library with typed analyses, so C fails decision 45's "C++" (low confidence on the survey).
4. E-matching: relational e-matching reduces a pattern to a conjunctive query over per-operator tables and answers it by generic join, worst-case optimal and much faster than egg's backtracking (https://arxiv.org/abs/2108.02290).
   Semi-naive evaluation, as in egglog, matches only against rows added since the last iteration (https://arxiv.org/abs/2304.04332).
   This targets today's waste directly: rules 2, 3, 4, and 7 have a bare-variable left-hand side that fires on every class every iteration (`plans/backend/plan/search.md:4`).
   Multi-patterns come free, which serves "complex rewrites with little code".
5. Payloads are part of the e-node key and matchable; today `Language::matches` compares only the discriminant, so payloads never match (`generic/src/ir/egraph.rs:162-165`) and conditions re-parse strings (`egraph.rs:63-94`).
6. Metadata: one class-data record (shape, element type, constant value) with `make` and a `merge` that is a coded fault on disagreement, as B7 plans (`plans/backend/plan/isel.md:30`); today `make` sets only `is_const` (`egraph.rs:396-401`).
   Preconditions are predicates over bound class data and payloads, evaluated as join filters as soon as their variables bind, replacing the three number-keyed functions per rule (`generic/src/isel/rewrites/applier.rs:15-58`).
7. No proof machinery: no explanation forest, so union-find is plain path compression and union by size.
8. Extraction is outside the library: the k-best (class, buffer) extractor of B10 (`plans/backend/plan/extractor.md:31`) reads the library through a read-only class view, so selection semantics do not move with the engine.
9. Migration, in order:
   - N3 drives egg through the B2 C ABI;
   - N6 implements the same ABI; both engines load the same data rules (B6 data form);
   - in B29's deterministic mode, both must give equal extracted term sets per iteration bound and identical chosen assembly and `--log` files (`plans/backend/plan/architecture.md:36-45`);
   - goldens are re-recorded at most once, in a commit of their own, only if a schedule difference changes a tie (`plans/backend/plan/search.md:17`);
   - N9 deletes egg once the gate holds on every example and the oracle run is equal.
10. MLIR stays at the edges: import from the StableHLO module into tables, export the chosen program as ISA-dialect ops.

Confidence: high on B over A; medium-high on relational matching from the start rather than a backtracking first version, given the measured node and rule counts are small.
What would change my mind: B34's budget measurement (`plans/backend/plan/search.md:38`) showing matching under 5 percent of compile time, which would favor a simpler backtracking matcher first and relational matching at the second need.
