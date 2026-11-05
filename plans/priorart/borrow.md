# What to borrow, ranked across systems

Ranked by expected value for the owner's goal: an unambiguous generic spec and simpler generators.
Each row names the plan item that carries it.

| rank | borrow | from | goes to | plan item |
| --- | --- | --- | --- | --- |
| 1 | Numbered per-operation constraints with Inputs and Outputs tables in S8, generating both generators' checks | StableHLO | spec, oracle, backend | P1 |
| 2 | Named typed rewrite rules with declared operand kinds and explicit priorities, replacing numbered strings and parallel match tables | ISLE, TableGen | backend | P3 |
| 3 | Phased saturation: expansion, mapping, cleanup, each with its own limits and stop reason | Isaria, 3LA, egglog | backend | P4 |
| 4 | Three-word taxonomy, rejected, fault, defined value, applied to every S1.3 item | StableHLO, Sail, NKI | spec | P2 |
| 5 | Skipped-pattern and stop-reason reports at generation and compile time | GlobalISel, egg Runner | backend, diagnostics | P5 |
| 6 | Examples as tests: one executable example per S8 row and per S9 row, shared by oracle and backend | StableHLO, Sail | tests | P7 |
| 7 | Reusable extractors: shape, dtype, attribute and constant checks generated once and composed per rule | ISLE, LLVM PatFrag and ComplexPattern | backend | P10 |
| 8 | Per-e-node cost in extraction instead of enumerate-then-score; after v3 | egg CostFunction, Diospyros, Hydride | backend | P6 |
| 9 | One normative `execute` function in the spec, cited by step number from both generators | Sail, ILA | spec, formal | P8 |
| 10 | Alignment as a buffer property; typed state kinds for descriptors, barriers and async groups wait for after v3 | NVGPU, ArmSME, ILA | spec, backend | P11 |
| 11 | Generated per-instruction documentation page with engine, constraints, faults and defined values | NKI | act umbrella | P9 |
| 12 | Rule provenance: every hand-written expansion rule named and tested against the oracle | Isaria | backend, tests | P12 |
| 13 | Alignment as a declared buffer property, not allocator-side integer checks; stride is a region property (C2, L19) | TVM tensorize, Exo memories | spec, backend | P11 |

## What to avoid, ranked

| rank | avoid | seen in | why it matters here |
| --- | --- | --- | --- |
| 1 | Two descriptions of one instruction that can drift, a semantics and a separate lowering | TVM tensorize, Exo `@instr` template | TAIDL's single semantics block is its main advantage; never add a hand-written per-instruction emitter |
| 2 | Unspecified overlap resolution between rules, specificity heuristics or e-node order | ISLE, Halide rule tables, act-backend fast extractor | selection results become unstable across refactors; declare priorities or reject overlap |
| 3 | Solver or synthesis work at per-kernel compile time | Hydride, Isaria | the owner's sub-second compile is a published claim; keep solvers in the generator |
| 4 | A general imperative language inside the ISA specification | Sail, ILAng | widens the generator surface without a tensor-level gain |
| 5 | Prose constraints with no checkable form | NKI pages, MLIR C++ verifiers | the spec must generate the check, not describe it |
| 6 | Time limits as the only termination story for saturation | act-backend `TIME_LIMIT` 1 s | results differ by machine; the wall-clock bound stays the user default, tests use the deterministic mode, and a bound never yields a wrong answer (owner decision 27) |
| 7 | SIMT per-thread fragments for tensor-core ops | NVGPU `mma.sync` | out of scope (owner decision 15); model warpgroup-level operands with a unit shape |
