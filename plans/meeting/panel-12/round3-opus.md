# Panel 12, round 3, Opus (lens: e-graph and compiler architect)

Read: the six round files, the three briefs (egraphs.md, mlir-ingest.md, torch-backends.md), and owner decisions 23, 28, 33, 39 to 46.
Backend code paths are relative to act-backend/act_backend/generic/src/; brief paths to panel-12/research/.

### Q12.1: port order

Where the panel agrees:
- Option A in strangler form: backend first, front end stays Python, every step under B3 and B29 (all three round-2 files).
- Step one is a C++ importer that reads StableHLO portable artifacts and prints the kernel text today's reader accepts; Fable and I conceded it (round2-fable.md Q12.1 point 1, round2-opus.md Q12.1 point 1).
- Portable artifacts are the supported input; text is a test path inside one pinned version (mlir-ingest.md 2.2).
- The middle step is decision 45 read literally: the Rust driver calls the in-house C++ library through FFI; the last step is one C++ driver and the Rust crate deleted.
- MLIR stays at the edges (importer, dialect, bridge), since "LLVM does not have a stable API" (mlir-ingest.md 1.7); the e-graph library includes no MLIR header.
- ACT is not an IREE plugin (dynamic plugins need exact IREE and LLVM revisions, mlir-ingest.md 4.2), and MLIR's destructive greedy driver never does selection.

What still divides it:
- Per-ISA ODS against one generic instruction op.
  - For per-ISA ODS: `irdl.c_pred` cannot register at run time, so checked guards need generated C++ (mlir-ingest.md 1.2, round1-fable.md Q12.1 point 5).
  - For the generic op: a per-ISA dialect puts MLIR headers and `mlir-tblgen` into every architect's build, against a 10-minute MLIR build and 147 MB of libraries (mlir-ingest.md 5).
- The allocator as an MLIR pass (Sonnet) against a C++ library behind the B2 flat struct (Fable, me).
  - For the pass: one IR end to end, IREE's layering (mlir-ingest.md 4).
  - Against: placement is chosen inside extraction keyed by (class, buffer) and ranked by a C cost model on assembly (egraphs.md 8), so the pass would build MLIR ops per candidate for no consumer.

Final position: option A in four gated steps: text-bridge importer; dialect spike then generator; library behind FFI with egg deleted; one C++ driver with the Rust crate deleted.
On the dialect form I move toward the generic op as default, because the e-graph and rules need no MLIR op classes and it keeps MLIR out of the per-ISA build; a spike measures per-ISA build time and decides.
The allocator stays a C++ library; an MLIR pass waits for a second consumer.

### Q12.2: PyTorch backend

Where the panel agrees:
- Compile path first, no PyTorch fork; PrivateUse1, an oracle-backed eager device, and a PJRT plugin are later rows with no wave.
- Strict by default under decision 40, as TorchNeuron fails compilation on an op it cannot lower (torch-backends.md, existing accelerators).
- Partition to eager is an opt-in that prints every undelegated node, in the B46 exclusion-report form.
- `torch_xla` is dropped: PyTorch/XLA is archived on 15 Nov 2026 (torch-backends.md, PyTorch/XLA).
- `dynamic=False`, symbolic dimensions rejected with a coded error, kernels cached by artifact hash.
- `torch.export` with core ATen decompositions is the public door; `torch.compile` wraps it through the private `torch._dynamo` hook, retested each minor.
- Testability: ACT assembly on the generated oracle against eager PyTorch within per-dtype tolerance.

What still divides it:
- The producer: torch-mlir across a file boundary (Fable, Sonnet) against an ACT-owned FX walker (me).
  - For torch-mlir: decision 44 says "built on that infrastructure" and decision 43 names torch-mlir input; it reuses the importer, and Neuron vendors it.
  - Against: torch-mlir pins `torch==2.15.0.dev20260917`, a nightly (torch-backends.md:89), while decision 33 forbids pre-release pins in public.
- Partition scheduled one wave after strict (Sonnet) or a later row (Fable); no evidence beyond the 18-operation count separates them.

Final position: Sonnet's pairing spike decides the producer: a released torch-mlir on a released torch; if that fails, the ACT-owned walker over the backend's operations.
I put the walker before vendoring, since vendoring makes ACT own an LLVM-incubator build; this order is the one dependency question for the owner when the item lands (decision 28).
Partition is scheduled one wave after strict, because with 18 supported operations strict alone rejects most real models.

### Q12.3: C++ e-graph library

Where the panel agrees:
- Option B under decision 46: standalone, in-house, no MLIR header; MLIR only at import and at the chosen program.
- Keep: arena e-nodes with typed payloads in the hash-cons key, since `matches` compares only the variant today (egraphs.md 8, ir/egraph.rs:162-165).
- Keep: union-find with path compression and union by size; deferred rebuild once per iteration, 88x on congruence and 21x on saturation (egraphs.md 1).
- Keep: one class-data record (shape, dtype, constant value) with `make` and `merge`, a disagreement as the coded fault of B26.
- Keep: preconditions evaluated as soon as their variables bind, egglog's in-query placement (egraphs.md 6) without its planner.
- Keep: a matcher that yields substitutions with matched e-node ids, which deletes the applier's second match (isel/rewrites/applier.rs:106-113).
- Keep: rules generated as C++ from B6's language-neutral records; no run-time rule language.
- Keep: a read-only class view for extraction (B9), insertion-ordered containers, the three limits with stop reasons.
- Drop: explanations and proof forests (unused), Datalog functions, threading, and the `BackoffScheduler`.
- Backtracking matcher first over per-operator storage, so generic join is a local addition; I withdrew relational-first, since its gains are measured at 1e5 to 3e6 e-nodes and dense patterns gain little (egraphs.md 3).
- Extraction stays in the backend: repeat-free enumeration and the (class, buffer) key are selection policy; Sonnet withdrew library k-best (round2-sonnet.md revision 9).
- B32's guards land before the swap, since rules 2, 3, 4, and 7 end every run at `NODE_LIMIT` today (plans/backend/plan/search.md:4).

What still divides it:
- The swap gate: identical logs under a no-ban schedule (Fable, me) against canonical term sets plus cost and oracle (Sonnet).
  - For identical logs: B29's verification is identical assembly and `--log` files (search.md, B29 Verification), and egg's `SimpleScheduler` runs every rule every iteration.
  - For Sonnet: an independent engine cannot reproduce egg's cut, which depends on rule order and bans (egraphs.md 1), and matching egg's enumeration order would copy egg (decision 46).
- `modify` kept for once-per-class identities (Fable) or dropped (Sonnet).
  - For: `(?x) => (copy ?x)` fires once per class at creation (ir2ir_rewrites.txt:8).
  - Against: B32's guards plus an applied-match memo give the same once-per-class firing without adding e-nodes inside rebuild.

Final position: the two gates compose, and I adopt the composition.
Search-then-apply per iteration makes the post-iteration e-graph unique up to class ids, so under a no-ban schedule and no node-limit stop both engines give equal canonical dumps per iteration.
Extraction then orders choices by canonical term key, not e-node id (Sonnet's point), which makes chosen assembly and `--log` files identical.
Where a node-limit stop remains, the gate is cost no worse and an equal oracle result.
I drop `modify`; generic join comes forward only if B34 shows matching above a third of compile time after B32.
One gate fail point is fixed before the swap: today's `merge` returns `b_merged = false` against egg's contract (ir/egraph.rs:404-415).

### Q12.4: joining the roadmap

Where the panel agrees:
- No new track and no platform item before M5; waves 8 to 10 follow wave 7, since wave 7 holds only items with no dependent (wave-4-7.md:118).
- The library core starts in wave 8, the swap with egg deleted in wave 9, the C++ driver with the Rust crate deleted in wave 10.
- The torch path sits in wave 9; fsim's four native-path steps run in waves 8 and 9; the interpreter step stays out (decision 28).
- The public pin is one released StableHLO version, fsim-lib's 1.20.0 (plans/oracle/fsim/native-path.md:25), with LLVM inherited; A16 records it.
- Earlier items change by text only, not by new work (constraints below).

What still divides it:
- A serialized `Model`: Sonnet adds one for a generic op's verifier, as U2's "second non-Python consumer" (plans/shared/plan/U2.md:15); Fable and I find no C++ reader when generators emit C++.
- The pin set's track: shared beside U9 (Fable), act (Sonnet), tests (my round 1).

Final position: the serialized `Model` is conditional on the dialect spike picking the generic op, so it is one row that lands only on that outcome.
The pin set is shared-track, because U9 owns the docker and CI scripts (wave-4-7.md:120) and both backend and oracle read it.

## Proposed roadmap items

IDs were free outside plans/meeting/ in a grep, and the author re-checks; sizes use S = 1, M = 3, L = 6 (waves.md, Rules).

| ID | Title | Track | Depends on | Size | Wave |
| -- | ----- | ----- | ---------- | ---- | ---- |
| U15 | Toolchain pin set: one released StableHLO, inherited LLVM, build image, CI cache | U | U9, A16 | M | 8 |
| B47 | C++ importer: StableHLO portable artifacts to today's kernel text | B | B42, B43, U15 | M | 8 |
| T39 | StableHLO corpus exported once from jax, differential against the HLO path | T | B47, T13 | S | 8 |
| B48 | ISA dialect spike: per-ISA ODS against one generic op, per-ISA build measured | B | B13, U15 | S | 8 |
| U16 | Versioned `Model` export, only if B48 picks the generic op | U | B48, U2, U3 | S | 8 |
| B50 | E-graph core: arena, typed hash-cons, union-find, rebuild, class data, read-only view, property tests | B | B31, B7, B9, B26 | L | 8 |
| O34 | fsim step 1: C++ emitter per instruction from O3's generic emitter | O | O29, O3, U15 | L | 8 |
| O35 | fsim step 2: `Pipeline::submit`, `end_trace`, `sync` replace jax's run | O | O34 | M | 8 |
| B51 | Matcher: backtracking per operator, early preconditions, matched ids, applied-match memo, C++ rule emitter | B | B50, B6, B8 | L | 9 |
| B52 | Dual-engine harness and swap behind the B2 C ABI; egg deleted | B | B51, B2, B3, B10, B32, B34, B35 | M | 9 |
| B49 | ISA dialect generator per B48, with each instruction's StableHLO semantics region | B | B48, U10 | M | 9 |
| B56 | Torch adapter: `ExportedProgram` to portable artifact; pairing spike picks the producer; strict | B | B47, B42, B46 | M | 9 |
| A29 | `torch.export` and `torch.compile` entry points, artifact-hash cache, one torch minor per release | A | B56, A23 | M | 9 |
| T40 | Torch tests: eager PyTorch against ACT on the oracle per operation and dtype | T | A29, T29 | M | 9 |
| O36 | fsim step 3: debug print out of `custom_call`; nanobind leaves | O | O35 | S | 9 |
| O37 | fsim step 4: PJRT as the one external, pinned to a released plugin build | O | O36, U15 | S | 9 |
| B53 | Typed import: importer builds library terms through the bridge; text bridge deleted | B | B52, B49, B47 | M | 10 |
| B55 | XLA HLO text reader in C++, holding every message of tests/test_reader.py | B | B52 | M | 10 |
| B54 | One C++ driver: reader, engine, extraction, allocator, cost; Rust crate deleted | B | B53, B55 | L | 10 |
| A30 | Partition opt-in: supported subgraphs to ACT, undelegated nodes printed | A | A29, B46 | M | 10 |
| T41 | Platform gate: B3 on the C++ driver, oracle on fsim, every golden and norm row | T | B54, O37 | M | 10 |

Weights: wave 8 is 24 (23 without U16), wave 9 is 23, wave 10 is 18; later rows, no wave and no dependent: PrivateUse1 eager device over the oracle; ACT PJRT plugin as the JAX door; generic join in B51 (trigger: B34 shows matching above a third of compile time after B32); allocator as an MLIR pass.

Text changes to earlier items, so these land cleanly:
- U2 and U3: every `Model` field stays plain data with no Python-only type.
- B31 and B2: the typed payload is plain integers and a dtype enum; it becomes B50's e-node payload and hash-cons key unchanged.
- B7: class data is (shape, dtype, constant value), and `merge` reports `b_merged` per egg's contract.
- B6: a rule is a language-neutral record (name, pattern, guard tree over bound names, right-hand side) with no engine type in any signature; a Rust emitter now.
- B10: fast extraction chooses in canonical term order, not e-node id order.
- B32: once its guards end node-limit stops, both modes run egg's `SimpleScheduler`, with goldens re-recorded once in its commit.
- B43: narrows to detection, including StableHLO portable artifacts; its StableHLO and torch-mlir readers move to B47 and B56, its torch.fx reader to B56's walker branch.
- O3: its generic emitter is the one source of per-instruction StableHLO, read by O34 and B49.
- O29 and A16: the study plans O34 to O37 as its four steps on U15's pin, and the release checklist records that pin.

## Escalation to the owner

None; B56's producer is a decision 28 dependency question asked when B56 lands.
