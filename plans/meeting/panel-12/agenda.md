# Panel 12 agenda: long-term platform

Binding owner decisions: 39 to 45, 23 (fsim-lib as the later oracle direction), 28 (dependencies), 33 (releases).
These items join the one roadmap as later waves of the existing tracks, with real dependencies on earlier items; nothing is set apart as a separate future plan.
They reuse the v3 contract (checked model, generator checks) and do not reopen it.

### Q12.4: How do these items join the roadmap?

- Questions: which track owns each item; which earlier items each depends on; what earlier items must do now so these land cleanly (for example a stable export of the checked model, the deterministic gate B29, the typed handoff B31); how the oracle's fsim-lib direction (decision 23, item O29) joins the same C++ platform; which waves they fall in.

### Q12.1: How is the stack ported to C++ on MLIR infrastructure, and in what order?

- Context: today the backend is Python generators emitting Rust (egg) and a C++ allocator; the front end is Python; the oracle is Python on JAX, with fsim-lib (C++, PJRT) as its later direction.
- Options: A. backend first as an MLIR pipeline (a TAIDL ISA dialect generated from the checked model, StableHLO input, rewrite-based selection, allocation), front end stays Python and exports the checked model; B. front end to C++ first; C. one-shot rewrite; D. other.
- Evidence: plans/backend/, plans/shared/ (typed contract), plans/oracle/fsim/; MLIR, StableHLO, IREE, torch-mlir docs (WebSearch and WebFetch via ToolSearch).

### Q12.2: How is a portable PyTorch backend built?

- Options: A. torch.compile custom backend taking FX graphs through torch-mlir or torch.export to StableHLO, then the ACT pipeline; B. a PrivateUse1 device with eager kernels; C. both, with the compile path first; D. other.
- Criteria: portability across accelerators described by TAIDL, no fork of PyTorch, testability with the oracle, decision 40 (invalid input graphs stop; candidate failures do not).

### Q12.3: What is the C++ e-graph library, and how does the backend migrate to it?

- Context: decision 45 lists union-find, fast e-matching, complex rewrites with little code, native e-node metadata with preconditions, no proof machinery, tensor-oriented.
- Owner decision 46: the library is built in-house, independent of egg and egglog, keeping only what compiler rewriting needs; adopting an existing library is not an option.
- Options: A. an MLIR-native e-graph over operations with attributes as metadata; B. a standalone C++ library with its own term representation and an MLIR bridge; C. other.
- Questions within: e-matching algorithm (for example relational or generic join as in egglog), metadata analyses and preconditions, extraction, migration from egg with the deterministic gate (B29) and goldens.
