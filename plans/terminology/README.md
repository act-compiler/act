# TAIDL terminology

Reference terms for the TAIDL language spec and the four repos `taidl`, `act`, `act-backend`, and `act-oracle`.
Each table compares the term in use today with MLIR, StableHLO / XLA, PyTorch, and AWS Neuron NKI, then recommends one term.
The owner's papers take precedence where their term is precise; otherwise the most common term across the tools wins.
A row marked CONFLICT needs an owner decision, and its recommended term is the suggested default.
Tags such as [ACT] and [SHLO] are defined in `sources.md`.

| file | contents |
| --- | --- |
| [description-and-memory.md](description-and-memory.md) | The ISA description, data model, buffers, control state, element types, shapes |
| [instructions.md](instructions.md) | Instructions, their attributes and regions, and the semantics language |
| [operations.md](operations.md) | Operation families and their names in each tool |
| [compiler-and-tools.md](compiler-and-tools.md) | Compiler backend pieces, test oracle, kernels, errors, conformance |
| [conflicts.md](conflicts.md) | Owner decisions, each with a suggested default |
| [words-to-avoid.md](words-to-avoid.md) | Vague or off-glossary words seen in the repos, with replacements |
| [sources.md](sources.md) | Source tags and URLs |
