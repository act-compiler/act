# Panels

Each of the eleven panels has three debaters with different lenses and a moderator who does not debate.
Lenses rotate across model classes so no class always plays the same role.

| Panel | Scope | Sonnet lens | Opus lens | Fable lens | Moderator |
| ----- | ----- | ----------- | --------- | ---------- | --------- |
| 1 | Defined behavior and numerics: faults, accumulation, literals, integer arithmetic, aliasing, state read order, initial contents | Oracle implementer | Semantics and spec author | Hardware architect | Fable |
| 2 | Language surface: constructs to keep or remove, extensions, scope, spec presentation | New user writing an ISA specification | Minimalist language designer | Accelerator architect | Opus |
| 3 | Front end and generator contract: shape and type rules, object model, diagnostics, clause ids | Third-party generator author | Front-end maintainer | Compiler infrastructure architect | Sonnet |
| 4 | Backend architecture: parsers, e-graph, rewrites, extraction, allocation, state support | Incremental maintainer | E-graph and instruction selection expert | Outside compiler architect | Fable |
| 5 | Oracle architecture and fidelity: lowering, loops, runtime, reference checks, numerics | Oracle maintainer | Numerics and verification expert | JAX and XLA expert | Opus |
| 6 | Test suite: conformance corpus, expectations, differential tests, migration | CI and test engineer | Conformance purist | Industry conformance suites | Sonnet |
| 7 | Vocabulary and usability: glossary leftovers within owner decisions, act repo, tutorials, releases | New user | Docs and terminology owner | ML-compiler researcher | Fable |
| 8 | Roadmap: waves, milestones, dependencies, risks | Engineer executing the plan | Program manager | Risk red team | Opus |
| 9 | Overlap and hazard declarations: relations between inputs and outputs, markers | ISA author | Semantics and spec author | Shipped ISAs that alias operands | not recorded |
| 10 | Attribute bits and binary encodings: width layer, encoding layer, derived attributes, generator metadata | ISA author who needs an assembler | Language designer and semantics author | Compiler and assembler architect | not recorded |
| 11 | Authoring surface and the check function | Hardware architect writing an ISA specification | Language designer and front-end maintainer | DSL and compiler-infrastructure architect | not recorded |

Two joint decisions rule where panels split: `joint-1-5` for panels 1 and 5, and `joint-unwritten` for reads of unwritten on-chip storage.
