# Compiler backend, tools, and errors

Evidence per concept; the glossary term is in `glossary/terms.md`.

## Tools and artifacts

| Concept | Owner papers | Other tools |
| --- | --- | --- |
| Tool that reads an ISA specification | none | frontend |
| Tool generated from an ISA specification | "compiler backend generator" [ACT] | n/a |
| Generated compiler | "compiler backend" [ACT] | backend (XLA, MLIR) |
| Executable model of the ISA | "test oracles (also known as functional simulators)" [TAIDL] | `nki.simulate_kernel`, NKI CPU simulator [NKI] |
| Input unit of compilation | "tiled kernels (sub-graphs with typically 10-50 nodes)" 48x [ACT] | `gpu.func` with `kernel` attribute [MLIR]; `@nki.jit` kernel [NKI] |
| Unit run on the test oracle | "kernel" [TAIDL] | `@nki.jit` kernel [NKI] |
| Compiled output | "assembly code ... a stream of concrete instructions and a constant tensor" [ACT] | n/a |
| Source IR | "XLA-HLO IR, XLA's tensor IR" [ACT] | HLO [XLA] |

## Compiler backend pieces

| Concept | Owner papers | Other tools |
| --- | --- | --- |
| Phase 1 | "instruction selection" using equality saturation [ACT] | lowering |
| Program set | e-graph, e-class, e-node [ACT] | n/a |
| Rewrite | "IR-to-IR" and "IR-to-ISA rewrite rules" [ACT] | `RewritePattern` [MLIR] |
| Choosing a program | "extraction", "graph extractor" [ACT] | n/a |
| Partial instruction plan | "partially instantiated instruction (pii)", "pii graph" [ACT] | n/a |
| Phase 2 | "memory allocation using integer constraint programming" [ACT] | bufferization, allocation [MLIR] |
| Backend correctness | "sound and complete" [ACT] | n/a |

## Errors and conformance

| Concept | Other tools |
| --- | --- |
| Error for a description | violated constraint (Cn) [SHLO]; legality (dialect conversion) [MLIR] |
| Valid but not handled | legality (dialect conversion) [MLIR] |
| Meeting the language | n/a |
| Result not given | "the behavior is undefined" [SHLO] |
| Left to each tool | implementation-defined [SHLO] |

## Notes

- StableHLO uses "undefined" and "implementation-defined" for open cases; TAIDL has none (Decision 16).
- "rewrite" alone also means regenerating goldens in `act-backend/README.md`.
