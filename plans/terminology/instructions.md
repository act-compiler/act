# Instructions and the semantics language

Evidence per concept; the glossary term is in `glossary/terms.md`.

## Instructions

| Concept | Owner papers | MLIR | StableHLO/XLA | PyTorch | Neuron NKI |
| --- | --- | --- | --- | --- | --- |
| ISA instruction | "abstract instruction" and "concrete instruction" [ACT] | operation | n/a | n/a | `nki.isa` instruction, such as `nc_matmul` [NKI] |
| Instruction parameter | "computational (alpha)" and "addressing (beta)" attributes [ACT]; "calling attributes" [TAIDL] | attribute [MLIR] | attribute [SHLO] | argument | parameter: `dst`, `data`, `stationary`, `moving` [NKI] |
| Immediate | none | constant attribute | constant attribute [SHLO] | scalar argument | compile-time constant [NKI] |
| Data an instruction reads or writes | "data slice d[Is:Ie]" [ACT] | `tensor.extract_slice` [MLIR] | slice [SHLO] | view | tile [NKI] |
| Index and extent | start (inclusive), end (exclusive) [ACT] | offsets, sizes | `start_indices`, `slice_sizes` [SHLO] | start, end | index [NKI] |
| Instruction inputs and outputs | none | `ins`, `outs` (linalg) [MLIR] | operands, results [SHLO] | args | `src`, `dst` [NKI] |
| In-place output | none | in-place bufferization [MLIR] | n/a | in-place variants (outside Core ATen) [PT] | n/a |
| Validity condition | "validity constraint" [ACT]; "assert statement specifies the constraints" [TAIDL] | n/a | constraint (C1, C2) [SHLO] | n/a | n/a |
| State change | "assignment statements update the control registers" [TAIDL] | n/a | n/a | n/a | n/a |
| What an instruction computes | "instruction semantics" [TAIDL] [ACT] | region, payload (linalg) [MLIR] | HLO computation [XLA] | n/a | n/a |

## Semantics language

| Concept | MLIR | StableHLO/XLA | PyTorch | Neuron NKI |
| --- | --- | --- | --- | --- |
| One `%x = type[shape] op(...)` | operation | op [SHLO] | node (fx) [PT] | n/a |
| What it defines | SSA value, result | value, result [SHLO] | node | n/a |
| Operation name | op name | op name [SHLO] | Core ATen op name [PT] | n/a |
| Operation input | operand [MLIR] | operand, input [SHLO] | argument | n/a |
| `name=value` after operands | attribute [MLIR] | attribute [SHLO] | keyword argument | n/a |
| Integer text over references | indexing map [MLIR] | n/a | n/a | n/a |
| Value read in another shape | `tensor.reshape` [MLIR] | `reshape` [SHLO] | `view` [PT] | n/a |

## Notes

- "slice" is also a tensor operator in StableHLO, and "tile" means an on-chip value in NKI.
- An ISA-level instruction is not SSA, so "operand" belongs to the semantics language.
