# Instructions and the semantics language

"Current" names the term in `taidl/docs/glossary.md` first, then other repo usage.

## Instructions

| Concept | Current TAIDL/ACT term(s) | Owner papers | MLIR | StableHLO/XLA | PyTorch | Neuron NKI | Recommended term | One-line rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISA instruction | instruction (`add_instruction`) | "abstract instruction" and "concrete instruction" [ACT] | operation | n/a | n/a | `nki.isa` instruction, such as `nc_matmul` [NKI] | instruction; abstract vs concrete when it matters | Universal for an ISA; the qualifiers are [ACT]'s. |
| Instruction parameter | instruction attribute: computational (`@c.`), addressing (`@a.`); params `computation_attr`, `addressing_attr` | "computational (alpha)" and "addressing (beta)" attributes [ACT]; "calling attributes" [TAIDL] | attribute [MLIR] | attribute [SHLO] | argument | parameter: `dst`, `data`, `stationary`, `moving` [NKI] | computational attribute, addressing attribute | [ACT] is newer and splits the two kinds. |
| Immediate | banned | none | constant attribute | constant attribute [SHLO] | scalar argument | compile-time constant [NKI] | computational attribute | An immediate is a computational attribute in TAIDL. |
| Data an instruction reads or writes | region `[data model, starts, sizes]`; "tile" and "window" (`act-oracle/README.md:67,122`) | "data slice d[Is:Ie]" [ACT] | `tensor.extract_slice` [MLIR] | slice [SHLO] | view | tile [NKI] | region -- CONFLICT | "data slice" is the owner's, but `slice` is also an operation word; "tile" means an on-chip value in NKI. |
| Region index and extent | start, size | start (inclusive), end (exclusive) [ACT] | offsets, sizes | `start_indices`, `slice_sizes` [SHLO] | start, end | index [NKI] | start, size | The API takes sizes, matching `dynamic_slice`. |
| Instruction inputs and outputs | input region, output region (`set_inputs`, `set_outputs`); `%In<k>`, `%Out<k>` | none | `ins`, `outs` (linalg) [MLIR] | operands, results [SHLO] | args | `src`, `dst` [NKI] | input region, output region | ISA level is not SSA, so "operand" stays for lines. |
| In-place output | in-place mark (`set_inplace`) | none | in-place bufferization [MLIR] | n/a | in-place variants (outside Core ATen) [PT] | n/a | in-place mark | Keep. |
| Validity condition | constraint (`add_constraints`) | "validity constraint" [ACT]; "assert statement specifies the constraints" [TAIDL] | n/a | constraint (C1, C2) [SHLO] | n/a | n/a | constraint | Owner and [SHLO] agree. |
| State change | state update | "assignment statements update the control registers" [TAIDL] | n/a | n/a | n/a | n/a | state update | Keep. |
| What an instruction computes | semantics (`add_semantics`, `ENTRY` block) | "instruction semantics" [TAIDL] [ACT] | region, payload (linalg) [MLIR] | HLO computation [XLA] | n/a | n/a | semantics | Owner's term; "computation" names XLA's object. |

## Semantics language

| Concept | Current TAIDL/ACT term(s) | MLIR | StableHLO/XLA | PyTorch | Recommended term | One-line rationale |
| --- | --- | --- | --- | --- | --- | --- |
| One `%x = type[shape] op(...)` | line (glossary); grammar rule `instruction` (`taidl/taidl/antlr4/IDLV2.g4:8`) | operation | op [SHLO] | node (fx) [PT] | line -- CONFLICT | The grammar rule name collides with the ISA instruction. |
| What a line defines | value | SSA value, result | value, result [SHLO] | node | value | Keep. |
| Operation name | operation word | op name | op name [SHLO] | Core ATen op name [PT] | operation word | Keep; "opcode" stays banned. |
| Line input | operand | operand [MLIR] | operand, input [SHLO] | argument | operand | Universal. |
| `name=value` after operands | operation attribute | attribute [MLIR] | attribute [SHLO] | keyword argument | operation attribute | Qualified to differ from instruction attribute. |
| Integer text over references | expression | indexing map [MLIR] | n/a | n/a | expression | Keep. |
| `@a.x`, `@c.x`, `@s.x` | reference | n/a | n/a | n/a | reference | Keep. |
| Value read in another shape | presumed reshape | `tensor.reshape` [MLIR] | `reshape` [SHLO] | `view` [PT] | presumed reshape | Keep. |
