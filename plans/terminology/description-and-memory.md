# Description, memory, and state

Evidence per concept; the glossary term is in `glossary/terms.md`.

| Concept | Owner papers | MLIR | StableHLO/XLA | PyTorch | Neuron NKI |
| --- | --- | --- | --- | --- | --- |
| The accelerator description as a whole | "ISA description" 54x [ACT]; "TAIDL definition", "ISA definition" [TAIDL]; "ISA specifications" [SITE] | n/a | n/a | n/a | n/a |
| The object it builds | "tensor accelerator" [ACT] [TAIDL] | n/a | n/a | n/a | "Trainium device", "NeuronCore" [NKI] |
| All storage units together | "storage units (collectively termed data model)" [ACT] [TAIDL] | n/a | n/a | n/a | n/a |
| One named storage unit | "tensor buffer" 13x [TAIDL], 23x [ACT]; "scratchpad" for the hardware | memory space (memref) | n/a | n/a | buffer: SBUF "state buffer", PSUM "partial sum buffer" [NKI] |
| Built-in memory behind `d0` | "byte-addressable 1-D global memory ... denoted d0" [ACT]; HBM as an alias | `global` memory space (gpu) | n/a | n/a | HBM, `nl.shared_hbm` [NKI] |
| Dimensions a region indexes | "access dimensions (S0)" [ACT] | shape (memref) | dimension, axis [SHLO] | dim | partition dimension (dim 0, 128 partitions) and free dimensions [NKI] |
| Shape always read whole | "granularity of data access (S1)" [ACT]; "base element" [TAIDL] | no analogue | no analogue | no analogue | closest: free dimension of one partition [NKI] |
| Persistent integer state | "control registers" 31x, part of the data model [TAIDL] | n/a | n/a | n/a | n/a |
| Contents of all buffers at a point | "memory state" [ACT] | n/a | n/a | n/a | n/a |
| Element type | "data type", "dtype" [TAIDL] | element type [MLIR] | element type [SHLO]; `PrimitiveType`, `new_element_type` [XLA] | `torch.dtype` [PT] | dtype (`nl.float16`) [NKI] |
| Element type spellings | `i8`, `xi8` (TAIDL syntax) [TAIDL] | `bf16`, `f32`, `i8` | `bf16`, `f32`, `si8`, `ui8` [SHLO] | `bfloat16`, `float32`, `int8`, `uint8` [PT] | `nl.bfloat16`, `nl.float32` [NKI] |
| Shape | `E[S0, S1]` [ACT] | shape [MLIR] | shape, rank [SHLO] | shape | shape [NKI] |
| Layout and strides | "layout transformation" (as reshape and transpose) [ACT] | layout, strided layout, strides, offset (memref) [MLIR] | layout | `torch.strided`, stride, `memory_format` [PT] | access pattern [NKI] |

## Notes

- TAIDL has no layout or stride construct, so "layout" is not borrowed for row-major order.
- NKI's partition dimension is one access dimension, not all of them.
- The papers name the base element idea but not its shape.
