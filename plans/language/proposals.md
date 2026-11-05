# Proposals: language extensions

Ranked by coverage gained per cost.
Each row names its plan item; no surface change is made yet.

| rank | proposal | motivating accelerators | minimal syntax sketch | cost |
| --- | --- | --- | --- | --- |
| 1 | Low-precision and block-scaled types, `scaled_dot` (L17) | NVIDIA Blackwell, AMD XDNA2, Trainium2, Gaudi3, MTIA | `element_type=f4e2m1fn`; `ops.scaled_dot(a, a_scale, b, b_scale, lhs_contracting_dims=[1], rhs_contracting_dims=[0])` | M: `taidl.types`, `TYPES`, B and O type tables |
| 2 | Rounding and saturation on `convert` (L21) | Trainium, Gaudi, Ascend, int8 NPUs | `ops.convert(x, s8, rounding='TOWARD_ZERO', saturate=True)` | S: two attributes |
| 3 | Multi-core and async DMA idioms (L24) | Tenstorrent, AMD AIE, Cerebras, NKI DMA | none: core grid as an addressed dimension, DMA token as a state variable, masked write by `select` and in place | S: docs and one tutorial |
| 4 | Strided regions (L19) | NVIDIA TMA, Gemmini, NVDLA, Tenstorrent | `d1.region(starts, size=sizes, stride=strides)` | M: regions, placement marks |
| 5 | `convolution` (L20) | Eyeriss, NVDLA, TPU | `ops.convolution(x, w, window_size=[3, 3], window_stride=[1, 1], ...)` | M: attribute set |
| 6 | Write masks (L18) | Arm SME, SVE, NKI, Ascend vector unit | `write(r, value, mask=m)` with `m` a `pred` value | M: S5.2 semantics, B allocator |
| 7 | Indirect regions (L23) | Trainium indirect DMA, TMA gather, TPU SparseCore, MTIA | a start read from a region value in place of an expression | M: B allocator |
| 8 | Sub-computations (L22) | any design with custom reductions or scatter | a traced reducer function, `ops.reduce(x, init, to_apply=body)` | L: tracer, B e-graph |

Already describable, no extension needed:

- Systolic arrays and matrix engines (TPU MXU, Gaudi MME, AMX tiles, Ascend Cube, Gemmini): `dot` over regions.
- Accumulators (Trainium PSUM, AMX tile registers, Ascend L0C): a data model plus an in-place input.
- Configuration registers (Gemmini `config`, NVDLA register groups): state variables and state updates.
- CGRAs and Groq-style static dataflow: only at the ISA level; their spatial mapping is outside TAIDL's scope.
