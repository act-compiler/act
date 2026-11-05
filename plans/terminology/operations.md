# Operation families

The TAIDL column lists operation words from `taidl/docs/operations.md`.
Each recommended term is the operation word the spec and the repos use in prose.

| Concept | TAIDL operation word(s) | MLIR | StableHLO / XLA | PyTorch Core ATen | Neuron NKI | Recommended term | One-line rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Elementwise | `add`, `multiply`, `exponential` | `linalg.generic` with parallel iterators; `arith.maximumf` | `add`, `multiply`, `exponential` [SHLO] | `add.Tensor`, `mul.Tensor`, `exp` | `tensor_tensor`, `tensor_scalar`, `activation` | elementwise operation | Universal; keep the XLA names. |
| Contraction | `dot` | `linalg.matmul`, `linalg.batch_matmul` | `dot_general` [SHLO]; `DotGeneral` [XLA] | `mm`, `bmm` | `nc_matmul` | `dot`, described as `dot_general` | Name `dot` and state that it takes `dot_general`'s dimension numbers. |
| Contraction dimensions | `lhs_contracting_dims`, `lhs_batch_dims` | indexing maps | `lhs_contracting_dimensions`, `lhs_batching_dimensions` [SHLO]; `lhs_batch_dimensions` [XLA] | n/a | n/a | contracting dimensions, batch dimensions | Concept names shared by both XLA documents. |
| Reduction | `reduce_add`, `reduce_max` | `linalg.reduce` | `reduce` with `body` and `init_values` [SHLO] | `sum.dim_IntList`, `amax`, `max.dim` | `tensor_reduce` | reduction; `reduce_<reducer>` words | The fused words are TAIDL's; define each as `reduce` with a fixed reducer. |
| Broadcast | `broadcast` | `linalg.broadcast` | `broadcast_in_dim` [SHLO]; `BroadcastInDim` vs `Broadcast` [XLA] | `expand` | n/a | `broadcast`, defined as `broadcast_in_dim` | XLA's `Broadcast` adds leading dimensions only, so state the meaning. |
| Reshape | `reshape` | `tensor.reshape`, `tensor.expand_shape`, `tensor.collapse_shape` | `reshape` | `view` | n/a | `reshape` | Universal. |
| Transpose | `transpose` | `linalg.transpose` | `transpose` with `permutation` | `permute` | `nc_transpose` | `transpose` | Universal. |
| Static slice | `slice` | `tensor.extract_slice` | `slice` with `start_indices`, `limit_indices`, `strides` | `slice.Tensor` | indexing | `slice` | Universal. |
| Dynamic slice | `dynamic_slice`, `dynamic_update_slice` | `tensor.extract_slice`, `tensor.insert_slice` | `dynamic_slice` with `start_indices`, `slice_sizes` | n/a | n/a | `dynamic_slice` | Universal. |
| Pad | `pad` | `tensor.pad` | `pad` with `edge_padding_low`, `edge_padding_high`, `interior_padding` | `constant_pad_nd` | n/a | `pad` | Universal. |
| Concatenate | `concatenate` | `tensor.concat` | `concatenate` with `dimension` | `cat` | n/a | `concatenate` | Universal. |
| Value conversion | `convert` | `arith.extf`, `arith.truncf`, `arith.sitofp`, `arith.fptosi` | `convert` [SHLO]; `ConvertElementType` [XLA] | `_to_copy`; `prims.convert_element_type` | n/a | `convert` | Universal. |
| Bit reinterpretation | `bitcast_convert` | `arith.bitcast`, `tensor.bitcast` | `bitcast_convert` [SHLO]; `BitcastConvertType` [XLA] | no Core ATen op | n/a | `bitcast_convert` | Never "cast" alone, which reads as `convert`. |
| Select | `select` | `arith.select` | `select` [SHLO]; `on_true`, `on_false` [XLA] | `where.self` | `affine_select` | `select` | Universal. |
| Clamp | `clamp` | n/a | `clamp` with `min`, `operand`, `max` | `clamp` | n/a | `clamp` | Universal. |
| Compare | `compare` | `arith.cmpi`, `arith.cmpf` with a predicate | `compare` with `comparison_direction` | `eq` | n/a | `compare` with a comparison direction | Universal. |
| Index sequence | `iota` | n/a | `iota` with `iota_dimension` | `arange.start_step`; `prims.iota` | `iota` | `iota` | Universal. |
| Constant | `constant` | `arith.constant`, `tensor.splat` | `constant` | `full` | `memset` | `constant` | Universal. |
| High half of a product | `mulhi` | `arith.mulsi_extended` | XLA-HLO only | n/a | n/a | `mulhi` | Say it is not a StableHLO operation. |
