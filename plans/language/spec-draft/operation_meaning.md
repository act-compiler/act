# Operation meaning

This page is S8.4 and S8.5 of the spec.

## S8.4 Meaning and edge cases

- Each operation word means its reference operation: the StableHLO operation of S8.1, or the XLA-HLO operation where only XLA-HLO has it.
- That reference operation gives the result's shape, its element type, and its values, including at edge cases such as NaN, infinity, and integer overflow.
- `dot` is StableHLO's `dot_general` (S8.2 gives its attribute defaults).
- `broadcast` is StableHLO's `broadcast_in_dim`; its `dimensions` gives, for each operand dimension, the result dimension it maps to.
- `reduce_<reducer>` is StableHLO's `reduce` over the dimensions its `dimensions` gives, with the reducer of S8.1.
- The initial value of `reduce_<reducer>` is not yet fixed by the spec and is implementation-defined.
- `constant` with a non-scalar shape gives every element the one literal it takes.
- `pad`'s second operand is the rank-0 padding value.
- The bit layout `bitcast_convert` uses between element types of different widths is implementation-defined (S1.3).
- A line's type and shape are the result's; a line whose type or shape differs from what its reference operation gives is undefined (S1.3).

## S8.5 Operations outside the language

The operation words of S8.1 are the whole set; every other XLA-HLO or StableHLO operation is outside the language, by kind:

- `ragged_dot` and `scaled_dot`: each takes group sizes or scales that no StableHLO operation takes.
- An operation that takes a body, a comparator or a callee, which a line cannot carry: `reduce` with a reducer of its own, `reduce_window`, `scatter`, `sort`, `select_and_scatter`, `map`, `while`, `conditional` (StableHLO's `if` and `case`), `call`, `fusion`, `scan`, `composite` and `custom_call`.
- `convolution`: its window and its dimension labels need syntax the grammar lacks.
- Complex numbers, random numbers and quantized values, as in `complex`, `fft`, `rng`, `stochastic_convert` and `uniform_quantize`: a semantics has no such type and no random state.
- An operation with several results, such as `batch_norm_training`, `topk` or `tuple`: a line defines one value.
- An operation on a shape known only at run time, such as `dynamic_reshape` or `get_dimension_size`: the text gives every shape.
- An operation that orders effects, reaches another device or serves the compiler alone, such as `after_all`, `all_reduce`, `send`, `optimization_barrier` or XLA-HLO's `bitcast`: it says how a program runs, not what an instruction computes.
- An operation StableHLO has but its specification leaves out, such as `einsum` or `torch_index_select`.
