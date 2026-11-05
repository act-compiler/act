# Operation meaning

This page is S8.4 and S8.5 of the spec.

## S8.4 Meaning and edge cases

### Reference

- Each operation word means its reference operation: the StableHLO operation of S8.1, or the XLA-HLO operation where only XLA-HLO has it.
- The reference is the StableHLO specification at the version bundled with the oracle's pinned jax, and XLA-HLO's operation semantics at that jaxlib's XLA commit; plan item L2 records the pin.
- The reference gives the result's shape, its element type, and its values; the rules below override it where they speak.
- `dot` is StableHLO's `dot_general` (S8.2 gives its attribute defaults); both operands have one element type.
- `broadcast` is StableHLO's `broadcast_in_dim`; its `dimensions` gives, for each operand dimension, the result dimension it maps to.
- `reduce_<reducer>` is StableHLO's `reduce` over the dimensions its `dimensions` gives, with the reducer of S8.1.
- `copy` returns its operand.
- `constant` with a non-scalar shape gives every element the one literal it takes.
- `pad`'s second operand is the rank-0 padding value.
- `compare` without `type` takes FLOAT for floats, SIGNED for `s` types, and UNSIGNED for `u` types and `pred`.
- An axis in an operation attribute lies in `[0, rank)` as StableHLO requires; a literal outside is rejected, a backtick outside is a fault (S1.3.8).
- A line's type and shape are the result's; a line whose type or shape differs from what its reference operation gives is an error: rejected when literal, a fault when it depends on a backtick value (S1.3.8).
- Operand order of every word is the reference operation's.

### Initial value of `reduce_<reducer>`

| reducer | initial value |
| --- | --- |
| `add`, `or`, `xor` | 0; +0.0 for floats |
| `multiply` | 1 |
| `and` | every bit set; `true` for `pred` |
| `max` | -inf for floats with an infinity; the lowest finite value for `f8e4m3fn`; the least value of an integer type |
| `min` | +inf for floats with an infinity; the greatest finite value for `f8e4m3fn`; the greatest value of an integer type |

### Integer edge cases inside a semantics

- Integer `x / 0` is -1 (every bit set); `x % 0` is `x`.
- `INT_MIN / -1` is `INT_MIN`, and `INT_MIN % -1` is 0.
- A shift by the width or more gives 0, or the sign fill for `shift_right_arithmetic`.
- A negative integer exponent of `power` gives 0, except for bases 1 and -1.
- A float-to-integer `convert` truncates toward 0, saturates out of range, and maps NaN to 0.

### Byte layout in `d0`

- A multi-byte value is little-endian: byte 0 of an element is its least significant byte.
- A sub-byte type packs 8/n elements per byte, the lowest index in the least significant bits.
- `pred` is one byte, 0 or 1.
- `bitcast_convert` follows this layout between element types of different widths.

### Float accuracy

- The accumulation type and order of a float `dot` or `reduce_*`, and the accuracy of transcendental words, are open (S1.3.15), pending the tech-lead panel.

## S8.5 Operations outside the language

The operation words of S8.1 are the whole set; every other XLA-HLO or StableHLO operation is outside the language, by kind:

- `ragged_dot` and `scaled_dot`: each takes group sizes or scales that no StableHLO operation takes.
- An operation that takes a body, a comparator or a callee, which a line cannot carry: `reduce` with a reducer of its own, `reduce_window`, `scatter`, `sort`, `select_and_scatter`, `map`, `while`, `conditional` (StableHLO's `if` and `case`), `call`, `fusion`, `scan`, `composite` and `custom_call`.
- `convolution`: its window and its dimension labels need syntax the grammar lacks.
- Complex numbers, random numbers and quantized values, as in `complex`, `fft`, `rng`, `stochastic_convert` and `uniform_quantize`: a semantics has no such type and no random state.
- An operation with several results, such as `batch_norm_training`, `topk` or `tuple`: a line defines one value.
- An operation on a shape that no expression of attributes and state gives, such as `dynamic_reshape` or `get_dimension_size`: the text gives every shape as a function of the instruction's attributes and state.
- An operation that orders effects, reaches another device or serves the compiler alone, such as `after_all`, `all_reduce`, `send`, `optimization_barrier` or XLA-HLO's `bitcast`: it says how a program runs, not what an instruction computes.
- An operation StableHLO has but its specification leaves out, such as `einsum` or `torch_index_select`.
