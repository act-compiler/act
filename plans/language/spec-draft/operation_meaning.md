# Operation meaning

This page is S8.4 and S8.5 of the TAIDL language specification.

## S8.4 Meaning and edge cases

### Reference

- Each tensor operator means its reference operator: the StableHLO operation of S8.1, or the XLA-HLO operation where only XLA-HLO has it.
- The reference is the StableHLO specification at the version bundled with the oracle's pinned jax, and XLA-HLO's operation semantics at that jaxlib's XLA commit; plan item L2 records the pin.
- The reference gives the result's shape, its element type, and its values; the rules below override it where they speak.
- `dot` is StableHLO's `dot_general` (S8.2 gives its attribute defaults); both operands have one element type.
- `broadcast` is StableHLO's `broadcast_in_dim`; its `dimensions` gives, for each operand dimension, the result dimension it maps to.
- `reduce_<reducer>` is StableHLO's `reduce` over the dimensions its `dimensions` gives, with the reducer of S8.1.
- `copy` returns its operand.
- `constant` with a non-scalar shape gives every element the one literal it takes.
- `pad`'s second operand is the rank-0 padding value.
- `compare` without `type` takes FLOAT for floats, SIGNED for `s` types, and UNSIGNED for `u` types and `pred`.
- An axis in an operation attribute lies in `[0, rank)` as StableHLO requires; a plain value outside is rejected, a symbolic one outside is a fault (S1.3.8).
- A value's type and shape are what its reference operator gives; the front end infers them, and operands the reference operator does not accept are an error: rejected when the shapes are plain, a fault when they depend on an attribute or state (S1.3.8).
- Operand order of every word is the reference operator's.

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
- An index inside a semantics, as in `dynamic_slice` or `gather`, clamps as StableHLO states.
- Tensor values never fault; the oracle lowers each fixed value above explicitly.

### `convert` out of range

| conversion class | result |
| --- | --- |
| float to float, int to float | round to nearest even; overflow gives a signed infinity, or NaN where the type has none (`f8e4m3fn`) |
| float to int | truncate toward 0, saturate out of range, NaN gives 0 |
| int to int | wrap |
| any type to `pred` | nonzero, NaN included, is `true` |

- A saturating ISA writes `clamp` before `convert`.

### Float rounding, NaN, and signed zero

- Every float operation rounds once to its result type, to nearest even.
- Denormals are kept; a unit that flushes them writes the flush as a `select` against the smallest normal.
- Any NaN equals any NaN; sign and payload are outside the meaning.
- `bitcast_convert` of a NaN from a float source gives the positive canonical quiet NaN bits of that type.
- `maximum` and `minimum` follow IEEE 754-2019: NaN propagates, and -0 is less than +0.
- Infinity, zero, and NaN results of every approximate word match the IEEE function exactly.

### Byte layout in `d0`

- A multi-byte value is little-endian: byte 0 of an element is its least significant byte.
- A sub-byte type packs 8/n elements per byte, the lowest index in the least significant bits.
- `pred` is one byte, 0 or 1.
- `bitcast_convert` follows this layout between element types of different widths.

### Float accuracy

- The order and width of a float accumulation are free; the result r meets a bound against the exact value (S1.3.15).
- For `dot` and `reduce_add`: `|r - exact| <= gamma_n * sum|terms| + u * |exact|`.
- For `reduce_multiply`: `|r - exact| <= gamma_n * |exact|`.
- `gamma_n = n*u / (1 - n*u)`, where u is the unit roundoff of the result type and n the term count: K for `dot`, the product of the reduced sizes for a reduce.
- When `sum|terms|` exceeds the largest finite value of the type, the result may also be an infinity or a NaN.
- For `n*u >= 1` the bound constrains nothing; an ISA author who needs one writes a wider result type.
- `reduce_max` and `reduce_min` are exact.
- A size-1 or empty reduction gives the element combined with the identity.

## S8.5 Operations outside the language

The tensor operators of S8.1 are the whole set; every other XLA-HLO or StableHLO operation is outside the language, by kind:

- `ragged_dot` and `scaled_dot`: each takes group sizes or scales that no StableHLO operation takes.
- An operation that takes a body, a comparator or a callee, which an `ops` call cannot carry: `reduce` with a reducer of its own, `reduce_window`, `scatter`, `sort`, `select_and_scatter`, `map`, `while`, `conditional` (StableHLO's `if` and `case`), `call`, `fusion`, `scan`, `composite` and `custom_call`.
- `convolution`: its window and its dimension labels need attributes S8.2 lacks.
- Complex numbers, random numbers and quantized values, as in `complex`, `fft`, `rng`, `stochastic_convert` and `uniform_quantize`: a semantics has no such type and no random state.
- An operation with several results, such as `batch_norm_training`, `topk` or `tuple`: an `ops` call returns one value.
- An operation on a shape that no expression of attributes and state gives, such as `dynamic_reshape` or `get_dimension_size`: the specification gives every shape as a function of the instruction's attributes and state.
- An operation that orders effects, reaches another device or serves the compiler alone, such as `after_all`, `all_reduce`, `send`, `optimization_barrier` or XLA-HLO's `bitcast`: it says how a program runs, not what an instruction computes.
- An operation StableHLO has but its specification leaves out, such as `einsum` or `torch_index_select`.
