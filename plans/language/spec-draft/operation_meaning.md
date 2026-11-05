# Operation meaning

This page is S8.4 and S8.5 of the TAIDL language specification.

## S8.4 Meaning and edge cases

### Reference

- Each tensor operator means its reference operator: the StableHLO operation of S8.1, or the XLA-HLO operation where only XLA-HLO has it.
- The reference is the StableHLO specification at the version bundled with the oracle's pinned jax, and XLA-HLO's operation semantics at that jaxlib's XLA commit; plan item L2 records the pin.
- The reference gives the result's shape, its element type, and its values; the rules below override it where they speak.
- `dot` is StableHLO's `dot_general` (S8.2 gives its attribute defaults); both operands have one element type.
- `ops.dot` takes `element_type=T`, by default its operands' type; the operands and T are all integer or all float.
- `dot` sums its exact products exactly: an integer T wraps the sum to T, and a float T meets the `dot` bound of Float accuracy below, with u of T and n = K.
- `broadcast` is StableHLO's `broadcast_in_dim`; its `dimensions` gives, for each operand dimension, the result dimension it maps to.
- `reduce_<reducer>` is StableHLO's `reduce` over the dimensions its `dimensions` gives, with the reducer of S8.1.
- `copy` returns its operand.
- `constant` with a non-scalar shape gives every element the one literal it takes.
- `pad`'s second operand is the rank-0 padding value.
- `compare` without `type` takes FLOAT for floats and `f8e8m0fnu`, SIGNED for `s` types, and UNSIGNED for `u` types and `pred`.
- An axis in an operation attribute is a plain `int` (S8.2) in `[0, rank)` as StableHLO requires; one outside is rejected.
- A value's type and shape are what its reference operator gives; the front end infers them, and operands the reference operator does not accept are an error.
- An operand element type or rank the reference operator does not accept is rejected; a dimension it does not accept is rejected when plain, and a fault when it depends on an attribute or state (S1.3.8).
- Operand order of every word is the reference operator's.

### Initial value of `reduce_<reducer>`

| reducer | initial value |
| --- | --- |
| `add`, `or`, `xor` | 0; +0.0 for floats |
| `multiply` | 1 |
| `and` | every bit set; `true` for `pred` |
| `max` | -inf for floats with an infinity; the lowest finite value for `f8e4m3fn` and `f4e2m1fn`; the least value of an integer type |
| `min` | +inf for floats with an infinity; the greatest finite value for `f8e4m3fn` and `f4e2m1fn`; the greatest value of an integer type |

- No reducer takes `f8e8m0fnu` (S8.3).

### Integer edge cases inside a semantics

- Every integer result wraps to its type, as two's complement for `s` types; `abs` and `negate` of the least value give the least value.
- Integer `x / 0` has every bit set: -1 for `s` types, the greatest value for `u` types; `x % 0` is `x`.
- `INT_MIN / -1` is `INT_MIN`, and `INT_MIN % -1` is 0.
- A shift by the width or more gives 0, or the sign fill for `shift_right_arithmetic`.
- A shift amount is read as unsigned, so a negative amount shifts by the width or more.
- A negative integer exponent of `power` gives 0, except for bases 1 and -1.
- An index inside a semantics, as in `dynamic_slice` or `gather`, clamps as StableHLO states.
- Tensor values never fault; the oracle lowers each fixed value above explicitly.

### `convert` out of range

| conversion class | result |
| --- | --- |
| float to float, int to float | round to nearest even; overflow gives a signed infinity, or NaN where the type has none (`f8e4m3fn`) |
| float or int to `f4e2m1fn` | round to nearest even; overflow and an infinity saturate to 6 or -6; NaN gives +0 |
| float or int to `f8e8m0fnu` | the power of two nearest by value, so 1.5 * 2^k rounds up; a positive value below `2^-127` gives `2^-127`; a value that rounds past `2^127`, an infinity, NaN, a zero, and a negative value give NaN |
| `f8e8m0fnu` to float or int | its value `2^k`, then the row of its target class; NaN as a float NaN |
| float to int | truncate toward 0, saturate out of range, NaN gives 0 |
| int to int | wrap |
| any type to `pred` | nonzero, NaN included, is `true` |

- A saturating ISA writes `clamp` before `convert`.
- A float word gives what `convert` to its result type gives from the exact or bounded value, so overflow gives NaN at `f8e4m3fn` and saturates at `f4e2m1fn`.

### Float rounding, NaN, and signed zero

- Every float operation, except an approximate word and the accumulations below, rounds once to its result type, to nearest even.
- Denormals are kept; a unit that flushes them writes the flush as a `select` against the smallest normal.
- When results are compared for conformance, any NaN matches any NaN, and sign and payload are outside the meaning; `compare` follows its `type`.
- `bitcast_convert` of a NaN from a float source gives the positive canonical quiet NaN bits of that type.
- `maximum` and `minimum` follow IEEE 754-2019: NaN propagates, and -0 is less than +0.

### Accuracy of approximate words

- The approximate words are `rsqrt`, `cbrt`, `power` at a float type, and the transcendental functions of S8.1, `exponential` to `atanh`.
- Let y be the exact value of the function at the operands, and r the result at result type t.
- The IEEE result is the correctly rounded value of y at t, with `pow` for `power`, `rSqrt` for `rsqrt`, `rootn(x, 3)` for `cbrt`, and the mathematical value for `erf` and `logistic`; at `f4e2m1fn` the IEEE result is converted as above.
- When the IEEE result is an infinity, a zero, or NaN, r equals it; otherwise r meets `|r - y| <= k_t * ulp_t(y)` (S1.3.19), and within it r may be 0 or a subnormal.
- `ulp_t(y) = 2^(e - p + 1)`, where p is the precision of t, mantissa bits plus one, and `e = floor(log2(max(|y|, m_t)))` with `m_t` the smallest normal value of t.
- When `|y| + k_t * ulp_t(y)` passes the largest finite value of t, r may also be what `convert` to t gives on overflow.

| result type t | `k_t` |
| --- | --- |
| `f64`, `f32` | 16 |
| `f16`, `bf16` | 8 |
| `f8e4m3fn`, `f8e5m2`, `f4e2m1fn` | 1 |

- The bound holds for any method a unit uses, a polynomial or a table, as long as it lands within `k_t` ulps of the result type.
- A unit less accurate than its result type's bound is written at a narrower result type and converted, such as an approximate `f32` `tanh` written at `f16`.
- No generator's values are the reference: a test compares against y within the bound, never bit for bit against one platform.

### Byte layout in `d0`

- A multi-byte value is little-endian: byte 0 of an element is its least significant byte.
- A sub-byte type packs 8/n elements per byte, the lowest index in the least significant bits.
- `pred` is one byte, 0 or 1.
- A `bitcast_convert` to `pred` reads a nonzero byte as `true`.
- A sub-byte type packs along the last dimension that `bitcast_convert` adds or removes, with index 0 in the least significant bits.
- `bitcast_convert` follows this layout between element types of different widths.

### Float accuracy

- The order and width of a float accumulation are free; the result r meets a bound against the exact value (S1.3.15).
- For `dot` and `reduce_add`: `|r - exact| <= gamma_n * sum|terms| + u * |exact|`.
- For `reduce_multiply`: `|r - exact| <= gamma_n * |exact|`.
- `gamma_n = n*u / (1 - n*u)`, where u is the unit roundoff of the result type and n the term count: K for `dot`, the product of the sizes of its contracting dimensions or 1 when there are none, the product of the reduced sizes for a reduce.
- When `sum|terms|` exceeds the largest finite value of the type, the result may also be an infinity or a NaN.
- For `n*u >= 1` the bound constrains nothing; an ISA author who needs one writes a wider result type.
- `reduce_max` and `reduce_min` are exact.
- A reduction over `dimensions=[]` or over dimensions of size 1 gives each element combined with the initial value.

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
