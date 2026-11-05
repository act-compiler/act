# Semantics

This page is S7 of the TAIDL language specification.
A semantics gives an instruction's meaning as the tensor values its function computes from region values, with operations of S8.

## S7.1 Values and operations

- The semantics of an instruction is every tensor value its function builds from `read` results, `ops.constant`, and `ops.iota`, through operations, up to the values it passes to `write`.
- Each operation call, and each tensor operator of S2.2, records one value with its element type and shape.
- A value is a Python object; a Python variable that holds it is a name for the reader, never for the front end.
- The value passed to the k-th `write` is the output value of output region k.
- Each value the function computes runs at issue, whether or not a `write` uses it (S5.6).
- An instruction with no `read` and no `write` only changes state.

With two outputs, each `write` takes its own value:

```python
@acc.instruction()
def split(src: Addr, lo: Addr, hi: Addr):
    x = read(v.region([src], size=[2]))
    write(v.region([lo], size=[1]), ops.slice(x, slice=[(0, 1), (0, 4)]))
    write(v.region([hi], size=[1]), ops.slice(x, slice=[(1, 2), (0, 4)]))
```

Here `v` is a buffer with base element shape `[4]`, so `x` has shape `[2, 4]` and each output `[1, 4]`.

## S7.2 Input and output values

- `read(r)` returns the input value of its region: the buffer's element type and the region shape (S5.2); for `d0` that is `u8[<size>]`.
- S5.6 gives the index map between a region value and its buffer.
- The value passed to `write(r, value)` has the region's element type and shape; another shape needs an explicit `ops.reshape`, another type an explicit `ops.convert`.
- A written value whose element type or rank differs from its region's is rejected.
- A dimension that differs is rejected when both are plain, and is a fault otherwise (S1.3.8).
- A symbolic object made in another trace is rejected, except a region whose starts and sizes are all plain, which belongs to no trace (S3.1).

## S7.3 Element types

A type is one of the 21 objects of `taidl.types` below, as XLA-HLO text spells them.

- `pred`, the boolean.
- `s2`, `s4`, `s8`, `s16`, `s32`, `s64`: the signed integers of n bits.
- `u2`, `u4`, `u8`, `u16`, `u32`, `u64`: the unsigned integers of n bits.
- `f16`, `bf16`, `f32`, `f64`: the floats of those names.
- `f8e4m3fn`, `f8e5m2`, `f4e2m1fn`: floats named by their exponent and mantissa bits.
- `f8e8m0fnu`: the 8-bit scale type of the MX formats, a power of two.

Each float type has the bit layout and special values of the `ml_dtypes` types `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1fn`, `float8_e8m0fnu`, and `bfloat16`, and of the IEEE types of the other names, at the version the pinned jax requires, as XLA-HLO uses.

| type | values | zero | sign | infinity | NaN |
| --- | --- | --- | --- | --- | --- |
| `f8e4m3fn` | finite up to 448 | +0, -0 | yes | none | yes |
| `f4e2m1fn` | 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives | +0, -0 | yes | none | none |
| `f8e8m0fnu` | `2^k` for k from -127 to 127 | none | none, every value positive | none | one, bits `0xFF` |

- The other floats have a signed zero, both infinities, and NaN.
- S8.4 gives the results of the three types of the table where a value has no encoding.
- XLA-HLO's complex types, one-bit integers, other 8-bit floats, and other 4- and 6-bit floats are outside the language.
- A quantized type, with a scale and a zero point beside its storage type, is outside the language.

## S7.4 Shapes

- The front end infers each value's shape and element type from its operands by the rules of S8; no line restates them.
- A shape is written only where an operation takes one, as a list of dimensions: `ops.reshape`, `ops.broadcast`, `ops.iota`, and `ops.constant`.
- A dimension is a plain `int` or an integer expression over computational attributes and state (S6.1).
- Every dimension of every value, written or inferred, is at least 1.
- An operation whose result has a dimension below 1 is rejected when plain, and is a fault otherwise (S1.3.7); a plain dimension below 1 is rejected.
- `[]` is the rank-0 shape.
- A symbolic dimension reads the state before the issue (S5.6).

## S7.5 Operands and operation attributes

- An operand is a tensor value, passed by position in the order of the reference operator (S8.4).
- A target element type or shape is a parameter of the words below; `...` in `dot` stands for its operation attributes (S8.2).

| word | signature |
| --- | --- |
| `constant` | `ops.constant(value, element_type=s64, shape=[])` |
| `iota` | `ops.iota(element_type, shape, iota_dimension=d)` |
| `convert` | `ops.convert(x, element_type)` |
| `bitcast_convert` | `ops.bitcast_convert(x, element_type)` |
| `reshape` | `ops.reshape(x, shape)` |
| `broadcast` | `ops.broadcast(x, shape, dimensions=...)` |
| `dot` | `ops.dot(lhs, rhs, ..., element_type=None)` |

- `iota_dimension` has no default; a target may also be passed by its keyword.
- `ops.constant(value, element_type=s64, shape=[])` takes one literal: a number, `True`, `False`, or an integer expression.
- At an integer type the value is a plain integer or an integer expression whose value lies in the type's range.
- A plain `constant` value is not an expression: at an integer type it is any `int` in the type's range, `u64` included; an integer expression stays within signed 64 bits (S6.2).
- At `pred` it is `True`, `False`, `0`, or `1`; `True` and `False` are values only at `pred`.
- At a float type it is a plain number, `float('inf')`, `float('nan')`, or an integer expression; a plain number is its exact value rounded once to nearest, ties to even, and an underflow rounds normally.
- A `float`'s exact value is its binary64 value, and an `int` or a `fractions.Fraction` is exact; a decimal rounded once is written as a `Fraction`, such as `Fraction('0.1')`.
- A finite value that rounds past the type's largest finite value, or an infinity at a type with no infinity, is rejected for a plain value and a fault for an integer expression (S1.3.9).
- A NaN is the type's NaN; at `f8e8m0fnu` it is `0xFF`.
- At `f4e2m1fn` a NaN, and at `f8e8m0fnu` a zero or a negative value, has no encoding: rejected for a plain value and a fault for an integer expression (S1.3.9).
- At `f8e8m0fnu` a positive value rounds as `convert` to `f8e8m0fnu` does (S8.4).
- An operation attribute is a keyword argument, named as S8.2 names it, such as `direction='EQ'` or `dimensions=[1, 0]`.

| S8.2 form | keyword value | example |
| --- | --- | --- |
| a word | a `str` | `direction='LT'` |
| an integer | a plain `int` or an integer expression over computational attributes and state; a plain `int` where it names an axis (S8.2) | `exponent_bits=5`, `iota_dimension=0` |
| a list of integers | a list, possibly empty | `dimensions=[1, 0]`, `dimensions=[]` |
| a list of ranges | a list of `(start, limit)` or `(start, limit, stride)` tuples | `slice=[(0, 4, 2), (1, 2)]` |

- Each entry of a list or a range is an integer of S8.2: a plain `int` or an integer expression, and a plain `int` where it names an axis.
- A list is a `list` or a `tuple`.
- `ops.pad` takes StableHLO's three attributes `edge_padding_low`, `edge_padding_high`, and `interior_padding`, each a list.
