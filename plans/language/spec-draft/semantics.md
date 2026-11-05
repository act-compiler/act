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
- A written value with another shape or element type than its region's is rejected when the shapes are plain integers, and a fault when they depend on an attribute or state (S1.3.8).
- A value from another instruction's function is rejected.

## S7.3 Element types

A type is one of `pred`, `s2`, `s4`, `s8`, `s16`, `s32`, `s64`, `u2`, `u4`, `u8`, `u16`, `u32`, `u64`, `f8e4m3fn`, `f8e5m2`, `f16`, `bf16`, `f32`, `f64`, as XLA-HLO text spells them, each an object of `taidl.types`.

- `pred` is the boolean; `s<n>` and `u<n>` are the signed and unsigned integers of n bits.
- `f16`, `bf16`, `f32`, and `f64` are the floats of those names; `f8e4m3fn` and `f8e5m2` are 8-bit floats named by their exponent and mantissa bits.
- Each float type has the bit layout and special values of the `ml_dtypes` type of the same name, as XLA-HLO uses; `f8e4m3fn` has no infinity.
- XLA-HLO's complex types, one-bit integers, other 8-bit floats, and 4- and 6-bit floats are outside the language, except `f4e2m1fn` and `f8e8m0fnu`, which plan item L17 adds with their edge cases (Q2.4).
- A quantized type, with a scale and a zero point beside its storage type, is outside the language.

## S7.4 Shapes

- The front end infers each value's shape and element type from its operands by the rules of S8; no line restates them.
- A shape is written only where an operation takes one, as a list of dimensions: `ops.reshape`, `ops.broadcast`, `ops.iota`, and `ops.constant`.
- A dimension is a plain `int` or an integer expression over computational attributes and state (S6.1).
- Each dimension is at least 1; a plain 0 is rejected, and a symbolic dimension below 1 is a fault (S1.3.7).
- `[]` is the rank-0 shape.
- A symbolic dimension reads the state before the issue (S5.6).

## S7.5 Operands and operation attributes

- An operand is a tensor value, passed by position in the order of the reference operator (S8.4).
- A target element type or shape follows the operands by position where the operation takes one: `ops.convert(x, f32)`, `ops.reshape(x, [64, 8])`.
- `ops.constant(value, element_type=s64, shape=[])` takes one literal: a number, `True`, `False`, or an integer expression.
- At an integer type the value is a plain integer or an integer expression whose value lies in the type's range.
- At `pred` it is `True`, `False`, `0`, or `1`.
- At a float type it is a plain number, `float('inf')`, `float('nan')`, or an integer expression; a plain number is its exact decimal value rounded once to nearest, ties to even, and an underflow rounds normally.
- A finite value that rounds past the type's largest finite value, or an infinity at a type with no infinity, is rejected for a plain value and a fault for an integer expression (S1.3.9).
- A NaN is the type's canonical quiet NaN, with the sign bit of the given value.
- An operation attribute is a keyword argument, named as S8.2 names it, such as `direction='EQ'` or `dimensions=[1, 0]`.

| S8.2 form | keyword value | example |
| --- | --- | --- |
| a word | a `str` | `direction='LT'` |
| an integer | a plain `int` or an integer expression over computational attributes and state | `iota_dimension=0` |
| a list of integers | a list, possibly empty | `dimensions=[1, 0]`, `dimensions=[]` |
| a list of ranges | a list of `(start, limit)` or `(start, limit, stride)` tuples | `slice=[(0, 4, 2), (1, 2)]` |

- `ops.pad` takes StableHLO's three attributes `edge_padding_low`, `edge_padding_high`, and `interior_padding`, each a list.
