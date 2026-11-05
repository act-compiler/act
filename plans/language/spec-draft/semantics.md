# Semantics

This page is S7 of the spec.
A semantics gives an instruction's meaning as a block of lines in a dialect of XLA-HLO text.

## S7.1 Lines, values, and `ROOT`

- `add_semantics(semantics)` takes one `ENTRY` block and parses it at once; a second call replaces the first.
- The block carries no name; it is the semantics of the instruction `add_semantics` is called on.
- Each line defines one value: its name, its element type and shape, an operation word with its operands, then any operation attributes.
- Each operation attribute follows a comma; the semicolon that ends a line is optional.
- A value's name is `%` followed by a letter or `_`, then letters, digits, `_`, and dots, as in `%In0` or `%dot`.
- A bare word, an `IDENTIFIER` with no `%`, stands only as an operation attribute's name or value, or as `constant`'s `true` or `false`.
- Text from `//` to the end of its line is a comment; there is no block comment.
- A semantics defines at least `%Out0`.
- Exactly one line carries `ROOT`: the line defining the output value of the highest k, wherever that line stands.
- `ROOT` has no run-time effect (S5.6).
- An instruction with no inputs, no outputs, and no semantics only changes state; `add_semantics` is not called for it.

```
ENTRY {
    // <comment>
    %In0 = <type>[<shape>] parameter(0);
    %<value> = <type>[<shape>] <operation>(<operands>), <attribute>=<value>;
    ROOT %Out0 = <type>[<shape>] <operation>(<operands>);
}
```

With two outputs, each output value has its own line and `ROOT` marks `%Out1`:

```
ENTRY {
    %In0 = f32[8] parameter(0);
    %Out0 = f32[4] slice(%In0), slice={[0:4]};
    ROOT %Out1 = f32[4] slice(%In0), slice={[4:8]};
}
```

## S7.2 Input and output values

- Input region k, counting from 0, enters as `%In<k>`, defined by `parameter(k)` and by no other line.
- Output region k, counting from 0, is written from the value `%Out<k>`.
- `%In<k>` and `%Out<k>` spell k in decimal with no leading zero.
- Each region has its value and each `%In<k>` and `%Out<k>` has its region.
- That check runs once both the semantics and the regions are given, in either order.
- A region value has its buffer's element type and the region shape (S5.2); for `d0` that is `u8[<size>]`.
- S5.6 gives the index map between a region value and its buffer.
- A region value line has its region's shape and element type; another shape needs an explicit `reshape` line.
- A region value line with another shape or element type than its region's is an error: rejected when the shapes are literal, a fault when they depend on a backtick value (S1.3.8).

## S7.3 Element types

A type is one of `pred`, `s2`, `s4`, `s8`, `s16`, `s32`, `s64`, `u2`, `u4`, `u8`, `u16`, `u32`, `u64`, `f8e4m3fn`, `f8e5m2`, `f16`, `bf16`, `f32`, `f64`, as XLA-HLO text spells them.

- `pred` is the boolean; `s<n>` and `u<n>` are the signed and unsigned integers of n bits.
- `f16`, `bf16`, `f32`, and `f64` are the floats of those names; `f8e4m3fn` and `f8e5m2` are 8-bit floats named by their exponent and mantissa bits.
- Each float type has the bit layout and special values of the `ml_dtypes` type of the same name, as XLA-HLO uses; `f8e4m3fn` has no infinity.
- XLA-HLO's complex types, one-bit integers, other 8-bit floats, and 4- and 6-bit floats are outside the language.
- A quantized type, with a scale and a zero point beside its storage type, is outside the language.

## S7.4 Shapes

- A shape lists dimensions between brackets, each an integer or an expression in backticks.
- Each dimension is at least 1; a literal 0 is rejected, and a backtick dimension below 1 is a fault (S1.3.7).
- `f32[]` is a rank-0 value.
- A backtick expression in a shape reads the state after the instruction's updates (S5.6).

## S7.5 Operands and operation attributes

- An operand is a value by its `%` name, defined by an earlier line.
- No two lines define the same name.
- `parameter` takes the number of an input region instead: an unsigned integer k.
- `constant` takes one literal instead: a number, `true`, `false`, or an expression in backticks.
- At an integer type the literal is an integer literal or a backtick expression whose value lies in the type's range.
- At `pred` it is `true`, `false`, `0`, or `1`.
- At a float type it is any number literal or a backtick expression; a literal is its exact decimal value rounded once to nearest, ties to even, and an underflow rounds normally.
- A finite value that rounds past the type's largest finite value, or `inf` at a type with no infinity, is rejected for a literal and a fault for a backtick expression (S1.3.9).
- `nan` is the type's canonical quiet NaN; `-nan` is it with the sign bit set.
- An operation attribute is `name=value`, named as XLA-HLO text names it.
- Its value takes one of three forms:

| form | example |
| --- | --- |
| value | `direction=LT` |
| brace list | `dimensions={1,0}`, `dimensions={}` |
| slice list | `slice={[0:4:2], [1:2]}` |

- A value is a number, a bare word such as `LT` or `true`, or an expression in backticks.
- A brace list holds zero or more values.
- A slice list holds one or more ranges, each `[start:limit]` or `[start:limit:stride]`.
- `pad` takes StableHLO's three attributes `edge_padding_low`, `edge_padding_high`, and `interior_padding`, each a brace list, not XLA-HLO's `padding=0_1x2_3`.
