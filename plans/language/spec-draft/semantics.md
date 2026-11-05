# Semantics

This page is S7 of the spec.
A semantics gives an instruction's meaning as a block of lines in a dialect of XLA-HLO text.

## S7.1 Lines, values, and `ROOT`

- `add_semantics(semantics)` takes one `ENTRY` block and parses it at once; a second call replaces the first.
- The block carries no name; it is the semantics of the instruction `add_semantics` is called on.
- Each line defines one value: its name, its element type and shape, an operation word with its operands, then any operation attributes.
- Each operation attribute follows a comma; the semicolon that ends a line is optional.
- A value's name is `%` followed by a letter or `_`, then letters, digits, `_`, and dots, as in `%In0` or `%dot`.
- A word with no `%` is never a type word, `ENTRY`, `ROOT`, `inf`, or `nan`, and is an operation word only where it names an operation attribute.
- Text from `//` to the end of its line is a comment; there is no block comment.
- Exactly one line carries `ROOT`: the line defining the output value of the highest k, wherever that line stands.
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
- Each region has its value and each `%In<k>` and `%Out<k>` has its region.
- That check runs once both the semantics and the regions are given, in either order.
- A region value has its data model's element type and the region shape (S5.2); for `d0` that is `u8[<size>]`.
- Element `[i_1, ..., i_n, u_1, ..., u_m]` of the region value is element `[s_1 + i_1, ..., s_n + i_n, u_1, ..., u_m]` of the data model, where `s_j` is the region's j-th start.
- For `d0`, element `[i]` of the region value is byte `s_1 + i`.
- A line may give a region value another shape with the same element count; the value is then read or written in row-major order, as `reshape` does: a presumed reshape.
- A region value with another element count, or another element type than its data model's, is an error in the ISA description, and undefined (S1.3).

## S7.3 Element types

A type is one of `pred`, `s2`, `s4`, `s8`, `s16`, `s32`, `s64`, `u2`, `u4`, `u8`, `u16`, `u32`, `u64`, `f8e4m3fn`, `f8e5m2`, `f16`, `bf16`, `f32`, `f64`, as XLA-HLO text spells them.

- `pred` is the boolean; `s<n>` and `u<n>` are the signed and unsigned integers of n bits.
- `f16`, `bf16`, `f32`, and `f64` are the floats of those names; `f8e4m3fn` and `f8e5m2` are 8-bit floats named by their exponent and mantissa bits.
- XLA-HLO's complex types, one-bit integers, other 8-bit floats, and 4- and 6-bit floats are outside the language.
- A quantized type, with a scale and a zero point beside its storage type, is outside the language.

## S7.4 Shapes

- A shape lists dimensions between brackets, each an integer or an expression in backticks.
- `f32[]` is a rank-0 value.
- A backtick expression in a shape is evaluated with the state after the instruction's updates (S5.4).

## S7.5 Operands and operation attributes

- An operand is a value by its `%` name, defined by an earlier line.
- No two lines define the same name.
- `parameter` takes the number of an input region instead: an unsigned integer k.
- `constant` takes one literal instead: a number such as `7`, `-128`, `0.5`, `1e-3`, `inf`, `-inf`, or `nan`; `true`; `false`; or an expression in backticks.
- An operation attribute is `name=value`, named as XLA-HLO text names it.
- Its value takes one of three forms:

| form | example |
| --- | --- |
| value | `direction=LT` |
| brace list | `dimensions={1,0}`, `dimensions={}` |
| slice list | `slice={[0:4:2], [1:2]}` |

- A value is a number, a word with no `%` such as `LT` or `true`, or an expression in backticks.
- A brace list holds zero or more values.
- A slice list holds one or more ranges, each `[start:limit]` or `[start:limit:stride]`.
- `pad` takes StableHLO's three attributes `edge_padding_low`, `edge_padding_high`, and `interior_padding`, each a brace list, not XLA-HLO's `padding=0_1x2_3`.
