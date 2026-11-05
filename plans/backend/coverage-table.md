# Coverage of TAIDL

This table lists each TAIDL language feature and whether ACT-Backend generates a backend for a spec that uses it.
Section ids are those of the TAIDL spec outline.
A `rejected` feature makes `generate_backend` raise `ValueError` naming the instruction and the feature, before anything is written.
A `partial` feature is supported within the limitation its row states.

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| Accelerator and its name | S4.2 | supported | |
| Names of data models, instructions, attributes, state variables | S4.1 | supported | |
| Data model with addressed dimensions and unit shape | S4.3 | partial | An addressed dimension above the signed 64-bit range is rejected. |
| Off-chip memory `d0` | S4.3 | supported | |
| Data model element type `var_type` | S4.3 | partial | The generator does not check that a region's value has its data model's element type. |
| State variable and its initial value | S4.4 | partial | Declaring one is accepted; any read or update of it is rejected (rows below). |
| Instruction with semantics | S5 | supported | |
| Instruction with no semantics | S5, S7 | partial | Left out of the backend: it is never selected. |
| Computational attribute `@c.` | S5.1, S6.1 | partial | Each must appear in a dimension of a shape, or in an operation attribute, alone or times a positive integer; else rejected. |
| Addressing attribute `@a.` in a start or a constraint | S5.1, S6.1 | supported | |
| Addressing attribute `@a.` in a semantics | S6.1 | rejected | |
| One input region or several | S5.2 | supported | |
| Exactly one output region | S5.2 | supported | |
| Zero or several output regions | S5.2 | rejected | |
| Start: integers, `@c.`, `@a.`, `+ - * / %`, unary `-`, brackets | S5.2, S6.2 | supported | A start whose integers alone pass 64 bits, or that divides by an operand whose integers alone compute zero, is rejected. |
| Size: integers, `@c.`, `+ - * / %`, unary `-`, brackets | S5.2, S6.2 | supported | A size whose integers alone pass 64 bits, or that divides by an operand whose integers alone compute zero, is rejected. |
| Size that overflows 64 bits for some attribute values | S6.2 | supported | The instruction is not selected for those values. |
| Presumed reshape of a region's value | S7.2 | partial | The element count is checked only when every dimension and size is an integer literal, and only a count above the region's is rejected. |
| Constraint: comparisons joined by `and`, `or`, brackets | S5.3, S6.3 | supported | A constraint whose arithmetic overflows does not hold. |
| Constraint over addressing attributes with `/` or `%` by an operand whose integers alone compute zero | S5.3 | rejected | |
| In-place mark `set_inplace` | S5.5 | supported | |
| Placement marks `set_apart`, `set_no_partial_overlap` | S5.5 | supported | Their constraints follow the constraint rows. |
| State variable `@s.` in a start, size, constraint or semantics | S6.1 | rejected | |
| State update | S5.4 | rejected | Only on an instruction with semantics; one with no semantics is left out. |
| Floor `/` and `%` with the sign of the divisor | S6.2 | supported | |
| Semantics lines, `ROOT`, comments, optional semicolons | S7.1 | supported | A value no later line reads, or `ROOT` above the last line, is rejected. |
| Element types `pred`, `s8` to `s64`, `u8` to `u64`, `f16`, `bf16`, `f32`, `f64` | S7.3 | supported | |
| Element types `s2`, `s4`, `u2`, `u4`, `f8e4m3fn`, `f8e5m2` | S7.3 | rejected | |
| Shape dimension: integer or expression | S7.4 | partial | A dimension whose integers alone pass 64 bits is rejected. |
| Agreement of shapes and element types between lines | S7.4, S8.3 | partial | Not checked; selection matches each value only at the shape and element type its line declares. |
| Operation words `add`, `bitcast_convert`, `broadcast`, `concatenate`, `constant`, `convert`, `copy`, `divide`, `dot`, `exponential`, `or`, `reduce_add`, `reshape`, `shift_left`, `shift_right_logical`, `slice`, `transpose`, `xor` | S8.1 | supported | |
| Every other operation word | S8.1 | rejected | |
| `slice` with a stride other than 1 | S8.2 | rejected | |
| `constant` with an expression in backticks | S7.5 | rejected | |
| `constant` with a literal outside its element type | S7.5 | rejected | |
| Operation attributes of `broadcast`, `concatenate`, `dot`, `reduce_add`, `slice`, `transpose` | S8.2 | supported | A `broadcast` of one element matches any dimensions map. |
| Element types per operation | S8.3 | partial | Not checked; selection matches each value only at the element type its line declares. |
