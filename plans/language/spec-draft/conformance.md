# Conformance

This page is S9 of the spec: every ISA description a conforming front end rejects.
Each row names the call that rejects it and the error it raises.
An error names the instruction and, for a text, its place, line, and column, both counted from 1.
A conforming front end rejects exactly the ISA descriptions this page lists, and no other.

## Declarations

| rejected | call | error |
| --- | --- | --- |
| A name that is not a `str`, not an ASCII identifier, or a Python keyword | any call that takes a name | `ValueError` |
| A buffer, an instruction, or a state variable named twice | `add_buffer`, `add_instruction`, `add_state` | `ValueError` |
| An instruction attribute named twice across both attribute lists | `add_instruction` | `ValueError` |
| Access dimensions or a unit shape that is not a list or tuple of `int` values, each at least 1 | `add_buffer` | `ValueError` |
| An element type outside S7.3 | `add_buffer` | `ValueError` |
| A buffer or a state variable after any instruction | `add_buffer`, `add_state` | `ValueError` |
| An initial value whose type is not exactly `int`, or outside signed 64 bits | `add_state` | `ValueError` |
| A bare string, or a non-`str` entry, where a list of strings is taken | `add_instruction` | `TypeError` |

## Regions and marks

| rejected | call | error |
| --- | --- | --- |
| A region that is not `[buffer, starts, sizes]` | `set_inputs`, `set_outputs` | `ValueError` |
| A region of a buffer never declared | `set_inputs`, `set_outputs` | `ValueError` |
| Starts or sizes whose count is not the buffer's count of access dimensions | `set_inputs`, `set_outputs` | `ValueError` |
| A start or size that is neither an `int` nor a `str`, or an `int` outside signed 64 bits | `set_inputs`, `set_outputs` | `SyntaxError` |
| A flag list whose length is not the count of input regions, or a flag that is not a `bool` | `set_inplace`, `set_apart`, `set_no_partial_overlap` | `ValueError` |
| More than one in-place flag, or an in-place input whose buffer or sizes differ from output region 0's | `set_inplace` | `ValueError` |
| An input marked both in place and by a placement mark | `set_inplace`, `set_apart`, `set_no_partial_overlap` | `ValueError` |
| A placement mark on an instruction without exactly one output region | `set_apart`, `set_no_partial_overlap` | `ValueError` |
| A placement mark on an input in another buffer than the output | `set_apart`, `set_no_partial_overlap` | `ValueError` |
| `set_apart` on an input whose starts are all spelled as the output's | `set_apart` | `ValueError` |

## Texts

| rejected | call | error |
| --- | --- | --- |
| A text the grammar of S3 does not accept | the call that takes it | `SyntaxError` |
| A reference to an attribute or state variable not declared | the call that takes it | `SyntaxError` |
| An addressing attribute in a size, a backtick expression, or a state update | the call that takes it | `SyntaxError` |
| An integer literal above 2^63 - 1, in an expression, a shape, or an operation attribute | the call that takes it | `SyntaxError` |
| A `/` or `%` whose right operand is exactly the token `0` | the call that takes it | `SyntaxError` |

## Semantics

All rows raise `SyntaxError` from `add_semantics`, and the last two rows also from `set_inputs` and `set_outputs`.

| rejected |
| --- |
| A line whose value name has no leading `%` |
| A semantics with no `%Out0` line |
| No line, or more than one, marked `ROOT` |
| `ROOT` on any line but the one defining the output value of the highest k |
| An operand count, or an operand kind, other than the operation's row in S8.2 |
| An operation attribute the operation does not take, given twice, of the wrong form, or required and missing |
| An operand no earlier line defines, or a value two lines define |
| A `parameter(k)` that defines a name other than `%In<k>`, or an `%In<k>` defined by another operation |
| A literal shape dimension of 0 |
| A `constant` literal the line's element type cannot hold (S7.5) |
| An input or output region without its value, or an `%In<k>` or `%Out<k>` without its region, including `%In01` |
| An instruction with regions but no semantics, or with semantics but no regions |

## Rows the reference front end does not yet check

Each row below is part of S9; the plan item that makes the front end check it is beside it.

| row | plan item |
| --- | --- |
| A buffer after any instruction | L11 |
| An addressing attribute outside a start or a constraint | L10 |
| An initial value outside signed 64 bits, or of type `bool` | L16 |
| A `constant` literal the element type cannot hold | L16 |
| A literal shape dimension of 0 | L16 |
| A non-`str` list entry, or a flag that is not a `bool` | L16 |
| An in-place input whose buffer or sizes differ from the output's | L5 |
| An instruction with regions but no semantics, or the reverse | L11 |
| Shapes and element types against S7.2 and S8 | L12 |
