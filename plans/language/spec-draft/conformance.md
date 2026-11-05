# Conformance

This page is S9 of the spec: every ISA description a conforming front end rejects.
Each row names the call that rejects it and the error it raises.
An error names the instruction and, for a text, its place, line, and column, both counted from 1.

## Declarations

| rejected | call | error |
| --- | --- | --- |
| A name that is not an ASCII identifier, or is a Python keyword | any call that takes a name | `ValueError` |
| A data model, an instruction, or a state variable named twice | `add_data_model`, `add_instruction`, `add_initial_state` | `ValueError` |
| An instruction attribute named twice across both attribute lists | `add_instruction` | `ValueError` |
| Addressed dimensions or a unit shape that is not a list of positive integers | `add_data_model` | `ValueError` |
| An element type outside S7.3 | `add_data_model` | `ValueError` |
| A state variable after any instruction | `add_initial_state` | `ValueError` |
| An initial value that is not an `int` | `add_initial_state` | `ValueError` |
| One string where a list of strings is taken | `add_instruction`, `add_constraints` | `TypeError` |

## Regions and marks

| rejected | call | error |
| --- | --- | --- |
| A region that is not `[data model, starts, sizes]` | `set_inputs`, `set_outputs` | `ValueError` |
| A region of a data model never declared | `set_inputs`, `set_outputs` | `ValueError` |
| Starts or sizes whose count is not the data model's count of addressed dimensions, or 1 for `d0` | `set_inputs`, `set_outputs` | `ValueError` |
| A flag list whose length is not the count of input regions | `set_inplace`, `set_apart`, `set_no_partial_overlap` | `ValueError` |
| An input marked both in place and by a placement mark | `set_inplace`, `set_apart`, `set_no_partial_overlap` | `ValueError` |
| A placement mark on an instruction without exactly one output region | `set_apart`, `set_no_partial_overlap` | `ValueError` |
| A placement mark on an input in another data model than the output | `set_apart`, `set_no_partial_overlap` | `ValueError` |
| `set_apart` on an input whose starts are all spelled as the output's | `set_apart` | `ValueError` |

## Texts

| rejected | call | error |
| --- | --- | --- |
| A text the grammar of S3 does not accept | the call that takes it | `SyntaxError` |
| A reference to an attribute or state variable not declared | the call that takes it | `SyntaxError` |
| An integer literal outside signed 64 bits, in an expression, a shape, or an operation attribute | the call that takes it | `SyntaxError` |
| A `/` or `%` whose right operand is the literal `0` | the call that takes it | `SyntaxError` |

## Semantics

All rows raise `SyntaxError` from `add_semantics`, and the last row also from `set_inputs` and `set_outputs`.

| rejected |
| --- |
| A line whose value name has no leading `%` |
| No line, or more than one, marked `ROOT` |
| `ROOT` on any line but the one defining the output value of the highest k |
| An operand count, or an operand kind, other than the operation's row in S8.2 |
| An operation attribute the operation does not take, given twice, of the wrong form, or required and missing |
| An operand no earlier line defines, or a value two lines define |
| A `parameter(k)` that defines a name other than `%In<k>`, or an `%In<k>` defined by another operation |
| An input or output region without its value, or an `%In<k>` or `%Out<k>` without its region |

## Not rejected by the reference front end

- Shapes and element types are not checked against S8; S1.3 lists this as undefined.
- An addressing attribute in a size, a backtick expression, or a state update is not rejected; S1.3 lists this as undefined.
- A data model declared after an instruction is accepted.
