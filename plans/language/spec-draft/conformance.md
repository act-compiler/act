# Conformance

This page is S9 of the spec: every ISA specification a conforming front end rejects.
Each row names the call that rejects it and the error it raises.
An error names the instruction and, for a text, its place, line, and column, both counted from 1.
A conforming front end rejects exactly the ISA specifications this page lists, and no other.

## Failure policy

- Nothing is rejected without a genuine cause: every row is a problem the architect can correct in the ISA specification.
- Each error names the clause it breaks by section id, and the instruction, buffer, attribute, or namespace at fault.
- A case the front end cannot decide from the text is not rejected; it is checked at issue as a fault (S1.3).
- A generator rejects only an unsupported feature (S1.2) or an invalid entry in its own metadata namespace (S12.3), with the same kind of error.

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
| A flag list whose length is not the count of input regions, or a flag that is not a `bool` | every mark | `ValueError` |
| More than one in-place flag, or an in-place input whose buffer or size text differs from output region 0's | `set_inplace` | `ValueError` |
| An input carrying more than one mark | every mark | `ValueError` |
| A mark on an instruction without exactly one output region | every mark | `ValueError` |
| A placement mark on an input in another buffer than the output | `set_apart`, `set_no_partial_overlap`, `set_may_overlap` | `ValueError` |
| `set_apart` on an input whose start and size texts all match the output's | `set_apart` | `ValueError` |
| An input whose start and size texts all match the output's, in its buffer, without `set_inplace` | the instruction's declaration | `ValueError` |
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
| A region value line whose shape differs from its region's, with no explicit `reshape` line |
| A float literal that rounds to an infinity, or to NaN at `f8e4m3fn` |

## Removed constructs

- A construct removed in v3 raises one coded rejection that names the construct, its replacement, and the section id.
- One removed-names table drives that rejection: the operation words `cholesky`, `triangular_solve`, and `batch_norm_inference`, and each renamed public name of the v3 rename table.
- The marks `set_inplace`, `set_apart`, and `set_no_partial_overlap` are not removed; `set_may_overlap` is added (S5.5).

## Layers and generator metadata

| rejected | call | error |
| --- | --- | --- |
| A type key naming no attribute of the instruction, or a type outside `s1` to `s64` and `u1` to `u63` | `add_instruction` | `ValueError` |
| An encoding block whose bits are not each covered exactly once by a slice or a fixed bit | the encoding declaration | `ValueError` |
| An encoded attribute with no declared type, or whose type width differs from its slice total | the encoding declaration | `ValueError` |
| Two instructions whose fixed bits one word can match | the generator that reads the blocks | `ValueError` |
| A derived attribute with no link, two links, a link not of the form of S11.4, or a link naming a derived attribute or two encoded ones | the encoding declaration | `ValueError` |
| A namespace that is not an identifier, or a metadata entry that is not plain data | any call that takes metadata | `ValueError` |

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
| An in-place input whose buffer or size text differs from the output's | L5 |
| `set_apart` on identical text, or identical input and output text without `set_inplace` | L36 |
| A removed construct, by the removed-names table | L33 |
| An instruction with regions but no semantics, or the reverse | L11 |
| Shapes and element types against S7.2 and S8 | U4 |
| Every row of Layers and generator metadata | L37, L38, L39, L40 |
