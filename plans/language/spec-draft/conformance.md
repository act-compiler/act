# Conformance

This page is S9 of the TAIDL language specification: the check function, every ISA specification a conforming front end rejects, and the checks a generator adds after it.
A conforming front end rejects exactly the ISA specifications this page lists, and no other.

## Failure policy

- Nothing is rejected without a genuine cause: every row is a problem the architect can correct in the ISA specification.
- Each error names the clause it breaks by section id, and the instruction, buffer, attribute, or namespace at fault.
- A case the front end cannot decide from the recorded expressions is not rejected; it is checked at issue as a fault (S1.3).
- A generator rejects only an invalid entry in its own metadata namespace (S12.3), a name it reserves (S4.1), fixed bits that two instructions share (S11.3), or an ISA specification with no instruction it can use; it excludes an instruction it cannot model (below).

## The check function

- `taidl.check(acc)` runs every rule of this page on an accelerator, freezes it, and returns a `Model` (S4.6).
- It needs no generator, and `python -m taidl check spec.py` runs it on a file alone.
- It collects every problem before it raises, so one run reports them all.
- It raises one `Rejection` whose entries each carry a code, a section id, the instruction, and the author's file and line.
- An entry's line 1 names the file, the line, and the instruction and ends with `[S<id>]`; line 3 lists further sections.
- A trace-time error of S3.2 is a Python exception at the author's line and stops at once; `check` does not collect it.
- A generator entry takes a `Model`, or an `Accelerator` on which it calls `check` once; its body reads only the `Model`.

## Declarations

| rejected | section |
| --- | --- |
| A name that is not a `str`, not an ASCII identifier, or a Python keyword | S4.1 |
| A buffer, an instruction, or a state variable named twice | S4.1 |
| Access dimensions or a base element shape that is not a list or tuple of `int` values, each at least 1 | S4.3 |
| An `element_type` outside S7.3 | S4.3 |
| A buffer or a state variable declared after any instruction | S4.5 |
| A state value whose type is not exactly `int`, or outside signed 64 bits | S4.4 |
| A parameter with no `Comp` or `Addr` annotation, a default, or `*args` or `**kwargs` | S5.1 |
| An instruction function that returns a value | S3.1 |

## Regions and markers

| rejected | section |
| --- | --- |
| A region whose start list or size list is not a list, or whose length is not the buffer's count of access dimensions | S5.2 |
| A start or a size that is neither an `int` nor an integer expression, or an `int` outside signed 64 bits | S5.2 |
| A marker naming a value that is not a `read` result of this instruction | S5.5 |
| An in-place input in another buffer, or whose size expressions differ from the output's | S5.5 |
| A placement marker on an input in another buffer than the output | S5.5 |
| An input named by more than one marker of one `write`, or two inputs tied to one output | S5.5 |
| `apart` on an input whose start and size expressions all equal the output's | S5.5 |
| An input whose start and size expressions all equal the output's, in its buffer, without `inplace` | S5.5 |

## Expressions, constraints, and updates

| rejected | section |
| --- | --- |
| An addressing attribute in a size, a shape, an operation attribute, a `constant`, or a state update | S6.1 |
| A plain integer outside signed 64 bits | S6.2 |
| `update` of something that is not a state variable, or two updates of one state variable | S5.4 |
| `require`, `any_of`, or `all_of` given something that is not a condition | S6.3 |

## Semantics

| rejected | section |
| --- | --- |
| An operand count, or an operand kind, other than the operation's row in S8.2 | S8.2 |
| An operation attribute the operation does not take, of the wrong form, or required and missing | S8.2 |
| A value from another instruction's function | S7.2 |
| A plain shape dimension of 0 | S7.4 |
| A plain `constant` value the element type cannot hold, or an infinity at a type with none | S7.5 |
| A written value whose shape or element type differs from its region's, when both are plain | S7.2 |
| An operation whose operand shapes or types break S8, when they are plain | S8.4 |

## Removed constructs

- A construct removed in v3 raises one coded rejection that names the construct, its replacement, and the section id.
- One removed-names table drives that rejection: `add_semantics` and the other v2 string-surface calls, the tensor operators `cholesky`, `triangular_solve`, and `batch_norm_inference`, and each renamed public name of the v3 rename table.
- The rejection of `add_semantics` names the converter script of the taidl 2.x line as the route to the Python surface.

## Layers and generator metadata

| rejected | section |
| --- | --- |
| A type outside `s1` to `s64` and `u1` to `u63` in `Comp[t]` or `Addr[t]` | S11.2 |
| An encoding whose bits are not each covered exactly once by a slice or a fixed bit | S11.3 |
| A field key naming no parameter of the instruction | S11.3 |
| An encoded attribute with no declared type, or whose type width differs from its slice total | S11.3 |
| Two instructions whose fixed bits one word can match, rejected by the generator that reads the blocks | S11.3 |
| A derived attribute with no link, two links, a link not of the form of S11.4, or a link naming a derived attribute or two encoded ones | S11.4 |
| A namespace that is not an identifier, or a metadata entry that is not plain data | S12.2 |

## Generator checks

- A generator runs its own checks after `check`, as plain functions on the `Model`; the oracle adds none.
- It rejects an unknown key in its own metadata namespace (S12.3) and an ISA specification it can use no instruction of.
- It excludes an instruction it cannot model, with a reason: a code, a section id, and the author's line.
- It prints each exclusion at generation and stores the list in its output; an excluded instruction has no other meaning there.
- The compiler backend excludes an instruction that reads or updates state, has other than one output region, has a size over an addressing attribute or state, has a constraint its allocator cannot hold, or uses a tensor operator outside its supported table.
- The compiler backend rejects a kernel that needs an excluded instruction at kernel compile time, naming the instruction.

## Rows the reference front end does not yet check

Each row is part of S9; the plan item that makes `check` hold it is beside it.

| row | plan item |
| --- | --- |
| Every row, collected into one `Rejection` from a traced specification | L42, L43 |
| A buffer after any instruction | L11 |
| An addressing attribute outside a start or a constraint | L10 |
| A state value outside signed 64 bits, or of type `bool` | L16 |
| A plain `constant` value the element type cannot hold, or a plain shape dimension of 0 | L16 |
| An in-place input whose buffer or size expressions differ from the output's | L5 |
| `apart` on equal expressions, or equal input and output expressions without `inplace` | L36 |
| A removed construct, by the removed-names table | L33, L45 |
| Shapes and element types against S7.2 and S8 | U4 |
| Every row of Layers and generator metadata | L37, L38, L39, L40 |
| Generator checks, with exclusion | L43 |
