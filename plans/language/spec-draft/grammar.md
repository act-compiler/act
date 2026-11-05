# Surface

This page is S2 and S3 of the TAIDL language specification: the Python surface an ISA specification is written in, and how the front end reads it.
An ISA specification is a Python file; the front end runs it and records what it declares, and never reads its source text.
The surface needs the Python standard library and `taidl` only.

## S2 The Python surface

### S2.1 Names a specification imports

| name | from | what it is |
| --- | --- | --- |
| `Accelerator` | `taidl` | the accelerator an ISA specification builds (S4.2) |
| `Comp`, `Addr` | `taidl` | the kind of an instruction attribute, as a parameter annotation (S5.1) |
| `read`, `write` | `taidl` | read an input region, write an output region (S5.2) |
| `require`, `any_of`, `all_of` | `taidl` | constraints and their nesting (S5.3, S6.3) |
| `update` | `taidl` | one state update (S5.4) |
| `ops` | `taidl` | one function per tensor operator of S8, as `ops.<word>` |
| `check` | `taidl` | the check function (S9) |
| `u1` to `u63`, `s1` to `s64` | `taidl.types` | attribute types of the width layer (S11.2) |
| element types | `taidl.types` | the 19 element types of S7.3, one object each, such as `bf16` |

- A type object of one name serves both as an element type and as an attribute type where both lists hold it, such as `u8`.
- A misspelled name is a Python `NameError` at the author's line.

### S2.2 Operators on symbolic values

A symbolic value stands for something known only at issue: an attribute, a state variable, or a region value.

| value | operators | meaning |
| --- | --- | --- |
| integer expression | `+`, `-`, `*`, `//`, `%`, unary `-`, parentheses | 64-bit integer arithmetic (S6.2) |
| integer expression | `==`, `!=`, `<`, `<=`, `>`, `>=` | one comparison, a condition (S6.3) |
| tensor value | `+`, `-`, `*`, unary `-` | `ops.add`, `ops.subtract`, `ops.multiply`, `ops.negate` |

- Python precedence and left grouping apply; the tracer records each operator as Python evaluates it.
- A plain `int` mixes with an integer expression; a plain number never mixes with a tensor value, which takes `ops.constant`.
- A comparison hashes by identity, so it can sit in a set or a dict key.

## S3 Tracing

### S3.1 Instruction functions and families

- An instruction is a function decorated by `@acc.instruction`; the decorator runs it once, at definition, on symbolic attributes.
- That run records every `read`, `write`, `require`, `update`, and operation in call order; the record is the instruction (S5).
- The function returns nothing; a returned value is rejected.
- Python control flow on plain values runs at definition, so a loop or an `if` over plain values builds an instruction family.
- `@acc.instruction(name=...)` names one member of a family; without it the instruction takes the function's name.

```python
for n in (1, 2, 4):
    @acc.instruction(name=f'copy_{n}')
    def copy(src: Addr, dst: Addr):
        write(d1.region([dst], size=[n]), read(d1.region([src], size=[n])))
```

### S3.2 What tracing rejects

Each case below raises at the author's line, at once, and names the spelling to use.

| construct on a symbolic value | spelling to use |
| --- | --- |
| `/` on an integer expression | `//` |
| `**` | repeated `*` |
| `bool()`, so `if`, `while`, `not`, `and`, `or`, and a chained comparison | `require`, `any_of`, `all_of`, or `ops.select` |
| `&` or `|` on a condition | `all_of` or `any_of` |
| `min` or `max` | `ops.minimum`, `ops.maximum` on tensors; two constraints on integers |
| `/`, `//`, or `%` on a tensor value | `ops.divide`, `ops.remainder` |
| a slice `buffer[a:b]` | `buffer.region(starts, size=sizes)` |
| `int()`, `len()`, or indexing that needs a concrete value | a plain value computed outside the instruction |

- A trace-time error is a Python exception in the author's frame; it stops the run, unlike the problems `check` collects (S9).
