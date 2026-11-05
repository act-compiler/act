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
| `Model`, `Rejection`, `TraceError`, `TaidlWarning`, `CONTRACT_VERSION` | `taidl` | the checked model (S4.6), the exception `check` raises (S9), the trace-time error (S3.2), the lint warning (S9), and the contract version (S4.6) |
| `u1` to `u63`, `s1` to `s64` | `taidl.types` | attribute types of the width layer (S11.2) |
| element types | `taidl.types` | the 21 element types of S7.3, one object each, such as `bf16` |

- A type object of one name serves both as an element type and as an attribute type where both lists hold it, such as `u8`.
- A misspelled name is a Python `NameError` at the author's line.
- An unknown or missing keyword of `Accelerator`, `acc.buffer`, `acc.state`, `@acc.instruction`, `buffer.region`, `write`, `update`, or `acc.encode` is a Python `TypeError` at once.
- An unknown operation attribute of an `ops` call is the collected S8.2 rejection of S9.

### S2.2 Operators on symbolic values

A symbolic value stands for something known only at issue.
It is an integer expression (an attribute, a state variable, or arithmetic over them), a condition, or a tensor value (a region value, or a value an `ops` call or a tensor operator records).

| value | operators | meaning |
| --- | --- | --- |
| integer expression | `+`, `-`, `*`, `//`, `%`, unary `-`, unary `+`, parentheses | 64-bit integer arithmetic (S6.2); unary `+` is the identity |
| integer expression | `==`, `!=`, `<`, `<=`, `>`, `>=` | one comparison, a condition (S6.3) |
| tensor value | `+`, `-`, `*`, unary `-` | `ops.add`, `ops.subtract`, `ops.multiply`, `ops.negate` |

- Python precedence and left grouping apply; the tracer records each operator as Python evaluates it.
- A plain integer (S6.2) mixes with an integer expression.
- A plain number never mixes with a tensor value, which takes `ops.constant`.
- An integer expression never mixes with a tensor value; write `ops.constant(n)`.
- Every symbolic value hashes by identity, so it can sit in a set or a dict key.
- A membership test on a sequence of symbolic values raises as `bool()` does (S3.2).
- Python objects carry bare names; the S4.6 printer and every message spell a reference `@s.<name>` for a state variable, `@a.<name>` for an addressing attribute, and `@c.<name>` for a computational one.

## S3 Tracing

### S3.1 Instruction functions and families

- An instruction is a function decorated by `@acc.instruction`; the decorator runs it once, at definition, on symbolic attributes.
- `@acc.instruction(...)` declares an instruction and returns its handle (S5.1); bare `@acc.instruction` means `@acc.instruction()`.
- Calling a handle raises a `TraceError` at once, naming S3.1.
- A recording call (`read`, `write`, `require`, `any_of`, `all_of`, `update`, or an `ops` call) outside an instruction function raises at once, naming S3.1.
- A region whose starts and sizes are all plain belongs to no trace and may serve any instruction; every other symbolic object belongs to the trace that made it, and its use in another trace is rejected (S7.2).
- The decorated object is a plain `def` function that is neither a generator function nor a coroutine function; any other object is rejected (S3.1).
- The decorator evaluates each annotation once, at decoration, in the function's globals and closure (`annotationlib.get_annotations(f, format=Format.VALUE)`); an annotation that fails to evaluate is rejected (S5.1).
- That run records every `read`, `write`, `require`, `update`, and operation in call order; the record is the instruction (S5).
- The function returns nothing; a returned value is rejected.
- Python control flow on plain values runs at definition, so a loop or an `if` over plain values builds an instruction family.
- `@acc.instruction(name=...)` names one member of a family; without it the instruction takes the function's name.
- A family without `name=` gives one name twice and is rejected (S4.1).

```python
for n in (1, 2, 4):
    @acc.instruction(name=f'copy_{n}')
    def copy(src: Addr, dst: Addr):
        write(d1.region([dst], size=[n]), read(d1.region([src], size=[n])))
```

### S3.2 What tracing rejects

Each case below raises at the author's line, at once, and names the spelling to use.

| construct | spelling to use |
| --- | --- |
| `/` on an integer expression | `//` |
| `**` | repeated `*` |
| `bool()`, so `if`, `while`, `not`, `and`, `or`, `in`, a chained comparison, `min`, and `max` | `require`, `any_of`, `all_of`, `ops.select`, `ops.minimum`, `ops.maximum`; the message lists every spelling of the row, since the tracer sees only `bool()` |
| `&` or `\|` on a condition | `all_of` or `any_of` |
| `/`, `//`, or `%` on a tensor value | `ops.divide`, `ops.remainder` |
| `==`, `!=`, `<`, `<=`, `>`, `>=`, `@`, `~`, `abs()`, indexing, iteration, or `bool()` on a tensor value | `ops.compare`, `ops.dot`, `ops.not_`, `ops.abs`, `ops.slice`, or `ops.select` |
| a plain number with `+`, `-`, or `*` on a tensor value | `ops.constant(value, element_type, shape)` |
| `//` or `%` by the plain integer 0 | a nonzero divisor |
| a slice `buffer[a:b]` | `buffer.region(starts, size=sizes)` |
| `int()`, `len()`, or indexing that needs a concrete value | a plain value computed outside the instruction |
| a recording call outside an instruction function (S3.1) | call it inside the function |

- Any other operator, builtin, or protocol (`__index__`, `__int__`, `__float__`, `__len__`, `__iter__`) on a symbolic value raises in the same way, naming S6.2.
- A trace-time error is a `taidl.TraceError`, a subclass of `TypeError`, whose text is one `R` message of diagnostics/scheme.md ending `[S<id>]`.
- It is raised in the author's frame and stops the run, unlike the problems `check` collects (S9); the instruction being traced is not added to the accelerator.
