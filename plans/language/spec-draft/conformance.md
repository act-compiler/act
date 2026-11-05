# Conformance

This page is S9 of the TAIDL language specification: the check function, every ISA specification a conforming front end rejects, and the checks a generator adds after it.
A conforming front end rejects exactly the ISA specifications this page lists and those whose tracing raises under S3.2, and no other.

## Failure policy

- Nothing is rejected without a genuine cause: every row is a problem the architect can correct in the ISA specification.
- Each error names the clause it breaks by section id, and the instruction, buffer, attribute, or namespace at fault.
- The front end decides a case only from plain integers and from the structural equality of recorded expressions (S6.1); it applies no other algebra, and every other case is checked at issue as a fault (S1.3).
- A generator rejects only what Generator checks below lists; it excludes an instruction it cannot model.

## The check function

- `taidl.check(acc)` runs every rule of this page on an accelerator, freezes it, and returns a `Model` (S4.6).
- It needs no generator, and `python -m taidl check spec.py` runs it on a file alone.
- It collects every problem before it raises, so one run reports them all.
- A signature problem (S5.1) is reported once, and the function body is not run.
- Any other collected problem is recorded at its call, which returns a placeholder of its result kind.
- A call with a placeholder argument checks nothing, records nothing, and returns a placeholder; `bool()` on a placeholder raises at once (S3.2).
- It raises one `Rejection` of entries, ordered by declaration, then by call order within an instruction.
- Each entry carries a code, the object at fault (instruction, buffer, state variable, attribute, or namespace), its section ids, and the author's file and line.
- An entry's text is the two lines of diagnostics/scheme.md: line 1 ends with the row's section column as `[S<id>]`, and line 2 gives `file:line`.
- A trace-time error of S3.2 is a `taidl.TraceError` at the author's line and stops at once; `check` does not collect it.
- `check` may run any number of times on one accelerator, and every run that returns gives an equal `Model`.
- After the first run that returns, a declaration on that accelerator raises at once (S4.5); a run that raises freezes nothing.
- `check` lists the metadata namespaces present as `Model.metadata_namespaces`, the sorted tuple of the namespaces on any node.
- `python -m taidl check` prints `note: metadata namespaces: <list>` from `Model.metadata_namespaces`; a note is not an entry of `Rejection` and does not change the exit code.
- A lint, such as the S5.5 warning, is a `taidl.TaidlWarning` issued by `warnings.warn_explicit` at the author's file and line, only when `check` finds no problem; the `Model` holds no lint.
- The command runs the file by `runpy.run_path` with its default `__name__`, so a `__main__` block stays inert.
- A collector active for that run checks every `Accelerator` constructed, in construction order.

| exit code | meaning |
| --- | --- |
| 0 | every check returned |
| 1 | a check raised a `Rejection` |
| 2 | the file raised |
| 3 | internal error |
| 4 | usage error |
| 5 | no `Accelerator` was constructed |

- A generator entry takes a `Model`, or an `Accelerator` on which it calls `check` once; its body reads only the `Model`.

## Declarations

| rejected | section |
| --- | --- |
| A name that is not a `str`, not an ASCII identifier, or a Python keyword | S4.1 |
| A buffer, an instruction, or a state variable named twice, `d0` included | S4.1 |
| Access dimensions or a base element shape that is not a list or tuple of plain integers (S6.2), each at least 1 | S4.3 |
| An `element_type` outside S7.3 | S4.3 |
| An `initial_value` that is not a plain literal its element type can hold (S7.5) | S4.3 |
| A buffer or a state variable declared after any instruction | S4.5 |
| A state value that is not a plain integer (S6.2), or outside signed 64 bits | S4.4 |
| A decorated object that is not a plain `def` function, or that is a generator function or a coroutine function | S3.1 |
| A parameter with no `Comp` or `Addr` annotation, an annotation that fails to evaluate, a default, `*args` or `**kwargs`, or a `/` or `*` separator | S5.1 |
| An instruction function that returns a value | S3.1 |

## Regions and markers

| rejected | section |
| --- | --- |
| A region whose start or size sequence is not a list or tuple, or whose length is not the buffer's count of access dimensions | S5.2 |
| A start or a size that is neither a plain integer nor an integer expression, or a plain integer outside signed 64 bits | S5.2 |
| A plain start below 0 or a plain size below 1 | S5.2 |
| A plain start plus a plain size past a plain access dimension, `d0` excepted | S5.2 |
| `read` given something that is not a region | S5.2 |
| An instruction with a `read` and no `write` | S5.2 |
| `write` given something that is not a tensor value | S7.2 |
| A marker that is not one read value for `inplace` or a list of read values for the others | S5.5 |
| A marker naming a value that is not a `read` result of this instruction | S5.5 |
| An in-place input in another buffer, or whose size expressions differ from the output's | S5.5 |
| A placement marker on an input in another buffer than the output | S5.5 |
| An input named by more than one marker of one `write`, or two inputs tied to one output | S5.5 |
| An input tied to more than one output | S5.5 |
| Two output regions of one buffer whose start expressions all equal | S5.5 |
| An exact-or-apart pair whose start expressions all equal and whose sizes differ as two plain integers in some dimension | S5.5 |
| `apart` on an input whose start expressions all equal the output's, since `apart` then never holds | S5.5 |
| An input whose start and size expressions all equal the output's, in its buffer, without `inplace` | S5.5 |

## Expressions, constraints, and updates

| rejected | section |
| --- | --- |
| An addressing attribute in a size, a shape, an operation attribute, a `constant`, or a state update | S6.1 |
| A plain integer outside signed 64 bits in an expression | S6.2 |
| `update` of something that is not a state variable, or two updates of one state variable | S5.4 |
| `require`, `any_of`, or `all_of` given something that is neither a condition nor a plain `bool` | S6.3 |
| A `require` whose condition is `False` from plain values alone | S6.3 |

## Semantics

| rejected | section |
| --- | --- |
| An operand count, or an operand kind, other than the operation's row in S8.2 | S8.2 |
| An operation attribute the operation does not take, of the wrong form, or required and missing | S8.2 |
| An operation attribute that names an axis and is not a plain integer | S8.2 |
| A symbolic object made in another trace, other than a region whose starts and sizes are all plain | S7.2 |
| A plain shape dimension below 1, written or inferred | S7.4 |
| A plain `constant` value the element type cannot hold, or an infinity at a type with none | S7.5 |
| A target element type outside S7.3 in an `ops` call | S7.3 |
| A written value whose element type or rank differs from its region's, or which differs in a dimension where both are plain integers | S7.2 |
| An operation whose operand element types or ranks break S8, or whose operand dimensions break S8 where they are plain integers | S8.4 |
| An element type outside a word's row of S8.3 | S8.3 |

## Removed constructs

- A removed construct raises at its call, at once, as a trace-time error does, and names the construct, its replacement, and the section id.
- A `str` where v3 takes a type object, an expression, or a condition is the removed string form, and its message names the Python spelling.
- One removed-names table drives that error: `add_semantics` and the other v2 string-surface calls, the tensor operators `cholesky`, `triangular_solve`, and `batch_norm_inference`, and each renamed public name of the v3 rename table.
- Each removed name points to its S2 spelling: `add_data_model` and `add_buffer` to `acc.buffer`, `add_initial_state` and `add_state` to `acc.state`.
- The error for `add_semantics` names the converter script of the taidl 2.x line as the route to the Python surface.

## Layers and generator metadata

| rejected | section |
| --- | --- |
| A type outside `s1` to `s64` and `u1` to `u63` in `Comp[t]` or `Addr[t]` | S11.2 |
| An `instr` that is not an instruction of `acc`, a second block for one instruction, or an `acc.encode` call before its instruction is declared | S11.3 |
| A `word` that is not an `int` of at least 1, or that differs from another block's `word`; `word` has no upper limit | S11.3 |
| A slice or fixed range with `high < low`, or with a bit at or above `word` | S11.3 |
| A fixed value outside `0` to `2^(high-low+1) - 1` | S11.3 |
| `fixed` or `fields` of another shape, or a field with no slice | S11.3 |
| An encoding whose bits are not each covered exactly once by a slice or a fixed bit | S11.3 |
| Two instructions whose fixed bits agree on every bit position both fix | S11.3 |
| A field key naming no parameter of the instruction | S11.3 |
| An encoded attribute with no declared type, or whose type width differs from its slice total | S11.3 |
| A derived attribute with no link or two links, a link whose fold leaves signed 64 bits or gives `k` = 0, or a link naming a derived attribute or two encoded ones | S11.4 |
| A namespace that is not an identifier, a `meta` that is not a dict, or a metadata entry that is not a `str`-keyed dict of plain data (S12.1) | S12.2 |

## Generator checks

- A generator runs its own checks after `check`, as plain functions on the `Model`; the oracle adds none.
- It rejects a key its schema does not hold, or a value of the wrong form, in its own namespace (S12.3).
- It repeats no check of `check`, the encoding-block checks of S11.3 included.
- It rejects an ISA specification none of whose instructions it can use.
- Beside its checks, it may reject at emission a name it reserves (S4.1), before it writes any file.
- It excludes an instruction it cannot model, with a reason: a code, a section id, and the author's line.
- It prints each exclusion at generation in the two-line warning form, stores the list in its output, and returns it in its report (S12.4); an excluded instruction has no other meaning there.
- The compiler backend excludes an instruction that reads or updates state, has other than one output region, has a constraint its allocator cannot hold, or uses a tensor operator outside its supported table.
- A type constraint of a computational attribute is never a reason to exclude an instruction (S11.2).
- The compiler backend rejects at kernel compile time, with class `K` and the kernel as the object, a kernel for which extraction finds no covering.
- That error names the lowest uncovered operation and the stop cause, then lists, as a hint outside the match key, each excluded instruction whose tensor operators include that operation.
