# Expressions

This page is S6 of the TAIDL language specification.
An expression is integer arithmetic over plain integers and symbolic references, such as `n * 128`, which the tracer records as a tree.

## S6.1 References and where each may appear

An expression stands in six places: a start or a size of a region, a shape dimension, an operation attribute value or `constant` value in a semantics, a constraint, and a state update.

| reference | written | refers to | may appear in |
| --- | --- | --- | --- |
| computational attribute | a parameter annotated `Comp` | that attribute's value at issue | every place |
| addressing attribute | a parameter annotated `Addr` | that attribute's value at issue | a start and a constraint |
| state variable | the object `acc.state` returned | its value before the issue | every place |

- An addressing attribute in any other place is rejected by `check` (S9).
- The target of `update` is a state variable.
- Two recorded expressions are equal when their trees are structurally equal; S5.5 compares expressions this way.

## S6.2 Integer arithmetic

- A plain integer is a value other than a `bool` that `operator.index` accepts; its value is the `int` that call returns, and the `Model` holds that `int`.
- A plain integer is taken wherever this document names a plain `int`; a `bool` or a `float` there is rejected, except `True` and `False` as an `ops.constant` value at `pred` (S7.5).
- A plain integer in an expression lies in signed 64 bits, or `check` rejects it.
- The operators are `+`, `-`, `*`, `//`, `%`, unary `-`, and parentheses, with Python's precedence and left grouping.
- Python folds operators between plain integers before the tracer sees them; only operators with a symbolic operand are recorded.
- Every recorded operator yields a signed 64-bit integer, or the issue faults (S1.3.1).
- A generator evaluates each recorded tree as recorded, one operator at a time; it never rewrites a tree in a way that changes whether an operator leaves signed 64 bits.
- The comparisons that implicit and marker constraints add are evaluated in exact integer arithmetic; their own sums never fault and never wrap.
- `//` is floor division: `floor(a / b)`, so `15 // 2` is 7 and `-7 // 2` is -4, as in Python.
- `%` is the remainder with the sign of the divisor, so `a == (a // b) * b + a % b`, as in Python.
- A `//` or `%` whose divisor is the plain integer 0 raises at the author's line; a symbolic divisor that evaluates to 0 is a fault (S1.3.2).
- There is no `/`, `**`, `not`, `min`, `max`, or conditional value; S3.2 lists the errors each raises.

## S6.3 Conditions

- A comparison is two expressions joined by one of `==`, `!=`, `<`, `<=`, `>`, `>=`; it is a condition.
- `any_of(c1, c2, ...)` holds when one of its conditions holds; `all_of(c1, c2, ...)` when all hold; each takes one or more conditions or nested `any_of` and `all_of`.
- A plain `bool` in `require` or `all_of` drops when `True`; in `any_of` it drops when `False`.
- A plain `bool` that cannot drop is recorded as the constant comparison `0 == 0` or `0 != 0`, so no sibling comparison loses its fault.
- A call written with no argument raises at the author's line; a call emptied by dropping is `True` for `all_of` and `False` for `any_of`.
- A `require` whose condition is `False` from plain values alone is rejected at `check` (S9), and its error names a Python `if`.
- `require(c1, c2, ...)` is `all_of` at the top level.
- Every comparison of a condition is evaluated, left to right, with no short circuit; a fault in any of them is a fault of the issue.
- An example: `require(any_of(x + n <= y, y + 64 <= x))`.
