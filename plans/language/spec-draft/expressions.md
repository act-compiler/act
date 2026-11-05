# Expressions

This page is S6 of the spec.
An expression is integer arithmetic over integers and references, such as `` `@c.n * 128` ``.

## S6.1 References and where each may appear

An expression stands in four places: a start or a size of a region, a backtick expression in a semantics, a constraint, and a state update.
A semantics refers to attributes and state only inside backticks.

| reference | written | refers to | may appear in |
| --- | --- | --- | --- |
| computational attribute | `@c.<name>` | an entry of `computational_attrs` | every place |
| addressing attribute | `@a.<name>` | an entry of `addressing_attrs` | a start and a constraint |
| state variable | `@s.<name>` | a state variable of `add_state` | every place |

- A reference names an attribute of its own instruction or a declared state variable; any other name is rejected.
- An addressing attribute in any other place is rejected with `SyntaxError` (S9).
- The target of a state update is a state variable.

## S6.2 Integer arithmetic

- An integer literal is decimal and unsigned; it is `0` or has no leading zero.
- An integer literal is at most 2^63 - 1, so the least signed 64-bit value is written `-9223372036854775807 - 1`.
- The operators are `+`, `-`, `*`, `/`, `%`, unary `-`, and parentheses, with precedence as the expression grammar gives (S3).
- Every operator of an expression yields a signed 64-bit integer, or the issue faults (S1.3.1).
- Operators group by the grammar's left-associative tree, and the check walks that tree as written.
- `/` is floor division: `floor(a / b)`, so `15 / 2` is 7 and `-7 / 2` is -4.
- `%` is the remainder with the sign of the divisor, so `a == (a / b) * b + a % b`.
- A `/` or `%` whose right operand is exactly the token `0` is rejected; `(0)`, `-0`, and a computed 0 are faults (S1.3.2).
- A start, size, shape dimension, operation attribute value, comparison operand, or state update with an operator whose value lies outside signed 64 bits is a fault (S1.3.1).
- There is no `//`, `**`, `not`, function such as `min`, or conditional value, and an update has no `+=`.

## S6.3 Conditions

- A condition is one or more comparisons joined by `and` and `or`, with parentheses, as the expression grammar gives.
- A comparison is two expressions joined by one of `==`, `!=`, `<`, `<=`, `>`, `>=`.
- Every comparison of a condition is evaluated, left to right, with no short circuit; a fault in any of them is a fault of the issue.
- An example: `@a.x + @c.n <= @a.y or @a.y + 64 <= @a.x`.
