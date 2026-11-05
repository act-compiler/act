# Expressions

This page is S6 of the spec.
An expression is integer arithmetic over integers and references, such as `` `@c.n * 128` ``.

## S6.1 References and where each may appear

An expression stands in four places: a start or a size of a region, a backtick expression in a semantics, a constraint, and a state update.
A semantics refers to attributes and state only inside backticks.

| reference | written | refers to | may appear in |
| --- | --- | --- | --- |
| computational attribute | `@c.<name>` | an entry of `computation_attr` | every place |
| addressing attribute | `@a.<name>` | an entry of `addressing_attr` | a start and a constraint |
| state variable | `@s.<name>` | a state variable of `add_initial_state` | every place |

- A reference names an attribute of its own instruction or a declared state variable; any other name is rejected.
- An addressing attribute in any other place is undefined; the front end does not yet reject it (S9).
- The target of a state update is a state variable.

## S6.2 Integer arithmetic

- An integer literal is decimal, unsigned, and has no leading zero.
- The operators are `+`, `-`, `*`, `/`, `%`, unary `-`, and parentheses, with precedence as S3.2 gives.
- Every integer an expression reads or computes is a signed 64-bit integer.
- `/` is floor division: `floor(a / b)`, so `15 / 2` is 7 and `-7 / 2` is -4.
- `%` is the remainder with the sign of the divisor, so `a == (a / b) * b + a % b`.
- A literal outside signed 64 bits, and a `/` or `%` whose right operand is the literal `0`, are rejected.
- A computed divisor of 0 is undefined (S1.3).
- A computed integer outside signed 64 bits is implementation-defined (S1.3).
- TAIDL-TO computes such an integer exactly and warns.
- ACT-Backend leaves an instruction unselected where an operand's size overflows.
- A constraint whose arithmetic overflows does not hold.
- There is no `//`, `**`, `not`, function such as `min`, or conditional value, and an update has no `+=`.

## S6.3 Conditions

- A condition is one or more comparisons joined by `and` and `or`, with parentheses, as S3.2 gives.
- A comparison is two expressions joined by one of `==`, `!=`, `<`, `<=`, `>`, `>=`.
- An example: `@a.x + @c.n <= @a.y or @a.y + 64 <= @a.x`.

## Parsed trees

- The front end parses each text when the call that takes it runs, and keeps its tree beside the text.
- `constraint_asts` and `update_asts` hold one tree per constraint and per update.
- `instr_input_asts` and `instr_output_asts` hold each region with one tree per start and per size.
- `instr_expression_asts` maps each backtick expression's node in `instr_semantics_ast` to its tree.
