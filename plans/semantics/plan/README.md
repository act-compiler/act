# Plan index

Sized for three to four engineers over two quarters; Q1 items run in parallel across the front end, the oracle, and the spec.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| F1 | [Typing rules that U4 implements in the front end](F1.md) | L | M | Q1 |
| F2 | [Region aliasing rules and in-place mark meaning](F2.md) | L | M | Q1 |
| F3 | [Total integer arithmetic: overflow and zero divisors fault](F3.md) | L | S | Q1 |
| F4 | [State variable semantics across both generators](F4.md) | L, F3 | L | Q2 |
| F5 | [Numeric classes, bounds, reduction identities](F5.md) | L, U, F6 | M | Q2 |
| F6 | [Byte order, sub-byte packing, size of d0](F6.md) | L | S | Q1 |
| F7 | [Conformance suite derived from the rules: cut, carried by T7, T8, T26, T29](F7.md) | -- | -- | -- |
| F8 | [Front-end rejections the spec already implies](F8.md) | L | S | Q1 |
| F9 | [Executable reference interpreter of the exact classes: cut, carried by T8](F9.md) | -- | -- | -- |

## Owners

- Each live item states a rule that one language item owns; the F item supplies the rule text, the evidence, and the holes it closes.
- F1 is owned by L3 and U4; F2 by L5, L35, and L36; F3 by L6; F4 by L32 with B and O; F5 by L31 and L32; F6 by L32; F8 by L10, L11, and L16.
- F5 follows F6, since `bitcast_convert` and sub-byte layout fix the values the numeric classes name.
- F7 and F9 are cut: the corpus items carry the conformance suite, and reference values come from NumPy or closed forms (Q6.6, owner decision 28), so no interpreter is built.
