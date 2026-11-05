# Plan index

Sized for three to four engineers over two quarters; Q1 items run in parallel across the front end, the oracle, and the spec.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| F1 | Typing rules that U4 implements in the front end | L | M | Q1 |
| F2 | Region aliasing rules and in-place mark meaning | L | M | Q1 |
| F3 | Total integer arithmetic: overflow and zero divisors fault | L | S | Q1 |
| F4 | State variable semantics across both generators | L, F3 | L | Q2 |
| F5 | Numeric classes, bounds, reduction identities | L, U, F9 | M | Q2 |
| F6 | Byte order, sub-byte packing, size of d0 | L | S | Q1 |
| F7 | Conformance suite derived from the rules | F1, F2, F3, F5, T, U | M | Q2 |
| F8 | Front-end rejections the spec already implies | L | S | Q1 |
| F9 | Executable reference interpreter of the exact classes | F1, F3, L | M | Q1 |
