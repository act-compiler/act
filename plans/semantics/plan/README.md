# Plan index

Sized for three to four engineers over two quarters; Q1 items run in parallel across the front end, the oracle, and the spec.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| F1 | Static shape and element type checker in the front end | L | L | Q1 |
| F2 | Region aliasing rules and in-place mark meaning | L | M | Q1 |
| F3 | Total integer arithmetic: overflow and zero divisors reject | L | S | Q1 |
| F4 | Control register semantics across both generators | L, F3 | L | Q2 |
| F5 | Numeric classes, tolerances, reduction identities | L, S, F9 | M | Q2 |
| F6 | Byte order, sub-byte packing, size of d0 | L | S | Q1 |
| F7 | Conformance suite derived from the rules | F1, F2, F3, F5, T, S | M | Q2 |
| F8 | Front-end rejections the spec already implies | L | S | Q1 |
| F9 | Executable reference interpreter of the exact classes | F1, F3, L | M | Q1 |
