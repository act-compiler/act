# Execution model

This page is S5.6 of the spec: what a program of instructions is, and how one issue runs.
A test oracle or a simulator runs programs; a compiler backend emits programs that never fault.

## S5.6 Programs and issues

### Program

- A program is a sequence of issues run in order on one accelerator.
- An issue names one instruction and gives a value to each of its attributes, by name, in any order.
- An attribute value is an integer; one outside signed 64 bits is a fault (S1.3.10).
- A program is run with `hbm` contents: a sequence of bytes that `d0` holds before the first issue.

### Storage before the first issue

- Every element of every buffer but `d0` is 0.
- `d0` holds the `hbm` contents; its size is their byte count.
- Every state variable holds its initial value (S4.4).
- An element no output region writes keeps its value from one issue to the next.

### One issue

1. Evaluate the constraints given to `add_instruction` against the state before the issue.
2. Run the state updates in order on a copy of the state (S5.4).
3. Evaluate starts, sizes, shape dimensions, backtick operands, operation attribute values, and placement-mark constraints against the updated copy.
4. Check every fault case of S1.3 that applies; if one fails, the issue is a fault and steps 5 and 6 do not run.
5. Read every input region, then compute every line of the semantics in file order.
6. Write every output region from its `%Out<k>` value, then replace the state with the updated copy.

- Every line of a semantics runs; there is no lazy or dead line.
- `ROOT` has no run-time effect; outputs come from the `%Out<k>` values.
- Every comparison of a condition is evaluated, left to right; a fault in any of them is a fault of the issue.
- A fault writes no region and no state; the program stops at it, and the generator reports the instruction and the S1.3 row id.

### Region values

- A region value has its buffer's element type and the region shape (S5.2).
- Element `[i_1, ..., i_n, u_1, ..., u_m]` of the region value is element `[s_1 + i_1, ..., s_n + i_n, u_1, ..., u_m]` of the buffer, where `s_j` is the region's j-th start.
- For `d0`, element `[i]` of the region value is byte `s_1 + i`.
- A buffer with no access dimension and no unit shape has one element; its region value has shape `[]` per S7.4.
