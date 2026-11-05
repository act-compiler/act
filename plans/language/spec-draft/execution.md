# Execution model

This page is S5.6 of the TAIDL language specification: what a program of instructions is, and how one issue runs.
A test oracle or a simulator runs programs; a compiler backend emits programs that never fault.

## S5.6 Programs and issues

### Program

- A program is a sequence of issues run in order on one accelerator.
- Program order is the only order: TAIDL has no concurrency, and a constraint used as a wait faults when false (S1.3.5).
- An issue names one instruction and gives a value to each of its attributes, by name, in any order.
- An attribute value is an integer; one outside signed 64 bits is a fault (S1.3.10).
- An issue that names no instruction, omits an attribute, names one the instruction lacks, or gives a value that is not an `int` is invalid assembly code: the test oracle stops with an error naming the issue (owner decision 40), and it is not a fault of S1.3.
- A program is run with `hbm` contents: a sequence of bytes that `d0` holds before the first issue.
- A run yields the `hbm` contents after the last issue, or the fault report; the final state and on-chip buffers are observable only through a generator's own interface.

### Storage before the first issue

- `d0` holds the caller's `hbm` contents in full; its size is their byte count.
- Every element of a buffer with a declared initial value holds that value (S4.3).
- No element of any other buffer holds a value yet: a read of an element that holds no value and that no earlier issue of the program wrote is a fault (S1.3.11).
- "Written" means written by an issue of this program, so a stale value left in a reused buffer by an earlier program is not detected.
- Every state variable holds its value from `acc.state` (S4.4).
- An element no output region writes keeps its value from one issue to the next.

### One issue

Every expression of an issue reads the state before the issue; its updates and its outputs take effect together at its end.

1. Evaluate the constraints, starts, sizes, shape dimensions, symbolic `constant` values, operation attribute values, and state update expressions against the state before the issue.
2. Check every fault case of S1.3 that applies; if one fails, the issue is a fault and steps 3 and 4 do not run.
3. Read every input region, then compute every value of the semantics in the order the function recorded it.
4. Write every output region from the value its `write` took, and assign every state update at once (S5.4).

- When an input and an output region overlap as their declared relation permits (S5.5), the outputs are the semantics on the state before the issue; no read and write order is assumed.
- Every recorded value of a semantics is computed; there is no lazy or dead value.
- Every comparison of a condition is evaluated, left to right; a fault in any of them is a fault of the issue.
- A fault depends only on attributes, state, the recorded expressions, the size of `hbm`, and which elements earlier issues of the program wrote; every tensor value gives a fixed result (S1.3.12).

### Faults

- A fault writes no region and no state.
- The program stops at the first fault, and every earlier issue has committed.
- The test oracle reports the faulting issue's index, counting from 0, the instruction, and a section id of a clause that applies.
- A compiler backend is a producer: it never emits an issue that faults, and lists in its coverage table each fault case it cannot see, such as state before the kernel or the size of `d0`.

### Region values

- A region value has its buffer's element type and the region shape (S5.2).
- Element `[i_1, ..., i_n, u_1, ..., u_m]` of the region value is element `[s_1 + i_1, ..., s_n + i_n, u_1, ..., u_m]` of the buffer, where `s_j` is the region's j-th start.
- For `d0`, element `[i]` of the region value is byte `s_1 + i`.
- A buffer with no access dimension and no base element shape has one element; its region value has shape `[]` per S7.4.
