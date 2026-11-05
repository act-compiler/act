# Object model

This page is S4.6 of the TAIDL language specification: what a generator reads from an accepted ISA specification.
The typed contract, its node classes, its members, and `CONTRACT_VERSION` are shared plan item U2; the `taidl` README documents each member with its type.

## S4.6 The `Model`

- `taidl.check(acc)` returns a `Model`, and only `check` constructs one (S9).
- A `Model` is frozen: no member of it or of its nodes can be set after `check` returns.
- It holds one typed node per buffer, state variable, and instruction, and per region, constraint, state update, and value of each instruction.
- An instruction node lists its attributes in parameter order, each with its kind and type, and its regions in `read` and `write` order.
- Its constraints include those every source of S5.3 expands to, each tagged with its source.
- Each same-buffer input-output pair carries its relation and, for `inplace`, the tie (S5.5); a backend reads that record.
- Each value node carries its tensor operator, operands, operation attributes, and its inferred element type and shape.
- Generator metadata sits on the accelerator, buffer, and instruction nodes, by namespace (S12).
- Every node records its call site: the file and line of the first frame outside `taidl`; a whole-instruction node records the function's first line.
- A generator reads only the `Model`, never the `Accelerator` or a trace record.

## Printing

- The `Model` prints each instruction as XLA-HLO text, with its attributes, regions, constraints, and state updates.
- The printed text answers "what did the front end understand" and is a log format; no front end reads it back.
- Two `Model` objects compare equal when every node matches, ignoring call sites.
