# Instructions

This page is S5 of the spec.
An instruction reads its input regions, its attributes, and the state, and writes its output regions and the state.

## S5.1 Instruction attributes

- `add_instruction(name, computation_attr, addressing_attr, *, update=(), constraints=())` declares an instruction and returns it.
- `computation_attr` lists the computational attributes, read as `@c.<name>`.
- `addressing_attr` lists the addressing attributes, read as `@a.<name>`.
- `update` lists state updates (S5.4) and `constraints` lists constraints (S5.3), each passed by keyword.
- Each of `computation_attr`, `addressing_attr`, `update`, and `constraints` is a list of strings; one string in its place is rejected.
- Each attribute takes a signed 64-bit integer value each time the instruction is issued.
- A computational attribute configures what the instruction computes and may appear in any expression.
- An addressing attribute says where a region lies and may appear only in a start or a constraint (S6.1).

## S5.2 Regions: inputs and outputs

- `set_inputs(inputs)` and `set_outputs(outputs)` each take a list of regions `[data model, starts, sizes]`.
- A region gives one start and one size per addressed dimension of its data model, and one of each for `d0`.
- A start or a size is an integer or an expression (S6).
- Along each addressed dimension, a region covers the indices from its start to its start plus its size, exclusive.
- Along the unit shape, a region covers every index.
- The region shape is the sizes followed by the unit shape; for `d0` it is the one size.
- Input region k, counting from 0, enters the semantics as `%In<k>`; output region k is written from `%Out<k>` (S7.2).
- The instruction reads every input region before it writes any output region.
- A region that reaches past its data model, or whose size is 0 or less, is undefined (S1.3).
- Two output regions of one instruction that overlap are undefined (S1.3).
- An instruction with no semantics, no inputs, and no outputs only changes state.

## S5.3 Constraints

- A constraint is a condition (S6.3) over attributes and state.
- `constraints` of `add_instruction` gives the first constraints; `add_constraints(constraints)` appends more.
- An instruction is valid only for attribute and state values where every constraint holds.
- Constraints read the state before the instruction's updates run.
- Issuing an instruction where a constraint does not hold is undefined (S1.3).

## S5.4 State updates and execution order

- A state update is `@s.<name> = <expression>` (S3.2 `update_text`).
- An instruction runs in this order:
  1. Its constraints are evaluated against the state before it.
  2. Its updates run in the order given, each seeing the updates before it.
  3. Its starts, sizes, and semantics expressions are evaluated against the state after all updates.
  4. It reads its input regions, computes its semantics, and writes its output regions.

## S5.5 In-place and placement marks

- `set_inplace(inplace)` takes one flag per input region; a true flag marks an input the instruction overwrites.
- `set_inputs` resets every in-place flag to false, so `set_inplace` comes after it.
- An in-place mark does not change what the instruction computes, since it reads its inputs before it writes (S5.2).
- What else an in-place mark tells a generator, such as storage reuse, is implementation-defined (S1.3).
- `set_apart(apart)` and `set_no_partial_overlap(no_partial)` are placement marks: each takes one flag per input region.
- A placement mark applies only to an instruction with exactly one output region.
- A marked input lies in the output's data model and is not marked in place.
- A placement mark means exactly the constraints it appends; generators see only those constraints.
- For each marked input, `set_apart` appends one constraint: along at least one addressed dimension whose starts differ in text, the two regions do not overlap.
- `set_no_partial_overlap` appends one constraint for each marked input and each addressed dimension whose starts differ in text.
- That constraint holds when the starts there are equal, or the regions are apart along some addressed dimension whose starts differ.
- Starts are compared by their parsed text, so spacing never makes them differ.
- `set_apart` rejects an input whose starts all match the output's in text.
- A placement mark given again replaces the constraints it appended before.
- `set_inputs` and `set_outputs` drop the constraints both placement marks appended, so marks come after the regions.
