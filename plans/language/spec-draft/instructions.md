# Instructions

This page is S5 of the spec.
An instruction reads its input regions, its attributes, and the state, and writes its output regions and the state.

## S5.1 Instruction attributes

- `add_instruction(name, computation_attr, addressing_attr, *, update=(), constraints=())` declares an instruction and returns it.
- `computation_attr` lists the computational attributes, read as `@c.<name>`.
- `addressing_attr` lists the addressing attributes, read as `@a.<name>`.
- `update` lists state updates (S5.4) and `constraints` lists constraints (S5.3), each passed by keyword.
- Each of `computation_attr`, `addressing_attr`, `update`, and `constraints` is a list of `str`; a bare string or a non-`str` entry is rejected.
- Each attribute takes a signed 64-bit integer value each time the instruction is issued (S5.6).
- A computational attribute configures what the instruction computes and may appear in any expression.
- An addressing attribute says where a region lies and may appear only in a start or a constraint (S6.1).

## S5.2 Regions: inputs and outputs

- `set_inputs(inputs)` and `set_outputs(outputs)` each take a list of regions `[buffer, starts, sizes]`.
- A region gives one start and one size per access dimension of its buffer; `d0` has one.
- A start or a size is an `int` or a `str` holding an expression (S6).
- Along each access dimension, a region covers the indices from its start to its start plus its size, exclusive.
- Along the unit shape, a region covers every index.
- The region shape is the sizes followed by the unit shape.
- Input region k, counting from 0, enters the semantics as `%In<k>`; output region k is written from `%Out<k>` (S7.2).
- The instruction reads every input region before it writes any output region.
- A region whose start is below 0, whose size is below 1, or whose start plus size passes its access dimension is a fault (S1.3.3).
- For `d0` the access dimension is the byte count of the `hbm` contents (S5.6).
- Two output regions of one instruction that overlap are a fault (S1.3.4).
- An instruction gives its semantics, its inputs, and its outputs, or none of them; with none it only changes state.

## S5.3 Constraints

- A constraint is a condition (S6.3) over attributes and state.
- `constraints` of `add_instruction` gives every constraint the ISA description writes; placement marks append more (S5.5).
- A constraint given to `add_instruction` reads the state before the instruction's updates run.
- A constraint a placement mark appends reads the state after them, as the starts it compares do (S5.5).
- Issuing an instruction where a constraint does not hold is a fault (S1.3.5).

## S5.4 State updates

- A state update is `@s.<name> = <expression>` (grammar rule `update_text`).
- Updates run in the order given, each seeing the updates before it.
- Starts, sizes, shape dimensions, and operation attribute values read the state after all updates.
- S5.6 gives the full order of an issue.

## S5.5 In-place and placement marks

- `set_inplace(inplace)` takes one flag per input region, each of type exactly `bool`.
- A true flag k makes output region 0 reuse the storage of input region k.
- The two regions name the same buffer and have equal sizes, compared by tokens ignoring whitespace; `set_inplace` rejects otherwise.
- Their starts must be equal when the instruction is issued, else the issue is a fault (S1.3.6).
- At most one flag is true.
- The mark never changes the computed result, since every input is read before any output is written (S5.2).
- `set_inputs` resets every in-place flag to false, so `set_inplace` comes after it.
- `set_apart(apart)` and `set_no_partial_overlap(no_partial)` are placement marks: each takes one `bool` flag per input region.
- A placement mark applies only to an instruction with exactly one output region.
- A marked input lies in the output's buffer and is not marked in place.
- A placement mark means exactly the constraints it appends; generators see only those constraints.
- For each marked input, `set_apart` appends one constraint: along at least one access dimension whose starts differ in text, the two regions do not overlap.
- `set_no_partial_overlap` appends one constraint for each marked input and each access dimension whose starts differ in text.
- That constraint holds when the starts there are equal, or the regions are apart along some access dimension whose starts differ.
- Starts are compared by their tokens, ignoring whitespace only; parentheses and operand order count.
- `set_apart` rejects an input whose starts all match the output's in text.
- A placement mark given again replaces the constraints it appended before.
- `set_inputs` and `set_outputs` drop the constraints both placement marks appended, so marks come after the regions.
