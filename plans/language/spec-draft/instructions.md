# Instructions

This page is S5 of the spec.
An instruction reads its input regions, its attributes, and the state, and writes its output regions and the state.

## S5.1 Instruction attributes

- `add_instruction(name, computational_attrs, addressing_attrs, *, update=(), constraints=())` declares an instruction and returns it.
- `computational_attrs` lists the computational attributes, read as `@c.<name>`.
- `addressing_attrs` lists the addressing attributes, read as `@a.<name>`.
- `update` lists state updates (S5.4) and `constraints` lists constraints (S5.3), each passed by keyword.
- Each of `computational_attrs`, `addressing_attrs`, `update`, and `constraints` is a list of `str`; a bare string or a non-`str` entry is rejected.
- Each attribute takes an integer value in its type each time the instruction is issued (S5.6); a negative value is legal unless its type is `u<w>`.
- An attribute's type is `s64` unless the width layer declares another (S11.2).
- An attribute name is a plain variable name; no tool infers meaning from it.
- A computational attribute configures what the instruction computes and may appear in any expression.
- An addressing attribute may appear only in a start or a constraint (S6.1).
- A start is any expression over attributes and state, such as `1024 - @a.addr_out`.

## S5.2 Regions: inputs and outputs

- `set_inputs(inputs)` and `set_outputs(outputs)` each take a list of regions `[buffer, starts, sizes]`.
- A region gives one start and one size per access dimension of its buffer; `d0` has one.
- A start or a size is an `int` or a `str` holding an expression (S6).
- Along each access dimension, a region covers the indices from its start to its start plus its size, exclusive.
- Along the unit shape, a region covers every index.
- The region shape is the sizes followed by the unit shape.
- Input region k, counting from 0, enters the semantics as `%In<k>`; output region k is written from `%Out<k>` (S7.2).
- An input region and an output region of one buffer overlap only as their declared relation allows (S5.5).
- Every region carries implicit constraints (S5.3) on its evaluated starts and sizes, never on attribute values: each evaluated start is at least 0, each size at least 1, and each start plus size at most its access dimension; a violated one is a fault (S1.3.3).
- For `d0` the access dimension is the byte count of the `hbm` contents (S5.6).
- Two output regions of one instruction that overlap are a fault (S1.3.4).
- An instruction gives its semantics, its inputs, and its outputs, or none of them; with none it only changes state.

## S5.3 Constraints

- A constraint is a condition (S6.3) over attributes and state.
- `constraints` of `add_instruction` gives every constraint the ISA specification writes; `constraints=` is the only spelling.
- One constraint mechanism holds four sources: constraints the ISA author writes, the implicit constraints of every region and every output pair (S5.2), the constraints each input-output pair's relation expands to (S5.5), and the range constraints of each declared type (S11.2).
- A constraint of any source reads the state before the issue, as every expression of the issue does (S5.6).
- Issuing an instruction where a constraint of any source does not hold is a fault (S1.3.5).

## S5.4 State updates

- A state update is `@s.<name> = <expression>` (grammar rule `update_text`).
- Updates are one parallel assignment: every update expression reads the state before the issue, and all updates take effect together at the end of the issue.
- An update that needs another update's new value substitutes its text: with `@s.a = @s.a + 64`, the new `@s.a` is written `@s.a + 64`.
- Starts, sizes, shape dimensions, and operation attribute values also read the state before the issue.
- Post-increment addressing is start `@s.ptr` with update `@s.ptr = @s.ptr + 64`.
- S5.6 gives the full order of an issue.

## S5.5 Overlap between inputs and outputs

TAIDL assumes no hazard model; the ISA author declares how an instruction behaves when an input and an output of one buffer overlap.
Each same-buffer pair of input region k and an output region has exactly one relation, declared by a mark or given by the default.
The front end expands that relation into constraints of S5.3, through the mechanism that gives every region its implicit constraints.

| relation | meaning | mark |
| --- | --- | --- |
| tied | the output reuses the storage of input k | `set_inplace` |
| apart | no shared element | `set_apart` |
| exact-or-apart | equal starts and equal sizes in every access dimension, or apart | `set_no_partial_overlap`, or no mark |
| may overlap | no overlap restriction | `set_may_overlap` |

- An undeclared same-buffer input-output pair is exact-or-apart.
- A pair in different buffers gets no relation and no constraint.
- Two output regions of one instruction stay apart; that is an implicit constraint of S5.3 (S1.3.4).
- Any further rule, such as two inputs that must differ or a positional rule, is a constraint the ISA author writes, such as `@a.s1 != @a.s2`.
- A value under any permitted overlap is the semantics evaluated on the state before the issue (S5.6); no read and write order is assumed.

### Expansions

Every term is the evaluated start or size expression of a region, never an attribute value or name (S5.1).

- `apart(R, S)`: along some access dimension, `R.start + R.size <= S.start` or `S.start + S.size <= R.start`.
- `same(R, S)`: along every access dimension, equal starts and equal sizes; a dimension whose start and size texts are identical is skipped.
- `set_inplace(k)`: `same(In k, Out)`, and the tie is recorded for the backend.
- `set_apart(k)`: `apart(In k, Out)`.
- `set_no_partial_overlap(k)`: `same(In k, Out) or apart(In k, Out)`, the explicit spelling of the default.
- `set_may_overlap(k)`: no constraint.
- The expanded constraint text is the canonical form; `apart` and `same` are names for it in this document.
- Starts and sizes that differ in text but evaluate equal satisfy `same` at issue.
- The front end records, per same-buffer input-output pair, its relation and, for `set_inplace`, the tie; a backend reads that record.

### Mark calls

- `set_inplace(inplace)`, `set_apart(apart)`, `set_no_partial_overlap(no_partial)`, and `set_may_overlap(may_overlap)` each take one flag per input region, each of type exactly `bool`.
- A mark addresses output region 0 and applies only to an instruction with exactly one output region; an instruction with several outputs gets the default on each pair.
- A marked input lies in the output's buffer; an input carries at most one mark, and at most one input is marked in place.
- A mark given again replaces the relation it declared before.
- `set_inputs` and `set_outputs` reset every mark, so marks come after the regions.

### Rejected at declaration

Text alone decides these cases, so the front end rejects them (S9).

- An in-place mark that pairs an input and an output in different buffers.
- An in-place input whose size text differs from the output's, compared by tokens ignoring whitespace.
- `set_apart` on an input whose start and size texts all match the output's.
- An input whose start and size texts all match the output's, in the same buffer, without `set_inplace`.

### Violation

- An issue where any constraint fails, implicit, from a mark, or written, is the one fault of S1.3.5, before any write.
- A run-time start depends on attributes and state, so the front end never rejects a pair for an overlap it cannot decide from text.
- A front-end lint warns on an unmarked same-buffer pair of an instruction whose semantics is not elementwise; the semantics is unchanged.
