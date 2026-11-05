# Coverage of TAIDL

This table lists each TAIDL feature and whether act-backend generates a backend for an ISA specification that uses it.
Rows describe the code today; where an owner or panel decision sets a different v3 norm, the Notes column says so and names the plan item.
Section ids are those of the TAIDL language specification outline.

- A `rejected` feature makes `generate_backend` raise `ValueError` before any new file is written.
- The old binary and crate are deleted before the checks run (`generator.py:90-98`).
- A rejection names the instruction, except a buffer dimension past 64 bits, which names the buffer (`spec_checks.py:297-305`).
- A missing cost library raises `FileNotFoundError` (`generator.py:117`).
- A `partial` feature is supported within the limit its row states.
- No case is left undefined (owner decision 16): each row says whether it is supported, rejected, or not selected.
- The backend never stops because one candidate fails (owner decision 40): a failing candidate is dropped with a recorded reason (B41), and only the input rows below stop it.

## Data model and instructions

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| Accelerator and its name | S4.2 | supported | |
| Names of buffers, instructions, attributes, state | S4.1 | supported | Generated names are prefixed; type names are de-duplicated. |
| Buffer with addressed and unit dimensions | S4.3 | partial | Any dimension, addressed or unit, past the signed 64-bit range is rejected. |
| `hbm` buffer `d0` | S4.3 | supported | |
| Buffer element type `var_type`, 13 types below | S4.3 | partial | Not checked against the element type of a region's value (D2); in v3 `validate()` checks it (B21). |
| Buffer element type `s2`, `s4`, `u2`, `u4`, `f8e4m3fn`, `f8e5m2` | S4.3, S7.3 | partial | Accepted on a buffer and never read; the same types in a semantics line are rejected. |
| State declared with its value | S4.4 | partial | Declaring state is accepted; every read and update is rejected (rows below). |
| Instruction with semantics | S5 | supported | |
| Instruction with no semantics | S5, S7 | partial | Left out unchecked and without a message (D4); v3 rejects it, or names it when it only updates state (B24). |
| Computational attribute `@c.` | S5.1, S6.1 | partial | Each must appear in a shape dimension or an operation attribute, alone or times a positive integer; else rejected. |
| Addressing attribute `@a.` in a start or a constraint | S5.1, S6.1 | partial | A negative value is never chosen, since its solver variable starts at 0 (D15); owner decision 34 allows it; in v3 the variable spans the declared width-layer range, `s64` by default (B38). |
| Attribute type `s<w>` or `u<w>` (width layer) | S11.2 | partial | New in v3: the allocator domain is the declared range and a value outside it is never selected (B38). |
| Encoding block and derived attribute | S11.3, S11.4 | partial | New in v3: the backend reads the link, solves the encoded attribute, and computes the derived one in closed form; it never reads the block (B44); the assembler and disassembler are B45. |
| Addressing attribute `@a.` in a size | S5.2, S6.1 | rejected | |
| Addressing attribute `@a.` in a semantics | S6.1 | rejected | |
| `@s.` in a start, size, constraint or semantics | S6.1 | rejected | Start and semantics errors name the state; size and constraint errors say only that the text is not supported (B27); v3 folds a read in a start, size or constraint at compile time (B16). |
| State update | S5.4 | rejected | Only on an instruction with semantics; one with no semantics is left out; stays rejected in v3 (B16). |

## Regions, expressions and constraints

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| One input region or several | S5.2 | supported | |
| Exactly one output region | S5.2 | supported | |
| Zero or several output regions | S5.2 | rejected | |
| Start: integers, `@c.`, `@a.`, `+ - * / %`, unary `-`, brackets | S5.2, S6.2 | supported | A start whose integers alone pass 64 bits, or that divides by an operand whose integers alone compute zero, is rejected. |
| Size: integers, `@c.`, `+ - * / %`, unary `-`, brackets | S5.2, S6.2 | supported | Same two rejections as a start. |
| Size that passes 64 bits for some attribute values | S6.2 | supported | The instruction is not selected for those values. |
| Presumed reshape of a region's value | S7.2 | partial | Counted only when every dimension and size is a literal; only a count above the region's is rejected (D1); in v3 `validate()` checks it (B21). |
| Constraint: comparisons joined by `and`, `or`, brackets | S5.3, S6.3 | supported | A constraint whose arithmetic overflows at selection is a fault, which the backend reads as never selected. |
| Constraint whose integers alone pass 64 bits | S6.2, S6.3 | rejected | |
| Constraint over `@a.` with `/` or `%` by an operand whose integers alone compute zero | S5.3 | rejected | |
| Floor `/` and `%` with the sign of the divisor | S6.2 | supported | |
| In-place mark `set_inplace` | S5.5 | partial | The allocator permits the output to overlap the marked input but does not force it (D14); in v3 the output takes the input's offsets (B30) and a still-live shared input is copied (B11). |
| Placement marks `set_apart`, `set_no_partial_overlap` | S5.5 | supported | Their constraints follow the constraint rows; in v3 each also gives the pair relation the allocator reads (B39, B30). |
| May-overlap mark `set_may_overlap` | S5.5 | partial | New in v3 (L35): the mark expands to nothing and the allocator applies no restriction to the pair (B30). |
| Default relation of an unmarked same-buffer input-output pair | S5.5 | partial | The v3 norm is exact-or-apart. Today a pair whose live ranges touch gets no allocator constraint (D16); B30 applies the relation. If the solver cannot take it, the backend allocates default pairs apart, which never faults. |

## Semantics, types and operations

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| Semantics lines, `ROOT`, comments, optional semicolons | S7.1 | partial | A value no later line reads, an input included, is rejected; `ROOT` above the last line is rejected. |
| `parameter` lines | S7.5, S8.1 | supported | |
| Element types `pred`, `s8` to `s64`, `u8` to `u64`, `f16`, `bf16`, `f32`, `f64` in a semantics line | S7.3 | supported | |
| Element types `s2`, `s4`, `u2`, `u4`, `f8e4m3fn`, `f8e5m2` in a semantics line | S7.3 | rejected | |
| Shape dimension: integer or expression | S7.4 | partial | A dimension whose integers alone pass 64 bits is rejected. |
| Rank-0 shape `[]` | S7.4 | partial | Accepted at generation; selection is unverified, since the e-graph treats an empty shape as unset (D8); fixed by B26. |
| Agreement of shapes and element types between lines | S7.4, S8.3 | partial | Not checked; selection matches each value only at the shape and element type its line declares (D3); in v3 U4 checks it in the front end, and a disagreement at selection is the shared coded shape fault (B26). |
| Operation words `add`, `bitcast_convert`, `broadcast`, `constant`, `convert`, `copy`, `divide`, `dot`, `exponential`, `or`, `reduce_add`, `reshape`, `shift_left`, `shift_right_logical`, `slice`, `transpose`, `xor` | S8.1 | supported | |
| `concatenate` | S8.1, S8.2 | partial | Exactly two operands; S8.2 allows one or more. |
| Every other operation word | S8.1 | rejected | |
| `slice` with a stride other than 1 | S8.2 | rejected | |
| `constant` with an expression in backticks | S7.5 | rejected | |
| `constant` literal outside a `pred` or integer element type | S7.5 | rejected | |
| `constant` literal outside a float element type | S7.5 | partial | Accepted without a range check. |
| Operation attributes of `broadcast`, `concatenate`, `dot`, `reduce_add`, `slice`, `transpose` | S8.2 | supported | A one-element `broadcast` matches any dimensions map; its guard assumes the operand is the previous line's value (D13). |
| Element types per operation | S8.3 | partial | Not checked; selection matches each value only at the element type its line declares. |

## Implementation-defined cases (S1.3)

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| `bitcast_convert` byte order | S1.3 | partial | Not documented by the backend (B15). |
| Initial value of `reduce_add` | S1.3 | partial | Not documented by the backend (B15). |
| Size of `hbm` buffer `d0` | S1.3 | partial | Not documented by the backend (B15). |
| What an in-place mark does beyond the result | S1.3 | partial | Owner decision 17 defines it: the output reuses the marked input's storage (B30); an HBM-pinned or shared-view operand is an error (B11). |

## Preconditions the backend cannot see

The backend never emits an issue that faults; these preconditions depend on values it does not see at compile time (B15).

| Precondition | Why the backend cannot see it | Plan item |
| --- | --- | --- |
| State values | Read-only state folds only under the kernel's one known state | B16 |
| Size of `hbm` buffer `d0` | Set by the caller, not by the ISA specification | B15 |
| Reads of unwritten on-chip storage | Checked on each candidate; a violating one is dropped | B36 |

## Input kernel (owner decision 40)

The backend stops only on an invalid input kernel or one outside the input specification, naming the reason (B42).

| Input | Status | Notes |
| --- | --- | --- |
| XLA HLO text, `.hlo` file | supported | Read by `parse_hlo_module_to_egraph`; today only the extension is checked (D22). |
| StableHLO | rejected | Not supported today; v3 names the format in the error (B43). |
| torch-mlir | rejected | Not supported today; v3 names the format in the error (B43). |
| torch.fx | rejected | Not supported today; v3 names the format in the error (B43). |
| Malformed or ill-typed kernel | rejected | Stops with the line and the rule; some reader paths panic today (D22, B42). |
| Well-formed kernel outside the input specification | rejected | Stops naming the operation, element type, or format (B42). |
| One candidate failing in rewrite, extraction, allocation, or cost | not a stop | Dropped with a reason; today a cost failure exits (D17), and several panics and asserts remain (D18 to D20, B41). |

## Outside this table

- S4.5 and S9 are front-end rules; the backend has no row for them.
- S8.5 operations are outside the language.
