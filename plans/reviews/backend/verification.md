# Verification of the act-backend coverage and bug tables

This file checks each row of `act-backend/docs/coverage.md` and `work/backend/plan/bugs.md` against the code.
Verdicts are confirmed, wrong, or unverifiable; a caveat follows a confirmed verdict where the claim holds but a detail is off.
Nothing was run; every verdict rests on code read in this review.

## Key

- `SC` is `act-backend/act_backend/spec_checks.py`.
- `GEN` is `act-backend/act_backend/generator.py`.
- `RS` is `act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py`.
- `G` is `act-backend/act_backend/generic`.
- A row id `C<n>` is line n of `coverage.md`; `D<n>` is a row of `bugs.md`.

## The three specific questions

| Question | Verdict | Evidence |
| --- | --- | --- |
| `@s.*` reads and state variables are rejected with a clear error | Confirmed | Start: `SC:249-251` names the variable. Semantics: `SC:379-385` names it. Update: `SC:374-378`. Size: `SC:274-292`. Constraint: `SC:328,336,351-367`. |
| Instructions without semantics are dropped unchecked | Confirmed | `GEN:119-122` filters on `instr_semantics_ast`; every `check_*` at `GEN:125-131` takes the filtered list; no message or print at generation. |
| Count of supported operation words | 18 (19 with `parameter`) of 78 in S8.1 | `OPERATIONS` has 18 keys (`isel_ir2isa_rewrites_txt_generator.py:11-30`); `parameter` is accepted separately (`SC:53-55`); `TensorOp` has the same 18 plus the internal `OpEye` (`G/src/ir/egraph.rs:10-29`). S8.1 lists 78 words (`taidl/docs/operations.md:12-22`). |

- The size and constraint errors do not name the state variable; they say the size or constraint is not supported (`SC:287-290`, `SC:364-367`).
- Those two errors are clear about the instruction and the text but not about the cause.

## coverage.md rows

| Row | Claim | Verdict | Evidence (file:line) |
| --- | --- | --- | --- |
| C5 | A rejected feature raises `ValueError` naming the instruction, before anything is written | Confirmed; caveat | `GEN:90-98` deletes the old binary and crate first; `SC:297-305` names the data model, not an instruction; cost-library errors are `FileNotFoundError` (`GEN:117`). |
| C10 | Accelerator and name supported | Confirmed | `GEN:91,95` use `accelerator.name`. |
| C11 | Names supported | Confirmed | Generated names are prefixed: `instruction_info.py:50-71`; type names de-duplicated `instruction_info.py:32-47`. |
| C12 | Addressed dimension past int64 rejected | Confirmed; understated | `SC:297-305` loops `model.var_dim`, which is addressed plus unit (`taidl/taidl/data_model.py:47`), so a unit dimension is rejected too. |
| C13 | `d0` supported | Confirmed | `instruction_info.py:71` maps `d0` to `HBM`; `malloc_globals_cc_generator.py:13`. |
| C14 | `var_type` not checked against region values | Confirmed | No hit for `var_type` anywhere under `act_backend`. |
| C15 | State variable declared is accepted | Confirmed | No hit for `initial_state` under `act_backend/*.py`. |
| C16 | Instruction with semantics supported | Confirmed | `GEN:120-122`, `GEN:143-176`. |
| C17 | No semantics: left out | Confirmed | `GEN:119-122`; `README.md:51`. |
| C18 | `@c.` must be in a dimension or an operation attribute, alone or times a positive integer | Confirmed | `SC:222-240`; `RS:176-217`. |
| C19 | `@a.` in a start or a constraint supported | Confirmed | `GEN:65-73` splits address constraints and passes starts; `malloc_instructions_h_generator.py:71-73`. |
| C20 | `@a.` in a semantics rejected | Confirmed | `SC:199-206`. |
| C21 | One or several inputs supported | Confirmed | `GEN:63` (`rhs_size`); `SC:194-198`. |
| C22 | Exactly one output supported | Confirmed | `SC:22-29`. |
| C23 | Zero or several outputs rejected | Confirmed | `SC:25-29`. |
| C24 | Start: operators supported; 64-bit overflow or zero divisor rejected | Confirmed | `SC:243-270` (`start_refusal`, `past_64_bits`, `division_refusal`); `SC:315-324`. |
| C25 | Size: same limits | Confirmed | `SC:284-294`; `@a.` and `@s.` also rejected there (`SC:274-281`). |
| C26 | Size overflow for some attribute values: not selected | Confirmed | `operand_sizes.py:6-20`; `templates/ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.size_check.txt:1-3`; `expression_writer.py:93-97`. |
| C27 | Presumed reshape: literal count above the region's is rejected | Confirmed | `SC:388-418`; only `tile > room` raises (`SC:414`). |
| C28 | Constraint: `and`, `or`, brackets; overflow does not hold | Confirmed | `RS:405-411,453,495`; `expression_writer.py:111-114`. |
| C29 | Address constraint dividing by a computed-zero operand rejected | Confirmed | `SC:338-339`. |
| C30 | `set_inplace` supported | Confirmed | `ir_egraph_rs_generator.py:94,167-171`; `instruction_info.py:112`. |
| C31 | Placement marks supported | Confirmed | `SC:357-362` reads `instruction.placed`; the front end appends constraints (`taidl/taidl/instruction.py:231`). |
| C32 | `@s.` in start, size, constraint or semantics rejected | Confirmed | `SC:249-251`, `SC:274-281`, `SC:328,336`, `SC:379-385`; see the questions table for the message caveat. |
| C33 | State update rejected | Confirmed | `SC:374-378`; unchecked on a dropped instruction (`GEN:119-122`). |
| C34 | Floor `/` and `%` | Confirmed | `expression_writer.py:10`; `G/cpp/malloc/include/solver.h:61,72`; `templates/ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.quotient.txt:2-8`. |
| C35 | Unread value, or `ROOT` above the last line, rejected | Confirmed | `SC:186-192`; the check covers parameter lines too, so an unread input is rejected. |
| C36 | 13 element types supported | Confirmed | `RS:24-38`. |
| C37 | Six element types rejected | Confirmed; scope | `SC:47-51` checks semantics lines only; a data model `var_type` of those types is not checked (C14). |
| C38 | Dimension past 64 bits rejected | Confirmed | `SC:213-221`. |
| C39 | Shape and type agreement not checked | Confirmed | Absent from `SC:32-91`; `RS:593-603` matches declared shape and `dtype`. |
| C40 | 18 operation words supported | Confirmed | Same 18 words as `isel_ir2isa_rewrites_txt_generator.py:11-30`, order aside. |
| C41 | Every other operation word rejected | Confirmed | `SC:57-61`. |
| C42 | `slice` stride other than 1 rejected | Confirmed | `SC:80-88`. |
| C43 | `constant` with a backtick expression rejected | Confirmed | `SC:72-78`, `SC:96-107`. |
| C44 | `constant` literal outside its element type rejected | Confirmed for `pred` and integers; overstated for floats | `SC:97-105` range-checks `pred`, `s`, `u`; `SC:106-107` accepts any non-boolean, non-backtick literal for a float type. |
| C45 | Operation attributes of six words supported; one-element `broadcast` matches any map | Confirmed | `RS:41` (`COMPARED`); `RS:347-352` (`spread` guard). |
| C46 | Element types per operation not checked | Confirmed | No per-operation type check in `SC`; `RS:593-603`. |

## bugs.md rows

| Row | Claim | Verdict | Evidence (file:line) |
| --- | --- | --- | --- |
| D1 | `check_regions` rejects only a literal count above the region's | Confirmed | `SC:408-414`; `taidl/docs/semantics.md:47` makes any other count an error. |
| D2 | Generator never reads `var_type` | Confirmed; effect unverifiable | No hit for `var_type` under `act_backend`; the row itself says the effect is unverified. |
| D3 | Shape and type agreement unchecked; ill-typed semantics silently never selected | Confirmed | `SC:32-91`; `RS:593-603`. |
| D4 | No-semantics instruction dropped before every check; silent | Confirmed | `GEN:119-122`; `GEN:125-131`; `README.md:51`. |
| D5 | `assert(false)` then a dummy return; plain `cmake ..` | Confirmed | `G/cpp/malloc/src/parser.cc:9-15,186-190`; `constraint.cc:113-117`; `G/build.rs:57-61`; no `NDEBUG` or `CMAKE_BUILD_TYPE` in `cpp/malloc/CMakeLists.txt`. |
| D6 | Rewrites bound by number to hand-indexed functions; missing arm panics | Confirmed | `G/src/isel/rewrites/applier.rs:15-58`, panics at `:27,41,55`; `ir2ir_rewrites.rs:9,60,89`; `ir2ir_rewrites.txt` has ten numbered lines. |
| D7 | Candidate construction panics on unknown op or unresolvable `ANY` | Confirmed; reachability unverifiable | `G/src/ir/pii.rs:36,42,48`; `G/src/ir/buffer.rs:15`. |
| D8 | `merge` treats an empty shape as unset | Confirmed | `G/src/ir/egraph.rs:405-415` (`!a.shape.is_empty() && !b.shape.is_empty()`). |
| D9 | `honoured` and banned terms in messages | Confirmed; one detail wrong | `honoured` at `SC:331`; "address attribute" at `SC:204`; "machine state" at `SC:376`; "refuse" occurs in docstrings (`SC:1,23,180`), not in an error message. |
| D10 | Candidate printed to text, parsed back by the C++ allocator | Confirmed | `G/src/pipeline.rs:252-253`; `G/src/ir/pii.rs:69-93`; `G/cpp/malloc/src/act_malloc.cc:45-66`; static library `cpp/malloc/CMakeLists.txt:40`. |
| D11 | Each extraction pass clones the e-graph | Confirmed; line off by one | `G/src/pipeline.rs:235` (`egraph.clone()` inside `pass`, `:226`); the mutation is `slow.rs:24`, not `:21-23`. |
| D12 | `expression_writer.py` writes `@s.n` as attribute `n`; latent | Confirmed | `expression_writer.py:51-53`; every path rejects `@s.` first (`SC:249,274,328,379`). |

## Missed features and gaps in coverage.md

- `parameter` is accepted (`SC:53-55`) and has no row; S7.5, S8.1.
- `concatenate` takes exactly 2 operands (`isel_ir2isa_rewrites_txt_generator.py:15`; `G/src/ir/egraph.rs:13` has `[Id; 2]`); S8.2 allows 1 or more.
- Row C40 lists `concatenate` as supported without that limit; it needs a `partial` row (S8.2).
- A `@a.` in a size is rejected (`SC:274-292`); no row names it (S6.1, S5.2).
- A constraint whose integers alone pass 64 bits is rejected (`SC:346-348`); C28 says only that overflow does not hold (S6.2, S6.3).
- A data model `var_type` of `s2`, `s4`, `u2`, `u4`, `f8e4m3fn` or `f8e5m2` is accepted; C37 reads as if it were rejected (S4.3, S7.3).
- Float constant literals are not range-checked (`SC:106-107`); C44 needs a `partial` note (S7.5).
- A rank-0 shape (`f32[]`, S7.4) has no row; D8 shows the e-graph treats an empty shape as unset (`G/src/ir/egraph.rs:406`), so support is unverified.
- Implementation-defined cases of S1.3 have no row: `bitcast_convert` byte order, the initial value of `reduce_add`, the size of `d0`, what an in-place mark does beyond the result; S1.2 asks each generator to document them.
- S4.5 and S9 are front-end rules and need no backend row.
- S8.5 operations are outside the language and need no row.
- A one-element `broadcast` guard reads `lhs_metadata[index - 1]` (`RS:348-350`); it assumes the broadcast operand is the previous assignment, and that is not verified.
