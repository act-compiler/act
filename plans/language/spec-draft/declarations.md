# Declarations

This page is S4 of the spec.
An ISA description builds an accelerator by calls in a fixed order: data models and state variables first, then instructions.

## S4.1 Names

- Every name an ISA description gives is an identifier: an ASCII letter or `_`, then ASCII letters, digits, and `_`.
- A name is not a Python keyword.
- This holds for the accelerator, a data model, an instruction, an instruction attribute, and a state variable.
- Data models, instructions, and state variables each have names unique within their kind.
- An instruction attribute's name is unique across both attribute lists of its instruction.
- A value in a semantics is named by `%` and a semantics `IDENTIFIER` (S2.1), which may also contain dots.

## S4.2 Accelerator

- `Accelerator(name, base_dir=None)` starts an accelerator with one data model, `d0`.
- `base_dir` is where generators write; by default it is the directory of the file that calls `Accelerator`, or the working directory when there is no such file.
- A generator takes the accelerator, as in TAIDL-TO's `generate_oracle(acc)` and ACT-Backend's `generate_backend(acc)`.

## S4.3 Data models and the off-chip memory `d0`

- `add_data_model(name, dimensions, unit_dim, var_type)` declares a data model.
- `dimensions` lists the addressed dimensions: a region gives one start and one size for each.
- `unit_dim` is the unit shape: the shape of one addressed unit, which a region always covers whole.
- Each of the two is a list of positive integers, and either may be empty.
- `var_type` is the data model's element type, one of S7.3.
- The data model's full shape is `dimensions` followed by `unit_dim`.
- `d0` is the off-chip memory: a flat sequence of `u8` bytes with no declared size.
- `d0` has no addressed dimension and no unit shape, yet a region of `d0` takes one start and one size, both in bytes.

## S4.4 State variables

- `add_initial_state(name, value)` declares a state variable `@s.<name>` and its initial value, a Python `int`.
- A state variable holds a signed 64-bit integer that persists from one instruction to the next.
- Instructions change it only by state updates (S5.4).
- An initial value outside signed 64 bits is undefined; the front end does not reject it.

## S4.5 Declaration order

- An ISA description declares its data models and state variables before its first instruction.
- The front end rejects `add_initial_state` after an instruction; it does not yet reject `add_data_model` there (S9).

## Names reserved by generators

This table is TAIDL-TO's, not the language's; `generate_oracle` rejects an ISA description that uses one of these names.

| name        | reserved by the oracle                                                                                                                                                                            |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| data model  | `In<k>`, `Out<k>`, `instruction_counter`, `total_loop_counter`, `loop_var`, `loop_final`, `call`, `debug_prefix`, `debug_slice` and `debug_slice_f64`                                             |
| instruction | a name between double underscores, such as `__debug__`, and a name the generated `api.py` defines or reads, such as `start_loop`, `end_loop`, `debug` or `str`                                    |
| attribute   | a name between double underscores, `comp_attrs`, `cur_scope`, `global_counters`, `instruction_scopes`, `ssa_types`, `state`, and a name its instruction's generated function reads, such as `str` |
| value       | `%` and a data model's name, or `%instruction_counter`, `%total_loop_counter`, `%loop_var` or `%loop_final`                                                                                       |
