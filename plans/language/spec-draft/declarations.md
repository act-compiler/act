# Declarations

This page is S4 of the spec.
An ISA description builds an accelerator by calls in a fixed order: buffers and state first, then instructions.

## S4.1 Names

- Every name an ISA description gives is a `str` that is an identifier: an ASCII letter or `_`, then ASCII letters, digits, and `_`.
- A name is not in `keyword.kwlist` of the Python version the front end runs on; soft keywords such as `match` and `type` are allowed.
- This holds for the accelerator, a buffer, an instruction, an instruction attribute, and a state variable.
- Buffers, instructions, and state variables each have names unique within their kind.
- An instruction attribute's name is unique across both attribute lists of its instruction.
- A value in a semantics is named by `%` and a semantics `IDENTIFIER` (S2.1), which may also contain dots.

## S4.2 Accelerator

- `Accelerator(name, base_dir=None)` starts an accelerator with one buffer, `d0`.
- `base_dir` is where generators write; by default it is the directory of the file that calls `Accelerator`, or the working directory when there is no such file.
- A generator takes the accelerator, as in TAIDL-TO's `generate_oracle(acc)` and ACT-Backend's `generate_backend(acc)`.

## S4.3 Buffers and the hbm buffer `d0`

- `add_buffer(name, dimensions, unit_dim, var_type)` declares a buffer.
- `dimensions` lists the access dimensions: a region gives one start and one size for each.
- `unit_dim` is the unit shape: the shape of one addressed unit, which a region always covers whole.
- Each of the two is a list or tuple of values of type exactly `int`, each at least 1, and either may be empty.
- `var_type` is the buffer's element type, one of S7.3.
- The buffer's full shape is `dimensions` followed by `unit_dim`; with both empty it is a rank-0 buffer of one element.
- `d0` is hbm, the off-chip memory: a `u8` buffer with one access dimension and an empty unit shape.
- The size of `d0`'s access dimension is not declared; it is the byte count of the `hbm` contents a program is run with (S5.6).

## S4.4 State

- `add_state(name, value)` declares a state variable `@s.<name>` and its initial value.
- The initial value has type exactly `int`, so `True` is rejected, and lies in signed 64 bits.
- A state variable holds a signed 64-bit integer that persists from one instruction to the next.
- Instructions change it only by state updates (S5.4).

## S4.5 Declaration order

- An ISA description declares its buffers and state before its first instruction.
- The front end rejects `add_buffer` and `add_state` after an instruction (S9).

## Names reserved by generators

This table is TAIDL-TO's, not the language's; `generate_oracle` rejects an ISA description that uses one of these names.

| name | reserved by the oracle |
| --- | --- |
| buffer | `In<k>`, `Out<k>`, `instruction_counter`, `total_loop_counter`, `loop_var`, `loop_final`, `call`, `debug_prefix`, `debug_slice` and `debug_slice_f64` |
| instruction | a name between double underscores, such as `__debug__`, and a name the generated `api.py` defines or reads, such as `start_loop`, `end_loop`, `debug` or `str` |
| attribute | a name between double underscores, `comp_attrs`, `cur_scope`, `global_counters`, `instruction_scopes`, `ssa_types`, `state`, and a name its instruction's generated function reads, such as `str` |
| value | `%` and a buffer's name, or `%instruction_counter`, `%total_loop_counter`, `%loop_var` or `%loop_final` |
