# Declarations

This page is S4 of the spec.
An ISA specification builds an accelerator by calls in a fixed order: buffers and state first, then instructions.

## S4.1 Names

- Every name an ISA specification gives is a `str` that is an identifier: an ASCII letter or `_`, then ASCII letters, digits, and `_`.
- A name is not in `keyword.kwlist` of the Python version the front end runs on; soft keywords such as `match` and `type` are allowed.
- This holds for the accelerator, a buffer, an instruction, an instruction attribute, and a state variable.
- Buffers, instructions, and state variables each have names unique within their kind.
- An instruction attribute's name is unique across both attribute lists of its instruction.
- A value in a semantics is named by `%` and a semantics `IDENTIFIER` (S2.1), which may also contain dots.
- TAIDL-TO reserves further names for the code it generates; its list lives in the TAIDL-TO docs, and its error names the word, the generator, and that list.

## S4.2 Accelerator

- `Accelerator(name, base_dir=None)` starts an accelerator with one buffer, `d0`.
- `base_dir` is where generators write; by default it is the directory of the file that calls `Accelerator`, or the working directory when there is no such file.
- A generator takes the accelerator, as in TAIDL-TO's `generate_oracle(acc)` and ACT-Backend's `generate_backend(acc)`.

## S4.3 Buffers and the hbm buffer `d0`

- `add_buffer(name, access_dims, base_element_dims, element_type)` declares a buffer.
- `access_dims` lists the access dimensions: a region gives one start and one size for each.
- `base_element_dims` is the unit shape: the shape of one addressed unit, which a region always covers whole.
- Each of the two is a list or tuple of values of type exactly `int`, each at least 1, and either may be empty.
- `element_type` is the buffer's element type, one of S7.3.
- The buffer's full shape is `access_dims` followed by `base_element_dims`; with both empty it is a rank-0 buffer of one element.
- `d0` is hbm: a predeclared `u8` buffer with one access dimension and an empty unit shape.
- The size of `d0`'s access dimension is not declared; it is the byte count of the `hbm` contents a program is run with (S5.6).

## S4.4 State

- State is a named value of a declared kind, an integer by default.
- `add_state(name, value)` declares a state variable `@s.<name>` and its value before the first issue.
- The value has type exactly `int`, so `True` is rejected, and lies in signed 64 bits.
- A state variable holds a signed 64-bit integer that persists from one instruction to the next.
- Instructions change it only by state updates (S5.4).

## S4.5 Declaration order

- An ISA specification declares its buffers and state before its first instruction.
- The front end rejects `add_buffer` and `add_state` after an instruction (S9).
