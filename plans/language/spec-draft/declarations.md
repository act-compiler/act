# Declarations

This page is S4 of the TAIDL language specification.
An ISA specification builds an accelerator by calls in a fixed order: buffers and state first, then instructions.

## S4.1 Names

- Every name an ISA specification gives is a `str` that is an identifier: an ASCII letter or `_`, then ASCII letters, digits, and `_`.
- A name is not in `keyword.kwlist` of the Python version the front end runs on; soft keywords such as `match` and `type` are allowed.
- This holds for the accelerator, a buffer, an instruction, an instruction attribute, and a state variable.
- Buffers, instructions, and state variables each have names unique within their kind.
- The predeclared `d0` counts as a buffer, so a buffer named `d0` is a name given twice.
- An instruction attribute is a parameter, so its name is unique within its instruction by Python's rule.
- A Python variable that holds a declared object carries no meaning; the declared name is the one passed to the call.
- The test oracle generator reserves further names for the code it generates; its list lives in its docs, and its error names the word, the generator, and that list.

## S4.2 Accelerator

- `Accelerator(name, base_dir=None, meta=None)` starts an accelerator with one buffer, `d0`, reachable as `acc.d0`.
- `base_dir` is where generators write; by default it is the directory of the file that calls `Accelerator`, or the working directory when there is no such file.
- A `base_dir` given to a generator's entry point replaces the accelerator's for that run; without one the generator writes under the accelerator's `base_dir`.
- `meta` is the accelerator's generator metadata (S12).
- A generator takes the accelerator or the `Model` that `check` returns (S9), as in `generate_oracle(acc)` of the test oracle generator and `generate_backend(acc)` of the compiler backend generator.

## S4.3 Buffers and the hbm buffer `d0`

```python
d1 = acc.buffer('d1', access_dims=[128], base_element_dims=[64], element_type=bf16)
acc_tile = acc.buffer('acc_tile', access_dims=[16], base_element_dims=[16], element_type=f32, initial_value=0.0)
```

- `acc.buffer(name, access_dims, base_element_dims, element_type, initial_value=None, meta=None)` declares a buffer and returns it.
- `access_dims` lists the access dimensions: a region gives one start and one size for each.
- `base_element_dims` is the base element shape: the shape of one base element, which a region always covers whole.
- Each of the two is a list or tuple of plain integers (S6.2), each at least 1, and either may be empty.
- `element_type` is the buffer's element type, one of the objects of S7.3.
- The buffer's full shape is `access_dims` followed by `base_element_dims`; with both empty it is a rank-0 buffer of one element.
- `initial_value` is the buffer's declared initial value: every element holds it before the first issue (S5.6).
- It is one plain literal of the element type, under the rules of a plain `ops.constant` value at that type (S7.5); an integer expression is rejected.
- Without it the buffer holds no value before the first issue, and a read of an element no earlier issue wrote is a fault (S1.3.11).
- An `initial_value` states what the architecture defines at kernel start, such as AMX tiles zeroed by `ldtilecfg`; the front end does not judge it.
- `d0` is hbm: a predeclared `u8` buffer with one access dimension and an empty base element shape.
- The size of `d0`'s access dimension is not declared; it is the byte count of the `hbm` contents a program is run with (S5.6).
- `d0` has no declared initial value: it holds the caller's `hbm` contents in full.
- `d0` carries no generator metadata of its own; an entry about hbm goes in the accelerator's `meta`.

## S4.4 State

```python
calls = acc.state('calls', 0)
```

- A state variable is a named signed 64-bit integer; v3 declares no other kind.
- `acc.state(name, value)` declares a state variable and its value before the first issue, and returns it.
- The value is a plain integer (S6.2), so `True` is rejected, and lies in signed 64 bits.
- A state variable persists from one instruction to the next.
- Inside an instruction function the returned object is an integer expression that reads the state before the issue (S6.1).
- Instructions change it only by state updates (S5.4).

## S4.5 Declaration order

- An ISA specification declares its buffers and state before its first instruction.
- `check` rejects `acc.buffer` and `acc.state` after an instruction (S9).
- After the first `check` that returns, a declaration on that accelerator raises at the author's line, at once (S9).
