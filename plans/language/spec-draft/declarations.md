# Declarations

This page is S4 of the TAIDL language specification.
An ISA specification builds an accelerator by calls in a fixed order: buffers and state first, then instructions.

## S4.1 Names

- Every name an ISA specification gives is a `str` that is an identifier: an ASCII letter or `_`, then ASCII letters, digits, and `_`.
- A name is not in `keyword.kwlist` of the Python version the front end runs on; soft keywords such as `match` and `type` are allowed.
- This holds for the accelerator, a buffer, an instruction, an instruction attribute, and a state variable.
- Buffers, instructions, and state variables each have names unique within their kind.
- An instruction attribute is a parameter, so its name is unique within its instruction by Python's rule.
- A Python variable that holds a declared object carries no meaning; the declared name is the one passed to the call.
- The test oracle generator reserves further names for the code it generates; its list lives in its docs, and its error names the word, the generator, and that list.

## S4.2 Accelerator

- `Accelerator(name, base_dir=None, meta=None)` starts an accelerator with one buffer, `d0`, reachable as `acc.d0`.
- `base_dir` is where generators write; by default it is the directory of the file that calls `Accelerator`, or the working directory when there is no such file.
- `meta` is the accelerator's generator metadata (S12).
- A generator takes the accelerator or the `Model` that `check` returns (S9), as in `generate_oracle(acc)` of the test oracle generator and `generate_backend(acc)` of the compiler backend generator.

## S4.3 Buffers and the hbm buffer `d0`

```python
d1 = acc.buffer('d1', access_dims=[128], base_element_dims=[64], element_type=bf16)
```

- `acc.buffer(name, access_dims, base_element_dims, element_type, meta=None)` declares a buffer and returns it.
- `access_dims` lists the access dimensions: a region gives one start and one size for each.
- `base_element_dims` is the base element shape: the shape of one base element, which a region always covers whole.
- Each of the two is a list or tuple of values of type exactly `int`, each at least 1, and either may be empty.
- `element_type` is the buffer's element type, one of the objects of S7.3.
- The buffer's full shape is `access_dims` followed by `base_element_dims`; with both empty it is a rank-0 buffer of one element.
- `d0` is hbm: a predeclared `u8` buffer with one access dimension and an empty base element shape.
- The size of `d0`'s access dimension is not declared; it is the byte count of the `hbm` contents a program is run with (S5.6).

## S4.4 State

```python
calls = acc.state('calls', 0)
```

- State is a named value of a declared kind, an integer by default.
- `acc.state(name, value)` declares a state variable and its value before the first issue, and returns it.
- The value has type exactly `int`, so `True` is rejected, and lies in signed 64 bits.
- A state variable holds a signed 64-bit integer that persists from one instruction to the next.
- Inside an instruction function the returned object is an integer expression that reads the state before the issue (S6.1).
- Instructions change it only by state updates (S5.4).

## S4.5 Declaration order

- An ISA specification declares its buffers and state before its first instruction.
- `check` rejects `acc.buffer` and `acc.state` after an instruction (S9).
