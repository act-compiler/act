# Coverage

This page lists each TAIDL language feature and whether the oracle supports it.
The language itself is specified in [TAIDL's `docs/`](https://github.com/act-compiler/taidl/tree/master/docs); section ids `S<n>` follow its outline.
A feature is `supported` when the oracle computes what the spec says, `rejected` when `generate_oracle` raises a clear error before it writes anything, and `partial` when the notes state the exact limitation.
What the oracle adds beyond the language, the reference operation each operation word lowers to and the test bounds, is in `docs/operations.md`.

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| Accelerator and `base_dir` | S4.2 | supported | The oracle is written to `<base_dir>/targets/<name>/oracle`. |
| Names of the accelerator, data models, instructions, attributes, state variables | S4.1 | partial | A name the oracle reserves is rejected with `NotImplementedError`; the list is in `README.md`, "The names an oracle reserves". |
| Data models: addressed dimensions, unit shape, element type | S4.3 | supported | Every element type of S7.3. |
| Off-chip memory `d0` | S4.3 | supported | A kernel's `hbm` argument gives its size in bytes. |
| State variables and initial values | S4.4 | supported | Values are Python integers, exact at any width. |
| Declaration order | S4.5 | supported | Checked by the front end. |
| Computational and addressing attributes | S5.1 | supported | A value passed at a call must be an integer or a loop expression; a loop expression as a computational attribute is rejected at the call with `TypeError`. |
| Input and output regions: starts and sizes | S5.2 | supported | A window outside its data model, or a size that is not positive, is rejected at the call, at every loop iteration. |
| Constraints | S5.3 | supported | Checked at each call and each loop iteration against the state before the call. |
| State updates and execution order | S5.4 | supported | Updates run in order before regions and semantics read the state; every loop around such a call is unrolled. |
| In-place marks (`set_inplace`) | S5.5 | partial | Ignored: an input region marked in place keeps its contents unless an output region writes it. |
| Placement marks (`set_apart`, `set_no_partial_overlap`) | S5.5 | supported | They append constraints, which the oracle checks. |
| References `@a.`, `@c.`, `@s.` | S6.1 | supported | |
| Integer arithmetic, floor `/` and `%` | S6.2 | partial | A result outside the signed 64-bit range is computed exactly and warns with `UserWarning`; a start given by a loop expression is computed in signed 64-bit integers, which wrap. |
| Conditions | S6.3 | supported | |
| Semantics lines and `ROOT` | S7.1 | supported | |
| Input and output values `%In<k>`, `%Out<k>` | S7.2 | supported | An instruction with output regions and no semantics is rejected with `ValueError`. |
| Presumed reshape | S7.2 | partial | Supported only where no loop expression gives a size of the region or a dimension of its value; otherwise a shape mismatch is rejected at the call with `ValueError`. |
| Element types | S7.3 | supported | All 19; a kernel's input, constant or output of a 2- or 4-bit type must fill whole bytes. |
| Shapes, with dimensions given by expressions | S7.4 | supported | A dimension given by an expression is checked at the call. |
| Operation attributes: value, brace list, slice list | S7.5 | partial | An expression in a value attribute, such as `iota_dimension` or `epsilon`, is rejected with `NotImplementedError`; expressions in lists and slice ranges are supported. |
| Operation words | S8.1 | supported | All 78. |
| Operands and attributes per operation | S8.2 | supported | Checked by the front end. |
| Element types per operation | S8.3 | supported | A type outside a word's row is rejected with `NotImplementedError`. |
| Meaning of `triangular_solve` at `f16` | S8.4 | partial | Solved at `f32` and rounded to `f16` on every platform. |
| Meaning of a float `constant` literal | S8.4 | partial | The literal is read as an `f64` first and then rounded to its type, so a decimal within an `f64` rounding of a tie of a narrower float can round to the wrong neighbour. |
| Initial value of `reduce_<reducer>` | S8.4 | supported | Each reducer starts from its identity at its type, a float sum from `+0.0`; at `f8e4m3fn`, which has no infinity, `reduce_max` and `reduce_min` start from its largest finite value. |
| Meaning of the other operation words | S8.4 | supported | As the reference operation in XLA on the CPU; the test bounds are in `docs/operations.md`. |
| Operations outside the language | S8.5 | rejected | Rejected by the front end, which does not lex them. |
