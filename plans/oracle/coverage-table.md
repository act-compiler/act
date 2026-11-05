# Coverage

This page lists each TAIDL language feature and what the oracle does with it today.
Section ids `S<n>` follow the outline of TAIDL's `docs/`; plan items name the fix where a row falls short.
Under owner decision 16, every run-time violation is a defined error the oracle detects and reports, or a defined behavior under one stated norm.

- `supported`: the oracle computes what the spec says, or detects the violation as a defined error.
- `rejected`: `generate_oracle` raises a clear error before it writes anything.
- `partial`: the notes state the exact gap; a violation the oracle does not detect is always `partial`.
- "At the call" means while a kernel compiles, before it runs, so an error raised at the call writes no region and no state.
- The shipped `docs/coverage.md` keeps only `partial` and `rejected` rows, per S1.2 (O17).

## Declarations, regions, and state

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| Accelerator and `base_dir` | S4.2 | supported | Written to `<base_dir>/targets/<name>/oracle`. |
| Names of the accelerator, data models, instructions, attributes, state | S4.1 | partial | A name the oracle reserves is rejected with `NotImplementedError`; the list is in the oracle's `README.md`. |
| Data models: addressed dimensions, unit shape, element type | S4.3 | supported | Every element type of S7.3. |
| `hbm` data model `d0` | S4.3 | supported | A kernel's `hbm` argument gives its size in bytes. |
| Buffer contents before the first write | S4.3 | supported | Every on-chip buffer starts zero-filled. |
| State and its initial values | S4.4 | partial | Values are exact Python integers; an initial value outside signed 64 bits is accepted with no error or warning. |
| Declaration order | S4.5 | partial | The front end rejects state declared after an instruction but accepts a data model declared after one (L11). |
| Computational and addressing attributes | S5.1 | supported | A call value must be an integer or a loop expression; a loop expression as a computational attribute is a `TypeError` at the call. |
| `@a.` outside a region start or a constraint | S6.1 | partial | Neither rejected nor documented; the oracle reads `@a.` anywhere (L10). |
| Region starts and sizes | S5.2 | supported | A region outside its data model, or a size that is not positive, is an error at the call, at every loop iteration. |
| Two output regions that overlap | S5.2 | partial | Not checked; the kernel check covers only `hbm` layout of inputs and constants (O24). |
| Region value of another element type or count than its data model | S7.2 | supported | Rejected by the oracle at generation; the check moves to the front end (O7). |
| Constraints, `add_constraints`, `constraints=` | S5.3 | supported | A violated constraint is an error at each call and iteration, read against the state before the call. |
| Constraint arithmetic past 64 bits | S6.2 | partial | Constraints are evaluated exactly with no 64-bit check, so an overflowing constraint can hold, while S6.2 says it does not hold (O24). |
| State updates, `update=`, and execution order | S5.4 | supported | Updates run in order before regions and semantics read the state; every loop around such a call is unrolled. |
| In-place mark: the output reuses the marked input's storage | S5.5 | partial | `set_inplace` is never read; the output is stored into its own region, so storage is reused only when the two regions coincide; every input loads before any output stores (O10). |
| Placement marks (`set_apart`, `set_no_partial_overlap`) | S5.5 | supported | They append constraints, which the oracle checks at each call and iteration. |
| Storage that no output region writes | S5.2 | supported | Unchanged: only output regions are stored. |

## Expressions

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| References `@a.`, `@c.`, `@s.` | S6.1 | supported | |
| Integer arithmetic, floor `/` and `%` | S6.2 | supported | Exact Python integers, floor division and the divisor's sign for `%`. |
| A value past signed 64 bits | S6.2 | partial | Warns with `UserWarning` only for an attribute, a region start or size, or a semantics dimension; state-update results and constraints never warn; decision 16 needs each detected as an error (O24). |
| A region start given by a loop expression | S6.2 | partial | Computed in the module in signed 64-bit integers, which wrap, while the region check uses the exact value (O20, else O11). |
| A computed divisor of 0 | S6.2 | supported | An error at the call that names the instruction and iteration. |
| Conditions | S6.3 | supported | |

## Semantics and operations

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| Semantics lines and `ROOT` | S7.1 | supported | `ROOT` placement is a front-end rule. |
| `%In<k>`, `%Out<k>` | S7.2 | supported | Output regions without semantics are rejected with `ValueError`. |
| Presumed reshape | S7.2 | partial | Only where no loop expression gives a region size or a dimension of its value; otherwise a shape mismatch is a `ValueError` at the call. |
| Element types | S7.3 | supported | All 19; a kernel's 2- or 4-bit input, constant or output must fill whole bytes. |
| Shapes with dimensions given by expressions | S7.4 | supported | Checked at the call. |
| Operation attributes: value, brace list, slice list | S7.5 | partial | An expression in a value attribute such as `iota_dimension` or `epsilon` is rejected with `NotImplementedError`; lists and slice ranges accept expressions. |
| `constant` forms `true`, `false`, `inf`, `-inf`, `nan`, backtick expression | S7.5 | supported | Non-finite values are written by their bits; a backtick value the type cannot hold is an error naming the instruction. |
| Float `constant` literal, and `epsilon` | S8.4 | partial | Read as `f64`, then rounded to its type: two roundings, so a decimal near a tie of a narrower float can round wrong; an integer literal above `2^53` at a float type also rounds twice (O8). |
| Operation words | S8.1 | supported | All 78. |
| Operands and attributes per operation | S8.2 | supported | Checked by the front end. |
| `compare` `type` attribute | S8.2 | supported | The oracle adds the rule that it fits the operand type; the rule moves to the front end (O7). |
| Element types per operation | S8.3 | partial | A type outside a word's row is rejected with `NotImplementedError`, the class for an oracle gap, not a spec error (O6). |
| `dot`, `broadcast`, `pad` | S8.4 | supported | Lowered to `dot_general` with default attributes, `broadcast_in_dim`, and `pad` with a rank-0 padding value. |
| Float `dot` and `reduce` accumulation | S8.4 | partial | XLA's CPU choice, no norm stated or checked; `xla_allow_excess_precision` is left on (O21, O22). |
| `bitcast_convert` between widths: byte order | S8.4 | partial | XLA's order, not stated; needs a norm from L and a golden test (O22). |
| Float `convert` that overflows | S8.4 | partial | Whatever XLA on the CPU gives; needs a norm or a defined error (O22). |
| Element `divide` or `remainder` by 0, `INT_MIN / -1` | S8.4 | partial | Whatever XLA on the CPU gives; data-dependent, so an error needs detection inside the module (O22, O23). |
| `triangular_solve` at `f16` | S8.4 | partial | Solved at `f32` and rounded to `f16` on every platform (O9). |
| Initial value of `reduce_<reducer>` | S8.4 | supported | The reducer's identity at its type; at `f8e4m3fn` max and min start from its largest finite value. |
| Meaning of the other operation words | S8.4 | partial | XLA on the CPU at the pinned jax; no differential run against a reference yet (O13). |
| Operations outside the language | S8.5 | rejected | Rejected by the front end, which does not lex them. |
