# Coverage

This page lists each TAIDL language feature and what the oracle does with it today.
Section ids `S<n>` follow the outline of TAIDL's `docs/`; plan items name the fix where a row falls short.
Under owner decision 16, every run-time violation is a defined error the oracle detects and reports, or a defined behavior under one stated norm.
Tensor data never faults: a fault depends only on attributes, state, and region text, and every data-dependent case takes a fixed value.
Under owner decision 40, a generated oracle stops with an error only on invalid assembly code or binary input, and the error names the instruction and the clause.
On valid input it runs correctly: a path that raises or computes a wrong result on valid input is a bug in `plan/bugs.md` (D28-D31), closed by O31.

- `supported`: the oracle computes what the spec says, or detects the violation as a defined error.
- `rejected`: `generate_oracle` raises a clear error before it writes anything; only a problem the architect can correct (a typo, an invalid ISA specification) may be `rejected`.
- `partial`: the notes state the exact gap; a violation the oracle does not detect is always `partial`.
- "At the call" means while a kernel compiles, before it runs, so an error raised at the call writes no region and no state.
- The shipped `docs/coverage.md` keeps only `partial` and `rejected` rows, per S1.2 (O17).

## Declarations, regions, and state

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| Accelerator and `base_dir` | S4.2 | supported | Written to `<base_dir>/targets/<name>/oracle`. |
| Names of the accelerator, data models, instructions, attributes, state | S4.1 | partial | A name the oracle reserves is rejected with `NotImplementedError`; the list is in the oracle's `README.md`; the architect can rename it, so the error must be a `ValueError` naming the name (D30, O6). |
| Data models: addressed dimensions, unit shape, element type | S4.3 | supported | Every element type of S7.3. |
| `hbm` data model `d0` | S4.3 | supported | A kernel's `hbm` argument gives its size in bytes. |
| Buffer contents before the first write | S4.3 | partial | Every on-chip buffer starts zero-filled; a read of an element no earlier instruction of the kernel wrote must be an error, the invalid-assembly kind, if L5 states it as a violation (O26). |
| State and its initial values | S4.4 | partial | Values are exact Python integers; an initial value outside signed 64 bits is accepted with no error or warning. |
| Declaration order | S4.5 | partial | The front end rejects state declared after an instruction but accepts a data model declared after one (L11). |
| Computational and addressing attributes | S5.1 | supported | A call value must be an integer or a loop expression; a loop expression as a computational attribute is invalid assembly code, raised as a bare `TypeError` that names no clause (D31, O31). |
| Negative addressing attributes; a start such as `1024 - @a.addr_out` | S5.2 | supported | No check reads an attribute's sign or name; the region check runs on the evaluated start (`generic/utils.py:704-712`); a test pins it (O24). |
| `@a.` outside a region start or a constraint | S6.1 | partial | Neither rejected nor documented; the oracle reads `@a.` anywhere (L10). |
| Region starts and sizes | S5.2 | supported | A region outside its data model, or a size that is not positive, is an error at the call, at every loop iteration. |
| Two output regions that overlap | S5.3 | partial | Not checked; the kernel check covers only `hbm` layout of inputs and constants; it must be an implicit constraint checked before any load (O10). |
| Same-buffer input-output pair, undeclared | S5.3 | partial | No relation is checked; the default is exact-or-apart: equal starts and sizes in every dimension, or apart, on evaluated terms (O10). |
| Order of loads, stores, and updates within an instruction | S5.4 | partial | Every input loads before any store (`instruction_codegen.py:162-181`); updates run first, so the permitted-overlap value must be the pre-issue semantics (O25, O10). |
| Region value of another element type or count than its data model | S7.2 | supported | Rejected by the oracle at generation; the check moves to the front end through U4 (O7). |
| Constraints, `add_constraints`, `constraints=` | S5.3 | supported | A violated constraint is an error at each call and iteration, read against the state before the call. |
| Constraint arithmetic past 64 bits | S6.2 | partial | Constraints are evaluated exactly with no 64-bit check, so an overflowing constraint can hold; it must raise the constraint error with the overflow named (O24). |
| State updates, `update=`, and execution order | S5.4 | partial | Updates run in order before regions and semantics read the state, so starts read the post-state; every expression must read the pre-state, with updates committed with the outputs (O25). |
| In-place mark: the output reuses the marked input's storage | S5.5 | partial | `set_inplace` is never read; the oracle must check its expansion `same(In k, Out)` and compute from the pre-state; the tie is the backend's (O10). |
| Placement marks (`set_apart`, `set_no_partial_overlap`) | S5.5 | partial | They append constraints, which the oracle checks at each call and iteration; `set_no_partial_overlap` compares starts only, so equal starts with unequal sizes pass (O10). |
| `set_may_overlap`: the empty expansion | S5.5 | partial | Not in the front end; a partial overlap an ISA permits has no spelling until it lands (O10). |
| A failed constraint of any source | S5.3 | partial | One fault at the call, before any load and any write; implicit, marker, and written constraints are one list (O10). |
| Storage that no output region writes | S5.2 | supported | Unchanged: only output regions are stored. |

## Expressions

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| References `@a.`, `@c.`, `@s.` | S6.1 | supported | |
| Integer arithmetic, floor `/` and `%` | S6.2 | supported | Exact Python integers, floor division and the divisor's sign for `%`. |
| A value past signed 64 bits | S6.2 | partial | Warns with `UserWarning` only for an attribute, a region start or size, or a semantics dimension; state-update results and constraints never warn; every intermediate must be a checked 64-bit value, evaluated as parsed (O24). |
| A region start given by a loop expression | S6.2 | partial | Computed in the module in signed 64-bit integers, which wrap, while the region check uses the exact value: valid input runs wrong (D11, D19, O20, else O11). |
| A computed divisor of 0 | S6.2 | supported | An error at the call that names the instruction and iteration. |
| Conditions | S6.3 | supported | |

## Semantics and operations

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| Semantics lines and `ROOT` | S7.1 | supported | `ROOT` placement is a front-end rule. |
| `%In<k>`, `%Out<k>` | S7.2 | supported | Output regions without semantics are rejected with `ValueError`. |
| Presumed reshape | S7.2 | partial | Only where no loop expression gives a region size or a dimension of its value; otherwise a valid presumed reshape raises a `ValueError` at the call (D29, O20, O31). |
| Element types | S7.3 | supported | All 19; a kernel's 2- or 4-bit input, constant or output must fill whole bytes. |
| Shapes with dimensions given by expressions | S7.4 | supported | Checked at the call. |
| Operation attributes: value, brace list, slice list | S7.5 | partial | An expression in a value attribute such as `iota_dimension` or `epsilon` is rejected with `NotImplementedError`, which is a bug when the ISA specification is valid (D28, O31); lists and slice ranges accept expressions. |
| `constant` forms `true`, `false`, `inf`, `-inf`, `nan`, backtick expression | S7.5 | supported | Non-finite values are written by their bits; a backtick value the type cannot hold is an error naming the instruction. |
| Float `constant` literal, and `epsilon` | S8.4 | partial | Read as `f64`, then rounded to its type: two roundings, so a decimal near a tie of a narrower float can round wrong; an integer literal above `2^53` at a float type also rounds twice; it must round once from the exact decimal (O8). |
| Operation words | S8.1 | supported | All 78. |
| Operands and attributes per operation | S8.2 | supported | Checked by the front end. |
| `compare` `type` attribute | S8.2 | supported | The oracle adds the rule that it fits the operand type; the rule moves to the front end through U4 (O7). |
| Element types per operation | S8.3 | partial | A type outside a word's row is rejected with `NotImplementedError`, the class for an oracle gap; a spec error must raise `ValueError` (O6). |
| `dot`, `broadcast`, `pad` | S8.4 | supported | Lowered to `dot_general` with default attributes, `broadcast_in_dim`, and `pad` with a rank-0 padding value. |
| Float `dot` and `reduce` accumulation | S8.4 | partial | A `bf16` dot is emitted at `bf16` and XLA CPU widens it in its own passes; the module must widen narrow floats to `f32` and round once, inside the language's bound with u of the result type (O27, O22). |
| Rounding between lines, denormals, `maximum` and `minimum` of signed zero | S8.4 | partial | `xla_allow_excess_precision`, `xla_cpu_ftz` and `xla_cpu_enable_fast_min_max` are left at their defaults, true (O21). |
| `bitcast_convert` between widths: byte order, NaN bits | S8.4 | partial | XLA's order and NaN bits; the norm is little-endian, and a NaN gives the positive canonical quiet NaN (O22). |
| Float `convert` that overflows | S8.4 | partial | Whatever XLA on the CPU gives; float targets overflow to a signed infinity (O22). |
| Float-to-integer `convert` out of range or NaN | S8.4 | partial | Whatever XLA on the CPU gives; the norm truncates toward zero, saturates, and maps NaN to 0, never a fault (O22, O23). |
| Shift by the width or more | S8.4 | partial | Whatever XLA on the CPU gives; the norm is zero, or sign fill for arithmetic right (O22). |
| Element `divide` or `remainder` by 0, `INT_MIN / -1` | S8.4 | partial | Whatever XLA on the CPU gives; the norm is fixed: `x / 0` is -1 (signed) or all ones (unsigned), `min / -1` is `min`, `x % 0` is `x`, never a fault (O22, O23). |
| `triangular_solve` at `f16` | S8.4 | partial | Solved at `f32` and rounded to `f16` on every platform; whether XLA CPU widens is measured first (O9). |
| Initial value of `reduce_<reducer>` | S8.4 | supported | The reducer's identity at its type; at `f8e4m3fn` max and min start from its largest finite value. |
| Meaning of the other operation words | S8.4 | partial | XLA on the CPU at the pinned jax; no NumPy reference or module-text golden per word yet (O13). |
| Operations outside the language | S8.5 | rejected | Rejected by the front end, which does not lex them. |

## Layers and generator metadata

| Feature | Spec section | Status | Notes |
| --- | --- | --- | --- |
| Width layer: a type `s<w>` or `u<w>` per attribute, default `s64` | panel 10 Q10.1 | partial | Not in the front end yet; the type is an implicit range constraint, checked with the constraint list like any constraint, and a value outside it is the violated-constraint fault (O32). |
| Today's 64-bit check of an attribute value | S1.3.10 | partial | Warns only (`warn_width`, D23); it becomes the `s64` case of the width layer (O24, O32). |
| Region start under a typed attribute | S5.2 | supported | Unchanged: the implicit region constraints bind the evaluated start, never an attribute's type (owner decision 34). |
| Encoding layer: per-instruction encoding block | panel 10 Q10.2 | supported | Ignored: the oracle never reads the block; only a future binary decoder in front of the oracle uses it (O33). |
| Derived attribute: one affine link `E == k*D + c` to an encoded attribute | panel 10 Q10.3 | supported | The oracle sees an ordinary attribute and a written constraint, and checks the link like any constraint; it runs no encode or decode (O33). |
| Binary input | owner decision 40 | partial | No decoder exists; a word that matches no instruction is the decoder's defined error, and the oracle runs on the decoded calls (O33). |
| Generator metadata: namespaced data at accelerator, buffer, instruction | owner decision 39 | supported | Never read, and its absence changes nothing; the oracle reads a metadata entry only when the entry is its own, and none is defined (O33). |
| Failure of a generated oracle | owner decision 40 | partial | Invalid assembly code or binary input stops with an error naming the instruction and the clause; every other path must run correctly, and the audit of today's raise sites is O31. |
