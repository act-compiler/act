# act-oracle semantics fidelity

An oracle is only useful if its result is the spec's result, so a silent deviation is worse than a rejection.
This plan orders the fixes for the silent deviations found this session and adds a differential test so new ones are caught.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O8 | Round float literals once, from the decimal | L16, T | S | Q1 |
| O10 | Define and implement `set_inplace` | L5 | S | Q1 |
| O11 | Reject a loop-expression start whose 64-bit steps overflow | L6 | S | Q1 |
| O12 | Scope jax configuration to the oracle | none | M | Q2 |
| O13 | NumPy reference interpreter for differential tests | O2, T | L | Q2 |

## O8 Float literals rounded once

- Breaking: no.
- Changes: `taidl_to/oracle_visitor.py`, `taidl_to/generic/utils.py`; preserves the range checks and the non-finite path.
- What: `constant_value` reads a literal with `float(text)` (`taidl_to/oracle_visitor.py:383`) and `mlir_literal` emits `repr(float(value))` (`generic/utils.py:408-409`), so a narrow float is rounded twice.
- What: example: `1.00048828125000000001` at `f16` becomes the `f64` tie `1 + 2^-11`, then rounds to even, `1.0`; the nearest `f16` is `1.0009765625`.
- What: round `fractions.Fraction(text)` once to the target format and emit its bits in hex, the path non-finite values already take (`utils.py:413-420`); same for `epsilon` (`oracle_visitor.py:161-168`) and the range check (`utils.py:400-406`).
- Benefits: literals match S8.4 exactly at every float type.
- Caveats: `f64` literals are unaffected; the fix matters for `f32` and narrower only.
- Fail points: hex emission for `f8e4m3fn` must respect its missing infinity; reuse `float_largest` (`utils.py:383-390`).
- Order and dependencies: independent; T adds the tie cases.
- Verification: a test with tie-adjacent decimals at `f16`, `bf16`, `f8e5m2` and `f32` compares against `numpy` with exact rational rounding.

## O10 In-place marks

- Breaking: yes, if S5.5 makes an in-place input unspecified.
- Changes: `taidl_to/instruction_codegen.py`; preserves region loads and stores.
- What: nothing under `taidl_to/` reads `set_inplace`; an input marked in place keeps its contents unless an output writes it.
- What: ask L what S5.5 says an in-place input holds after the call: unchanged, unspecified, or the output.
- Benefits: removes an unstated assumption; `docs/coverage.md` already lists it as `partial`.
- Caveats: if the answer is "unspecified", an oracle that poisons it (NaN fill) catches kernels that read it, which is a behavior change.
- Fail points: kernels in `examples/` may read an in-place input after the call; their goldens would change.
- Order and dependencies: L5 decides first.
- Verification: a test that reads a marked input after the call asserts the defined result.

## O11 Wrapping loop-expression starts

- Breaking: no.
- Changes: `taidl_to/generic/utils.py`; preserves the exact-value window check.
- What: per `README.md:141-142`, a start given by a loop expression is computed in signed 64-bit integers in the module, while the window check uses the exact value.
- What: with `/` or `%` after an overflowing step, the traced start and the checked start differ.
- What: reject at the call when any step of such a start leaves the 64-bit range, at any iteration.
- What: if a kernel's compile time is measured past an agreed threshold, add an affine fast path that checks starts and sizes at loop bounds only (D17).
- Benefits: no silent mismatch between checked and executed address.
- Caveats: needs the step values, which the loop algebra in `generic/utils.py:18-150` computes symbolically; check per iteration as `check_window` does.
- Fail points: a false rejection of a start that never overflows; property-test against exact arithmetic.
- Order and dependencies: after L6 states the overflow rule.
- Verification: a test with an overflowing intermediate and a final in-range start is rejected with `ValueError`.

## O12 Scoped jax configuration

- Breaking: yes.
- Changes: `taidl_to/generic/decorator.py`; preserves platform choice and its errors.
- What: `set_simulation_backend` sets `jax_enable_x64` and `jax_default_matmul_precision` for the whole process (`generic/decorator.py:22-23`).
- What: scope x64 to the oracle's compile and run, and rely on the per-dot `HIGHEST` the emitter already writes (`generic/utils.py:604`).
- Benefits: an oracle no longer changes a caller's own jax numerics.
- Caveats: jax's x64 context manager is experimental; it may not cover compiled-function calls.
- Fail points: an `s64` or `f64` value silently truncated to 32 bits outside the scope; the `f64` and `s64` cases in `tests/test_elementwise.py` would fail.
- Order and dependencies: independent; Q2 because of the jax API risk.
- Verification: a test runs caller jax code after a kernel and asserts x64 is off; suite green.

## O13 Reference interpreter

- Breaking: no.
- Changes: `tests/`, new interpreter module beside them; preserves every existing test.
- What: a NumPy interpreter over the line record of O2: one exact function per operation word, regions read and written from NumPy arrays, no jax.
- What: tests run each instruction through both the oracle and the interpreter on random inputs and compare within the bounds of `docs/operations.md`.
- Benefits: catches the next D1-class deviation; documents the meaning of each word in executable form.
- Caveats: NumPy lacks some types (`bf16`, `f8`, sub-byte) natively; use `ml_dtypes`, which jax already depends on, so no new dependency.
- Caveats: the interpreter can share the oracle's bug if it reuses its tables; it must read the spec, not the oracle's code.
- Fail points: tolerance tuning turns into noise; keep exact comparison for every non-transcendental word.
- Order and dependencies: after O2; T owns the harness.
- Verification: every word in S8.1 has an interpreter function and a differential test.
