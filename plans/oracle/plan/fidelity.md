# act-oracle semantics fidelity

An oracle is only useful if its result is the spec's result, so a silent deviation is worse than a rejection.
This plan fixes the known deviations, detects data-dependent errors inside the module, and adds a differential reference.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O8 | Round float literals once, from the decimal | L16, T | S | Q1 |
| O10 | In-place mark: the output reuses the marked input's storage | L5 | S | Q1 |
| O11 | Reject a loop-expression start whose 64-bit steps overflow; only if O20 does not land | O19 | S | Q1 |
| O12 | Drop the matmul-precision flag; transfer buffers without `jax.device_put` | none | S | Q2 |
| O13 | Differential reference over the instruction record | O2, T | M | Q2 |
| O23 | Detect data-dependent errors inside the module | O3, O22 | M | Q2 |

## O8 Float literals rounded once

- Breaking: no.
- Changes: `taidl_to/oracle_visitor.py`, `taidl_to/generic/utils.py`; preserves the range checks and the non-finite path.
- What: `constant_value` reads a literal with `float(text)` (`oracle_visitor.py:383`) and `mlir_literal` emits `repr(float(value))` (`generic/utils.py:408-409`), so a narrow float rounds twice.
- What: `1.00048828125000000001` at `f16` becomes the `f64` tie `1 + 2^-11`, then `1.0`; the nearest `f16` is `1.0009765625`.
- What: an integer literal above `2^53` at a float type takes the same path (`utils.py:408`).
- What: round `fractions.Fraction(text)` once to the target type and emit its bits in hex, as non-finite values are (`utils.py:411-420`); same for `epsilon` (`oracle_visitor.py:161-168`) and the range check (`utils.py:400-406`).
- Benefits: literals match S8.4 at every float type.
- Caveats: S8.4 states no rounding rule; L16 must say "nearest, ties to even, from the decimal".
- Fail points: `f8e4m3fn` has no infinity; reuse `float_largest` (`utils.py:383-390`).
- Order and dependencies: before O19; T adds the tie cases.
- Verification: tie-adjacent decimals at `f16`, `bf16`, `f8e5m2`, `f32`, against exact rational rounding.

## O10 In-place marks

- Breaking: yes.
- Changes: `taidl_to/instruction_codegen.py`, `runtime/` checks.
- What: owner decision 17: the output reuses the marked input's storage; nothing under `taidl_to/` reads `set_inplace` today.
- What: store the paired output into the marked input's region, and raise at the call when the two regions differ in element count or type.
- What: every input loads before any output stores (`instruction_codegen.py:169-181`), so reads see the values before the call.
- Benefits: an in-place kernel reads what the hardware leaves in that storage.
- Caveats: `set_inplace` takes one flag per input (`taidl/instruction.py:163-172`); L5 states which output pairs with it.
- Fail points: kernels in `examples/` that read a marked input after the call change their goldens.
- Order and dependencies: L5 first.
- Verification: tests read a marked input after the call and assert it holds the output.

## O11 Wrapping loop-expression starts

- Breaking: no.
- Changes: `taidl_to/generic/utils.py`; preserves the exact region check.
- What: only if O19 rejects unrolling; otherwise O20 deletes the code this item fixes.
- What: a start given by a loop expression is computed in signed 64-bit integers in the module (`utils.py:214-250`), while the region check uses the exact value; reject at the call when any step leaves 64 bits.
- What: also D19 (floored `/` and `%` add the divisor, `utils.py:241-243`) and D20 (a coefficient past 64 bits, `utils.py:223`).
- Fail points: a false rejection; property-test against exact arithmetic.
- Verification: an overflowing intermediate with an in-range final start is rejected.

## O12 Drop the jax configuration

- Breaking: yes.
- Changes: `taidl_to/generic/decorator.py`; preserves platform choice and its errors.
- What: drop `jax_default_matmul_precision` (`decorator.py:23`): the module carries `HIGHEST` per dot (`generic/utils.py:603-604`).
- What: x64 is needed only for `jax.device_put` (`decorator.py:188`); hand buffers to the compiled module without it, then drop `jax_enable_x64` (`decorator.py:22`).
- Benefits: an oracle no longer changes a caller's own jax numerics.
- Fail points: an `s64` or `f64` value truncated to 32 bits; the `f64` and `s64` cases in `tests/test_elementwise.py` fail loudly.
- Verification: a test runs caller jax code after a kernel and asserts x64 is off.

## O13 Differential reference

- Breaking: no.
- Changes: `tests/`, a reference module beside them; preserves every existing test.
- What: factor the NumPy references the tests already hold (`tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`) into one module over the O2 record, with `ml_dtypes` for narrow types.
- What: parse literals with `Fraction`, since NumPy's string parse also rounds through `f64`.
- What: the StableHLO reference interpreter, which evaluates the emitted module text itself, is the option for words the NumPy module lacks; it is a new dependency and needs owner approval first.
- What: a jax version bump regenerates every golden and reruns this comparison.
- Benefits: catches the next D1-class deviation.
- Caveats: compare exactly only where a norm (O22) fixes the order; elsewhere within the bound table.
- Order and dependencies: after O2; T owns the harness.
- Verification: every S8.1 word has a reference or is listed as a gap.

## O23 Data-dependent errors in the module

- Breaking: yes.
- Changes: `runtime/` emitter and `runtime/decorator.py`.
- What: every check runs before tracing (`generic/utils.py:651-806`), so a violation that depends on element values, such as an integer `divide` by zero, cannot be reported today.
- What: for each O22 row whose norm is an error, the module computes a flag (`compare`, `reduce_or`); the oracle raises after the run, naming the instruction.
- What: the kernel commits its outputs only when no flag is set, so an error writes no region and no state.
- Caveats: the flag adds outputs to the module; measure compile and run time.
- Fail points: a flag reduced over the wrong axis misses an error; one test per row with a single bad element.
- Order and dependencies: after O3 and O22.
- Verification: each error row has a test that sets one bad element and expects the error.
