# act-oracle semantics fidelity

An oracle is only useful if its result is the spec's result, so a silent deviation is worse than a rejection.
This plan fixes the known deviations, makes state reads follow the spec, checks overlap constraints, fixes data-dependent values, and adds a differential reference.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O8 | Round float literals once, from the exact value received | L16, T | S | Q1 |
| O25 | Expressions read the pre-state; updates commit with the outputs | L5 | S | Q1 |
| O10 | Overlap constraints and in-place marks: check the list before any load, value from the pre-state | L5, O25 | S | Q1 |
| O11 | Fallback only if O19 fails: keep `while` with an equivalence test; reject wrapping starts | O19 | S | Q1 |
| O12 | Drop the matmul-precision flag; drop x64 after a feed test | none | S | Q2 |
| O13 | Differential reference: NumPy over the instruction record, module-text goldens | O2, T | M | Q2 |
| O23 | Fixed values for integer divide, remainder and float-to-integer convert; no data faults | O22 | S | Q1 |
| O30 | Overlap test matrix: declarations by placements | O10, T | S | Q1 |

## O8 Float literals rounded once

- Breaking: no.
- Changes: `taidl_to/oracle_visitor.py`, `taidl_to/generic/utils.py`; preserves the range checks and the non-finite path.
- What: `constant_value` reads a literal with `float(text)` (`oracle_visitor.py:383`) and `mlir_literal` emits `repr(float(value))` (`generic/utils.py:408-409`), so a narrow float rounds twice.
- What: `1.00048828125000000001` at `f16` becomes the `f64` tie `1 + 2^-11`, then `1.0`; the nearest `f16` is `1.0009765625`.
- What: an integer literal above `2^53` at a float type takes the same path (`utils.py:408`).
- What: take the exact value received, `fractions.Fraction(value)` of a `float`, an `int`, or a `Fraction`, never through `float()`; round once to nearest even, and emit the bits in hex, as non-finite values are (`utils.py:411-420`) (Q5.5, Q1.10, Q13b.9).
- What: the range check compares the exact value with the largest finite value plus half an ulp (`utils.py:400-406`); a literal that rounds to infinity is rejected.
- What: the same path serves `epsilon` (`oracle_visitor.py:161-168`).
- Fail points: `f8e4m3fn` has no infinity; reuse `float_largest` (`utils.py:383-390`).
- Order and dependencies: L16 states the rule; T adds the tie cases; before O20.
- Verification: tie-adjacent decimals at `f16`, `bf16`, `f8e5m2`, `f32`, against exact rational rounding.

## O25 Pre-state reads

- Breaking: yes.
- Changes: `taidl_to/instruction_codegen.py`, `tests/test_state.py`.
- What: every expression of an instruction reads the state before the instruction; its updates and its outputs take effect together at its end (Q1.1).
- What: today constraints, updates, sizes, inputs, semantics and outputs are emitted in that order (`instruction_codegen.py:162-181`), so starts read the post-state.
- What: emit the updates after the outputs, as one parallel assignment.
- Benefits: a guard and the write it guards read the same state.
- Fail points: `tests/test_state.py:24-28` changes its expected start; update it in the same commit.
- Order and dependencies: L5 states the rule; in the fidelity step, before O19.
- Verification: a test whose output start reads a state its update changes.

## O10 Overlap constraints and in-place marks

- Breaking: yes.
- Changes: `taidl_to/instruction_codegen.py`, `runtime/` checks, `tests/`.
- What: the front end emits one constraint list per instruction: bounds, output-output apart, one relation per same-buffer input-output pair, and written constraints (Q9.2, Q9.4).
- What: the oracle checks every constraint of that list, whether implicit, marker, or written, at the call and at each iteration, before any load (S1.3.5).
- What: a failed constraint is the one fault; it writes no region and no state (Q9.3).
- What: the value under any permitted overlap is the instruction's semantics on the pre-issue state; the oracle needs no code per relation (Q9.4).
- What: the oracle loads every input before any store (`instruction_codegen.py:169-181`) and commits updates with the outputs (O25).
- What: the checks run on evaluated start and size terms, so texts that differ and evaluate equal satisfy `same` (owner decision 34).
- What: `set_inplace`, `set_apart`, `set_no_partial_overlap` and `set_may_overlap` reach the oracle only as their expansions in the list; the tie is for the backend, and nothing under `taidl_to/` reads `set_inplace` today.
- What: the undeclared default of a same-buffer pair is exact-or-apart, with exact meaning equal starts and equal sizes in every dimension (Q9.3).
- What: rejection of a cross-buffer tie (owner decision 24), unequal size text on a tie, `set_apart` on identical text, and identical input and output text without `set_inplace` is the front end's; the oracle keeps no copy.
- Caveats: markers address the single output; `set_may_overlap` is a working name.
- Fail points: the check placed after a load or on the post-state; a partial overlap that passes because only starts are compared.
- Order and dependencies: L5 and O25 first.
- Verification: the matrix of O30.

## O11 Loop fallback

- Breaking: no.
- Changes: `taidl_to/generic/utils.py`, `tests/`; preserves the exact region check.
- What: only if O19 fails its bound; otherwise O20 deletes the code this item fixes.
- What: keep `while` emission, with a test that the `while` and unrolled paths agree on every loop case (Q5.1).
- What: a start given by a loop expression is computed in signed 64-bit integers in the module (`utils.py:214-250`); reject at the call when any step leaves 64 bits.
- What: also D19 (floored `/` and `%` add the divisor, `utils.py:241-243`) and D20 (a coefficient past 64 bits, `utils.py:223`).
- Fail points: a false rejection; property-test against exact arithmetic.
- Verification: the equivalence test green; an overflowing intermediate with an in-range final start is rejected.

## O12 Drop the jax configuration

- Breaking: yes.
- Changes: `taidl_to/generic/decorator.py`; preserves platform choice and its errors.
- What: drop `jax_default_matmul_precision` (`decorator.py:23`): the module carries `HIGHEST` per dot (`generic/utils.py:603-604`).
- What: drop `jax_enable_x64` (`decorator.py:22`) only after a test with `f64` and `i64` arguments, or a bytes feed, shows no silent downcast through `jax.device_put` (`decorator.py:188`) (Q5.10).
- Benefits: an oracle no longer changes a caller's own jax numerics.
- Fail points: an `s64` or `f64` value truncated to 32 bits; the feed test fails loudly.
- Verification: the feed test; a test runs caller jax code after a kernel and asserts x64 is off.

## O13 Differential reference

- Breaking: no.
- Changes: `tests/`, a reference module beside them; preserves every existing test.
- What: factor the NumPy references the tests already hold (`tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`) into one module over the O2 record, with `ml_dtypes` for narrow types (Q5.7).
- What: literals in the reference are exact `Fraction` values, since NumPy's string parse also rounds through `f64`.
- What: golden module-text tests per word and per norm row pin the lowering.
- What: the StableHLO interpreter is not added (owner decision 28).
- Benefits: catches the next D1-class deviation, which is a lowering bug an interpreter of the emitted text would pass.
- Caveats: compare exactly only where a norm (O22) fixes the order; elsewhere within the bound table.
- Order and dependencies: after O2; T owns the harness.
- Verification: every S8.1 word has a reference or is listed as a gap.

## O23 Fixed values for data-dependent cases

- Breaking: yes, where XLA's CPU value differs.
- Changes: `taidl_to/generic/utils.py` emitters of `divide`, `remainder` and `convert`, `tests/`; preserves `runtime/decorator.py`.
- What: tensor data never faults: a fault depends only on attributes, state and region text (joint decision of panels 1 and 5).
- What: integer `divide` by 0 is -1 (signed) or all ones (unsigned), `min / -1` is `min`, and `x % 0` is `x`, each lowered as an explicit `select`.
- What: float-to-integer `convert` truncates toward zero, saturates at the target range, and maps NaN to 0.
- What: a saturating or trapping ISA writes its own `select` or `clamp` in its semantics.
- What: every element is guarded, including zero divisors inside a padded region; there is no flag and no host check.
- What: a host expression such as `/` by 0 in an addressing value still faults (O24).
- What: optional and non-normative: an opt-in diagnostic reports a zero divisor or a NaN conversion without changing any result.
- Benefits: a correct compile of a legal kernel never raises on data, so the backend's closed-loop test stays satisfiable.
- Fail points: a guard that changes a non-edge element; one test per row with the edge element next to ordinary ones.
- Order and dependencies: after O22.
- Verification: each row against its fixed value at signed and unsigned types, and a state-free check that no run raises.

## O30 Overlap test matrix

- Breaking: no.
- Changes: `tests/`; preserves every existing test.
- What: declarations (none, `set_apart`, `set_no_partial_overlap`, `set_inplace`, `set_may_overlap`) by evaluated placements (apart, exact, partial), each expecting the pre-state value or the one fault (Q9.4).
- What: a fault test asserts that no region and no state changed.
- What: F5 row: equal starts and unequal sizes fault under the default and pass under `set_may_overlap`.
- What: owner decision 34 row: starts differing in text and equal in value, with a negative attribute, pass under the default.
- What: written rows: a positional constraint and an input-input constraint.
- What: cross-check: each marker and its hand-written expansion give the same fault set over a swept attribute range.
- What: an output-output overlap row for the implicit apart constraint.
- What: region-bounds row (S1.3.3): an evaluated start below 0, a size below 1, and a start plus size past the access dimension each fault, while a negative attribute value with an in-bounds start passes.
- Order and dependencies: after O10; T owns the harness.
- Verification: every cell of the matrix has a test or is listed as a gap.
