# act-oracle lowering and numerics

The oracle lowers every operation word to StableHLO or CHLO text, and S8.4 gives each word its StableHLO meaning.
StableHLO leaves open dot accumulation, reduce order, integer divide at zero, `convert` out of range and numerical accuracy, so under owner decision 16 each needs a norm the oracle enforces.
The language track owns the norm table; this plan enforces it and removes the flags that let XLA skip rounding.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O4 | One word table: folded into O3 | O3 | -- | -- |
| O9 | `triangular_solve` at `f16`: measure, then widen only where needed | L3 | S | Q1 |
| O21 | Set `xla_allow_excess_precision` false; test per-op rounding | none | S | Q1 |
| O22 | Enforce the numerics norm table: a guard column and a golden per row | L6, O3 | M | Q2 |

## O4 One word table

- Folded into O3: its lowering columns land with the generic-form emitter, its language columns with O7.

## O9 triangular_solve at f16

- Breaking: no.
- Changes: `taidl_to/generic/utils.py`, `taidl_to/generic/decorator.py`; preserves the GPU widening.
- What: `sh_triangular_solve` (`generic/utils.py:498-512`) widens `f16` to `f32` on every platform; on jax CPU a direct `f16` solve works and differs by up to `2^-10`.
- What: the CPU is the reference platform; measure whether XLA CPU already widens, then widen only on GPU and list it as a GPU deviation in `docs/coverage.md` (O14).
- Benefits: CPU results match the reference operation.
- Caveats: the emitter does not know the platform today; `set_simulation_backend` (`generic/decorator.py:18-36`) passes it down.
- Fail points: `tests/test_linalg.py` bounds move by an ulp and fail loudly.
- Order and dependencies: L3 states the meaning; T owns the bound.
- Verification: CPU `f16` results with and without widening on the linalg cases; difference recorded.

## O21 No excess precision

- Breaking: yes, for results XLA computed without intermediate rounding.
- Changes: `taidl_to/generic/decorator.py`, `tests/`; preserves platform choice.
- What: `xla_allow_excess_precision` defaults to true and lets XLA skip rounding between ops; the oracle sets no XLA flag (no `XLA_FLAGS` under `taidl_to/`).
- What: set it false before the backend starts, and test that a `bf16` chain of `exponential`, `reduce_add` and `divide` (`examples/mini_trn/mini_trn.py:101-104`) rounds per op on both platforms.
- Benefits: each line's result is rounded to its type, as the line states.
- Caveats: an XLA flag must be set before the backend initializes, so it is read once per process.
- Fail points: goldens computed with excess precision change; regenerate them in the same commit.
- Order and dependencies: before O19, so the structural gate compares against the fixed baseline.
- Verification: the `bf16` chain test against a per-op NumPy reference with `ml_dtypes`.

## O22 Numerics norm table

- Breaking: yes, where XLA's choice differs from the norm.
- Changes: `runtime/` emitter, the lowering table of O3, `tests/`.
- What: the language track writes one norm per open case; the oracle enforces each row.

| Case | Oracle today | Enforcement |
| --- | --- | --- |
| Float `dot` and `reduce_*` accumulation and order | XLA CPU, `HIGHEST` per dot (`generic/utils.py:603-604`) | As the norm states: a tolerance check, or explicit widening in the emitter |
| Integer `divide`, `remainder` at 0 and `INT_MIN / -1` | XLA CPU value | A `select` guard to the norm's value, or an error flag (O23) |
| Float `convert` out of range | XLA CPU value | A clamp or `select` guard, or an error flag (O23) |
| Shift past the width | XLA CPU value | A `select` guard |
| `bitcast_convert` byte order between widths | XLA's order | Golden test of the stated order |
| Ulp bound per transcendental word | Test bounds in `docs/operations.md` | Bound table in `docs/coverage.md`, re-measured on a jax bump |

- What: the guard is a column of the lowering table, empty for most words.
- Benefits: an element result never depends on XLA's unstated choice.
- Caveats: a differential reference cannot settle a norm, since it makes its own choice; each row needs a golden.
- Fail points: `tests/test_mini_trn.py:91-92` asserts "rounds to bf16 once" against an `f64` product, which an `f32` accumulation over 64 terms does not guarantee; it changes with the norm.
- Order and dependencies: L6 and the language track's norm table first; after O3.
- Verification: one golden test per row, on CPU and on GPU within bounds.
