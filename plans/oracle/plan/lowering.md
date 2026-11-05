# act-oracle lowering and numerics

The oracle lowers every tensor operator to StableHLO or CHLO text, and S8.4 gives each word its StableHLO meaning.
StableHLO leaves open dot accumulation, reduce order, integer divide at zero, `convert` out of range and numerical accuracy, so under owner decision 16 each needs a norm the oracle enforces.
The language track owns the norm table; this plan enforces it and sets the XLA flags that let XLA change a rounded result.
CPU is the normative platform; GPU is held to stated bounds, and each GPU deviation is a row of `docs/coverage.md` (Q5.10).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O4 | One word table: folded into O3 | O3 | -- | -- |
| O9 | `triangular_solve` at `f16`: measure, then widen only where needed | L3 | S | Q1 |
| O21 | XLA flags per compile: no excess precision, no flush to zero, IEEE min and max | none | S | Q1 |
| O27 | Widen narrow float `dot` and `reduce` to `f32` in the module | L6 | S | Q1 |
| O22 | Enforce the numerics norm table: a guard per row and a golden per row | L6 | M | Q1 |

## O4 One word table

- Folded into O3: its lowering columns land with the generic-form emitter, its language columns with O7.

## O9 triangular_solve at f16

- Breaking: no.
- Changes: `taidl_to/generic/utils.py`, `taidl_to/generic/decorator.py`; preserves the GPU widening.
- What: `sh_triangular_solve` (`generic/utils.py:498-512`) widens `f16` to `f32` on every platform; on jax CPU a direct `f16` solve works and differs by up to `2^-10`.
- What: measure whether XLA CPU already widens an `f16` solve before keeping or dropping the widening (Q5.10).
- What: if it does not, widen only on GPU and list it as a GPU deviation in `docs/coverage.md` (O14).
- Benefits: CPU results match the reference operator.
- Caveats: the emitter does not know the platform today; `set_simulation_backend` (`generic/decorator.py:18-36`) passes it down.
- Fail points: `tests/test_linalg.py` bounds move by an ulp and fail loudly.
- Order and dependencies: L3 states the meaning; T owns the bound.
- Verification: CPU `f16` results with and without widening on the linalg cases; difference recorded.

## O21 XLA flags per compile

- Breaking: yes, for results XLA computed without intermediate rounding, flushed, or took from Eigen min and max.
- Changes: `taidl_to/generic/decorator.py`, `tests/`; preserves platform choice.
- What: set `xla_allow_excess_precision` false on the oracle's own compile options, per compile (Q5.10, Q1.6); a process-wide flag is the fallback if the per-compile route fails its test.
- What: set `xla_cpu_ftz` false, since it flushes denormals by default (Q1.6).
- What: set `xla_cpu_enable_fast_min_max` false, so `maximum` and `minimum` follow IEEE 754-2019 (Q1.7).
- What: test a `bf16` chain of `exponential`, `reduce_add` and `divide` (`examples/mini_trn/mini_trn.py:101-104`) against a per-op NumPy reference on CPU and GPU.
- What: test a denormal result per word on the pinned jaxlib, and `maximum(-0, +0)` in both operand orders.
- Benefits: each line's result is rounded to its type, as the line states.
- Fail points: goldens computed with excess precision change; regenerate them in the same commit.
- Order and dependencies: before O19, so the spike measures the fixed baseline.
- Verification: the three tests above, with `ml_dtypes` for the narrow types.

## O27 f32 widening of narrow float dot and reduce

- Breaking: no on CPU, where XLA already widens `bf16`; yes on GPU where it does not.
- Changes: `taidl_to/generic/utils.py` emitters of `dot` and `reduce_*`, `tests/`.
- What: a narrow float `dot` or `reduce` converts its operands to `f32` in the module, accumulates there, and converts once to the result type (Q5.2).
- What: today `DOT_WIDTHS` holds only `i2` and `ui2` (`utils.py:575`), and a `bf16` dot is emitted at `bf16` with `HIGHEST` (`utils.py:597-606`).
- What: the norm is the language's bound `|r - exact| <= gamma_n * sum|terms| + u * |exact|` for `dot` and `reduce_add`, and `gamma_n * |exact|` for `reduce_multiply`, checked in `taidl.testing`; CPU bit-exact goldens are regression tests only.
- What: `gamma_n = n*u / (1 - n*u)` with u the unit roundoff of the result type and n the term count, plus the overflow clause.
- What: the bound holds for every order and every accumulation at least as wide as the result type; a tighter test against the oracle's own `f32` value is a test choice, not the norm.
- Benefits: the oracle's value no longer depends on an XLA pass.
- Caveats: the bound is vacuous when `n*u >= 1` (`bf16`, n >= 256); the ISA author writes a wider result type and converts.
- Fail points: a CPU golden moves; that means XLA did not widen as `cpu_compiler.cc` shows, and the module text diff names the word.
- Order and dependencies: after L6; in the fidelity step, before the spike (Q5.9).
- Verification: goldens unchanged on CPU; `tests/test_mini_trn.py:51-55,91-92` within the bound against an `f64` reference.

## O22 Numerics norm table

- Breaking: yes, where XLA's choice differs from the norm.
- Changes: the `sh_*` emitters today, the lowering table of O3 later, `tests/`.
- What: the language track writes one norm per open case; the oracle enforces each row with a guard in today's emitter.
- What: O3 later carries each guard as a column of the lowering table, empty for most words.

| Case | Oracle today | Norm and enforcement |
| --- | --- | --- |
| Float `dot` and `reduce_*` accumulation | XLA CPU, `HIGHEST` per dot (`generic/utils.py:603-604`) | `f32` widening (O27) inside the language's result-type bound |
| Integer `divide`, `remainder` at 0 and `INT_MIN / -1` | XLA CPU value | Fixed by `select`: `x / 0` is -1 (signed) or all ones (unsigned), `min / -1` is `min`, `x % 0` is `x`; never a fault (O23) |
| Float to integer `convert` out of range or NaN | XLA CPU value | Fixed by `select`: truncate toward zero, saturate, NaN to 0; never a fault (O23) |
| Float to float and integer to float `convert` overflow | XLA CPU value | Signed infinity, or NaN at `f8e4m3fn` (Q1.8); golden test |
| Shift past the width | XLA CPU value | `select` guard: zero, or sign fill for arithmetic right (Q5.4) |
| `bitcast_convert` byte order between widths | XLA's order | Little-endian, one golden per type pair (Q5.4) |
| `bitcast_convert` of a NaN from a float source | XLA's bits | `select` to the positive canonical quiet NaN (Q1.7) |
| Ulp bound per transcendental word | Test bounds in `docs/operations.md` | Bound table in `docs/coverage.md`; special values exact (Q1.7) |

- What: a jax bump regenerates and diffs every golden and remeasures `ULPS` (Q5.10).
- Benefits: an element result never depends on XLA's unstated choice.
- Caveats: a differential reference cannot settle a norm, since it makes its own choice; each row needs a golden.
- Fail points: a guard that skips padded lanes inside a region; every element is guarded.
- Order and dependencies: after L6 and the language track's norm table; in the fidelity step, before the spike (Q5.9).
- Verification: one golden test per row, on CPU and on GPU within bounds.
