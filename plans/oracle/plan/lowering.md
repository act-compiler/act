# act-oracle lowering strategy

The oracle lowers every operation word to StableHLO or CHLO text, and the word's facts are spread across six tables in `taidl_to/oracle_visitor.py`.
This plan keeps text emission and folds those tables into one, so a word is added or changed in one row.
Lowering through `jax.lax` instead was argued and rejected in the architecture debate.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O4 | One word table: types, dialect, emitter, reducer | O5 | S | Q1 |
| O9 | `triangular_solve` at `f16`: decide and narrow the widening | L3 | S | Q1 |

## O4 One word table

- Breaking: no.
- Changes: `taidl_to/oracle_visitor.py`; preserves every row of the six tables, merged.
- What: merge `TYPES` (`oracle_visitor.py:19-98`), `PRETTY` (line 100), `REDUCERS` (103-111), `INDICES` (121), `CHLO` (124) and `op_to_template` (208-221) into one dict keyed by word.
- What: each row gives the emitter and the dialect; the types column holds `TYPES` as it is until O6 replaces it.
- Benefits: six lookups become one; the unreachable branch at `oracle_visitor.py:237-241` disappears by construction.
- Caveats: `PRETTY` changes the MLIR form, not the meaning; keep it as a row field so emitted text stays identical.
- Fail points: a row with the wrong dialect fails at MLIR parse with a jax error that names no instruction; the per-word tests in `tests/test_elementwise.py` and `tests/test_linalg.py` run every word.
- Order and dependencies: after O5; before O3; O6 later replaces the types column with the front end's table.
- Verification: generated `api.py` byte-identical for both examples; full suite green.

## O9 triangular_solve at f16

- Breaking: no.
- Changes: `taidl_to/generic/utils.py`, `taidl_to/generic/decorator.py`; preserves the GPU widening.
- What: `sh_triangular_solve` (`taidl_to/generic/utils.py:498-512`) widens `f16` to `f32` on every platform because XLA's GPU backend fails at `f16`.
- What: ask the L track whether S8.4 defines `triangular_solve` at `f16` as XLA's `f16` solve; if so, widen only when the platform is GPU and record the GPU deviation in `docs/coverage.md`.
- Benefits: CPU results match the reference operation; the deviation is stated once, where coverage lives.
- Caveats: the emitter does not know the platform at emit time today; `set_simulation_backend` in `generic/decorator.py:18-36` would have to pass it down.
- Caveats: XLA's own `f16` CPU solve may itself upcast; measure before changing.
- Fail points: CPU `f16` results move by an ulp and `tests/test_linalg.py` bounds may need adjusting; that test fails loudly.
- Order and dependencies: L3 decides the meaning first; T owns the bound.
- Verification: compare CPU `f16` results with and without widening on the existing linalg cases; document the measured difference.
