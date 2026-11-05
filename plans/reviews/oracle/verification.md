# Verification of the act-oracle track

Scope: every row of `act-oracle/docs/coverage.md` (C1 to C29, in file order) and `work/oracle/plan/bugs.md` (D1 to D20).
Verdicts are confirmed, wrong (wholly or in part), or unverifiable.
Paths are abbreviated: `OV` is `act-oracle/taidl_to/oracle_visitor.py`, `U` is `taidl_to/generic/utils.py`, `IC` is `taidl_to/instruction_codegen.py`, `T` is `taidl_to/templates/templates.txt`.
`FE` is `taidl/taidl/`, `SPEC` is `taidl/docs/`.
Experiments ran read-only in the work venv (jax 0.11.2, CPU, no bytecode, no files written).

## Coverage rows

| Row | Claim | Verdict | Evidence (file:line) |
| --- | --- | --- | --- |
| C1 | S4.2 oracle written to `<base_dir>/targets/<name>/oracle` | confirmed | `taidl_to/generator.py:22,27,31-32,87` |
| C2 | S4.1 reserved name rejected with `NotImplementedError`, list in README | confirmed | `generator.py:40-43`; `api_generator.py:57-60`; `semantic_generator.py:29`; `IC:202`; `README.md:73` |
| C3 | S4.3 data models, every S7.3 type | confirmed | `U:325-346` (19 entries); `U:851-870` (19 dtypes) |
| C4 | S4.3 `d0`, kernel `hbm` is a size in bytes | confirmed | `decorator.py:70-72,118-122` |
| C5 | S4.4 state values are exact Python integers | confirmed | `api_generator.py:43-44` (`str(variable.value)`); no width check on an initial value |
| C6 | S4.5 declaration order "checked by the front end" | confirmed, incomplete | `FE accelerator.py:46-53` rejects state after an instruction; `accelerator.py:33-44` accepts a data model after one (`SPEC conformance.md` "Not rejected") |
| C7 | S5.1 call values int or loop expression; loop expression as computational attribute is `TypeError` | confirmed | `U:276-292` |
| C8 | S5.2 window outside model, size not positive, rejected at every iteration | confirmed | `U:651-666`, `U:698-712`; `T:324-332` |
| C9 | S5.3 constraints checked per call and iteration against the pre-call state | confirmed | `IC:32-44`; `T:353-360`; `IC:169-175` (constraints precede updates) |
| C10 | S5.4 updates in order before regions and semantics; enclosing loops unrolled | confirmed | `IC:169-181`; `T:98-104` |
| C11 | S5.5 `set_inplace` ignored | confirmed | grep of `inplace` in `taidl_to/`: no hit; only `FE instruction.py:163-172` |
| C12 | S5.5 placement marks append constraints the oracle checks | confirmed | `FE instruction.py:190-231`; `IC:36`; `tests/test_constraints.py:41,100-108` |
| C13 | S6.1 `@a.`, `@c.`, `@s.` | confirmed | `expression_writer.py:6,81` |
| C14 | S6.2 floor `/` `%`; result past 64 bits computed exactly and warns; loop-expression start wraps in i64 | wrong in part | floor and exact: `expression_writer.py:21` (`//`, Python `%`). The warning is narrower: only attribute, region start, region size or semantics dimension (`U:287,659,706`; `README.md:134-139`). State-update results (`T:363`, `bound` at `U:724`) and constraint arithmetic (`T:353-360`) never warn. Wrap confirmed: `U:214-250` emits `tensor<i64>` ops; jax CPU gives `3*2^62+2^62 -> 0` |
| C15 | S6.3 conditions | confirmed | `expression_writer.py:45-63` |
| C16 | S7.1 semantics lines and `ROOT` | confirmed (shallow) | `semantic_generator.py:22,58-59`; `OV:223-299`; `ROOT` itself is a front-end rule |
| C17 | S7.2 outputs without semantics rejected with `ValueError` | confirmed | `semantic_generator.py:14-19`; `tests/test_refusals.py:92` |
| C18 | S7.2 presumed reshape only where no loop sizes it; else `ValueError` | confirmed | `U:733-747` (`looped or prod` test) |
| C19 | S7.3 all 19 types; 2- and 4-bit kernel values fill whole bytes | confirmed | `U:325-346`; `decorator.py:96-100`; `U:874` (`PACKED`) |
| C20 | S7.4 expression dimensions checked at the call | confirmed | `IC:116-144`; `U:651-666` |
| C21 | S7.5 expression in a value attribute is `NotImplementedError`; lists and slices accept expressions | confirmed | `OV:412-418`, `OV:423-435` |
| C22 | S8.1 "All 78" words | confirmed | `OV:19-97` (77 keys) plus `parameter` at `OV:234`; `taidl/antlr4/IDLV2.g4:45` lexes 78 |
| C23 | S8.2 checked by the front end | confirmed | `FE operations.py` `OPERATIONS` has 78 entries |
| C24 | S8.3 type outside a row is `NotImplementedError` | confirmed | `OV:254-258`; `SPEC operations.md` S8.3 calls it undefined, so any rejection conforms |
| C25 | S8.4 `triangular_solve` at `f16` solved at `f32`, on every platform | confirmed | `U:498-511`, no platform branch; `tests/test_linalg.py:149-163`; `README.md:151` |
| C26 | S8.4 float literal read as `f64` then rounded to its type | confirmed | `OV:383`, `U:408-409`; see D1; `docs/operations.md:103` says "rounds to the nearest value" and contradicts it |
| C27 | S8.4 reduce identities, `f8e4m3fn` max/min from largest finite | confirmed | `U:439-451`; `U:383-390` (448 at `f8e4m3fn`) |
| C28 | S8.4 other words mean the XLA reference; bounds in `docs/operations.md` | unverifiable | lowering table `OV:19-97`, `docs/operations.md:15-95` exist; no XLA comparison run |
| C29 | S8.5 operations outside the language rejected by the front end, not lexed | confirmed | `IDLV2.g4:45` lists only the 78 words; `ragged_dot`, `convolution`, `scatter` and the rest of S8.5 are absent |

## Bug rows

| Row | Claim | Verdict | Evidence (file:line) |
| --- | --- | --- | --- |
| D1 | Float literal rounded twice; `1.00048828125000000001` at `f16` gives `1.0` | confirmed | `OV:383` (`float(text)`), `OV:389`, `U:408-409`. Python: the `f64` equals `1+2^-11` exactly, `struct` `e` packs it to `1.0`, `Fraction` shows the decimal is above the tie. MLIR parse of the emitted text: `1.0004882812500000` -> `1.000000e+00`, `1.0009765625` -> `1.000980e+00` |
| D2 | Range check compares the rounded `f64` with the bound | confirmed | `U:404-406` (`abs(value) > bound`) |
| D3 | `epsilon` takes the same `float(text)` path | confirmed | `OV:161-168` (`number = float(text)`, `mlir_literal(number, 'f32', ...)`) |
| D4 | `f16` solve widens to `f32` on every platform; CPU differs from an `f16` solve | confirmed for code and CPU; GPU part unverifiable | `U:498-511`. jax CPU 8x8 random: direct `f16` solve works and differs from the `f32` path by up to `2^-10`. "Only the GPU backend fails" cannot be tested without a GPU (asserted at `README.md:151`) |
| D5 | Type outside an S8.3 row raised as `NotImplementedError` | confirmed | `OV:254-258`. Spec: undefined, front end does not check, so the class is a design choice |
| D6 | Non-integer index of the three index words, same class | confirmed | `OV:259-264`; `INDICES` at `OV:121` |
| D7 | `TYPES` copies S8.3 row by row, all equal | confirmed | `OV:19-97` (the file says 19-98; line 98 is blank). A script compared all 77 words with the S8.3 table: 0 differences, 78 words with `parameter` |
| D8 | Missing-from-`TYPES` branch unreachable | confirmed | `OV:237-241`; 78 lexed words = 77 + `parameter` (`OV:234-235`) |
| D9 | Attributes visited twice with `holding` toggled | confirmed | `OV:266-274` |
| D10 | `set_inplace` never read; premise that L must say what an in-place input holds | code confirmed, premise stale | grep as C11. `SPEC instructions.md` S5.5: an in-place mark does not change the computed result; the rest is implementation-defined. C11 could read `supported` |
| D11 | Loop-expression start wraps in i64; the window check is exact; `/` or `%` after an overflowing step can diverge | confirmed by reading | `README.md:141-142`; `U:214-250` (i64 ops); `U:704-707` (exact `evaluate`); i64 wrap shown on jax CPU. A divergent trace was not built |
| D12 | Log line names the final directory while copying into staging | confirmed | `generator.py:62-65` |
| D13 | `sh_copy` `ValueError` names no instruction | text confirmed, path unreachable | `U:551-557`. A copy type mismatch is refused earlier with the instruction named: `shapes.py:98-103,136-140,501`; `tests/test_refusals.py:135`. The `U:554-555` check is dead |
| D14 | Setting a platform sets jax globals process-wide | confirmed | `decorator.py:22-23`; `U:603-604` (`HIGHEST`) |
| D15 | Only the first operand's type is checked | confirmed | `OV:251-254`; `dot` is `EVERY` at `OV:34`, so moot |
| D16 | Sequential `str.replace` can re-substitute | confirmed (mechanism); "latent" unverifiable | `template.py:41-42` |
| D17 | Checks enumerate the product of all loop trip counts | wrong in part | `iterations` enumerates only the loop variables the value reads (`U:68-72`), not all loops. The cited `U:82-86` is the constant-fold branch of `divide`; the product is at `U:71`. Callers confirmed at `U:657,704,739,758,777,792` |
| D18 | `layout` re-implements ruff splitting, about 95 lines | confirmed; fix not free | `template.py:61-156`; `tests/conftest.py:278-282` runs a ruff format check on every generated oracle |
| D19 | Floored remainder adds the divisor, overflowing for divisors past `2^62` | confirmed | `U:241-243`. Example: `a=2^62`, `b=2^62+1`: `t=a`, `t+b=2^63+1` wraps, result and operands in range |
| D20 | Coefficient past 64 bits fails in the MLIR parser, no instruction named | confirmed | `U:223` (no range check). Parse of `dense<2^64> : tensor<i64>`: "integer constant out of range for type", no instruction |

## Numeric claims, summary

- Double rounding of float literals (D1, D2, D3, C26): confirmed, with the MLIR step run directly.
- Same path rounds an integer literal above `2^53` at a float type twice (`U:408` calls `float(value)` on an `int`); no row mentions it.
- Wrapping i64 loop starts (C14, D11, D19, D20): wrap confirmed; the divergence trace in D11 is by reading only.
- `triangular_solve` at `f16` (C25, D4): code and CPU difference confirmed; the GPU failure is not testable here.

## Features of the TAIDL docs that coverage.md misses

- S1.2 status vocabulary: the spec allows `partial` and `rejected` only; coverage.md adds `supported` and lists fully supported features.
- S1.3 and S8.4 byte order of `bitcast_convert` between widths: implementation-defined, not listed; only the sub-byte packing is documented (`docs/operations.md:14`).
- S1.3 and S8.4 float `convert` that overflows: implementation-defined, not listed.
- S4.4 initial value outside signed 64 bits: undefined; the oracle stores it exactly with no warning (`api_generator.py:43-44`), unlike C14's warning claim.
- S5.2 two overlapping output regions: undefined; the oracle has no check (grep `overlap`: only the kernel layout at `decorator.py:138`).
- S6.1 and S9 an addressing attribute outside a start or constraint: undefined and not rejected by the front end; the oracle neither rejects nor documents it (`expression_writer.py:6`, `expression_writer.py:100` reads `@a.` anywhere).
- S6.2 a constraint whose arithmetic overflows "does not hold": the oracle evaluates constraints exactly (`T:353-360`), so such a constraint can hold; the row does not say.
- S6.2 a computed divisor of 0: the oracle raises `ValueError` naming the iteration (`U:78-80`, `T:359-360`); not in C14.
- S7.2 region value of another element type or count than its model: rejected by the oracle (`semantic_generator.py:44-56`); not listed.
- S7.5 `constant` operand forms: `true`, `false`, `inf`, `-inf`, `nan` and a backtick expression (`OV:376-380`); only the float literal has a row.
- S8.2 compare `type` attribute: the oracle adds a rule that it fit the operand type (`OV:277-283`, `COMPARE_TYPES` at `OV:114`); not listed.
- S8.4 `dot` as `dot_general` with attribute defaults, `broadcast` as `broadcast_in_dim`, `pad` rank-0 second operand: covered only by the generic C28 row.
- S5.1 `add_constraints` and the `update=`, `constraints=` keywords: no row.
