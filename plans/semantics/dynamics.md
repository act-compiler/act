# Dynamic semantics

An instruction is a function from a state, its attribute values, and its operands to a new state, or to a fault.
Where the function faults, the issue writes no region and no state; no case is left undefined.

## State

```
Mem   = {m -> (Z^|A_m| x Z^|U_m|) -> Val_{t_m}}      one array per data model; d0 : Z -> u8
St    = {s -> i64}                                  state variables
Wr    = {m -> (Z^|A_m| -> bool)}                    written set per on-chip buffer; d0 is written in full
sigma = (Mem, St, Wr)
Attr  = {x -> i64}  for x in C_i + P_i               each attribute an i64 at issue (S5.6)
```

- `Wr` records the elements some earlier issue of the kernel wrote; it is tracked on the host from region integers.
- A kernel starts with `Wr` empty for every on-chip buffer; `d0` holds the caller's `hbm` contents.

## Expression evaluation

```
[[k]]rho           = k
[[@c.x]]rho        = rho(x)        [[@a.x]]rho = rho(x)        [[@s.x]]rho = rho.St(x)
[[e1 op e2]]rho    = a op b for op in + - *, FAULT unless the result is in i64
[[e1 / e2]]rho     = floor(a / b)      with b = [[e2]]rho, FAULT if b = 0 or the result leaves i64    S6.2
[[e1 % e2]]rho     = a - b * floor(a / b)                                                       S6.2
[[e1 cmp e2]]rho   = the comparison in Z
[[c1 and c2]]rho   = both hold;  [[c1 or c2]]rho = either holds
```

- Every operator of an expression yields a signed 64-bit integer, or the issue faults; the check walks the tree as written, in the grammar's left-associative order.
- A fault of an expression is a fault of the issue (holes.md H4, H5).
- Both generators compute `/` and `%` this way: act-backend `quotient` and `remainder` (templates quotient.txt, remainder.txt), act-oracle Python `//` and `%` (expression_writer.py:15-24).

## Issue of an instruction

```
issue(i, attrs, sigma) = sigma'   where rho_0 = (attrs, St_0) is the only environment of the issue:

1. CONSTRAINTS   for every K_j:  [[K_j]]rho_0 holds, else FAULT (every REQUIRE failure is a fault)    S5.3
2. REGIONS       for each region R = region(m, e*, f*):  s_j = [[e_j]]rho_0,  n_j = [[f_j]]rho_0
                 REQUIRE n_j >= 1 and 0 <= s_j and s_j + n_j <= A_m[j]  (d0: s + n <= |d0|)         S5.2
3. READ          for each In_k:  in_k = Mem_0[m](s* + idx, u*) for idx < n*, u* < U_m,
                 REQUIRE every element is in Wr_0, then reshape row-major to the dims of %In<k>      S7.2
4. COMPUTE       V = eval(B, rho_0, in*)  by the operation semantics below
5. COMMIT        outputs and updates take effect together:
                 St_1[x] := [[e]]rho_0 for every W_j = (@s.x = e)     a parallel assignment
                 Mem_1[m](s* + idx, u*) := V(%Out<k>) reshaped row-major; Wr_1 marks those elements
6. RESULT        sigma' = (Mem_1, St_1, Wr_1)
```

- Every expression of an issue reads the state before the issue; its updates and its outputs take effect together at its end.
- A chained update is written by substitution: the value of a state variable after an update is the update text in place of the variable.
- Every fault is decided on host integers before any write, so a fault writes no region and no state.
- A program stops at its first fault, and every earlier issue has committed; the report names the lowest issue index, and within one issue any applicable clause id may be named.
- Faults depend only on attributes, state, and region text; every tensor value gives a fixed result.
- A read of an element no earlier issue of the kernel wrote is a fault; a stale read of a reused buffer is not caught.
- Output regions are pairwise apart, an implicit constraint; a failed constraint of any source (implicit, marker, written) is the one fault of S1.3.5.
- Each same-buffer input-output pair has one relation: tied (`set_inplace`), apart (`set_apart`), exact-or-apart (default and `set_no_partial_overlap`), or may overlap (`set_may_overlap`); a marker expands to constraints (owner decision 30).
- `apart(R, S)`: some dimension has `R.start + R.size <= S.start` or `S.start + S.size <= R.start`; `same(R, S)`: equal starts and equal sizes in every dimension; all terms are evaluated (owner decision 34).
- Under any permitted overlap the semantics runs on the pre-issue state: all inputs load before any output stores.
- The backend is a producer: it never emits an issue that faults, and lists each precondition it cannot see (state before the first issue, `|d0|`) in its coverage doc.
- act-oracle writes with `dynamic_update_slice` (utils.py:831-847); act-backend models no state (spec_checks.py:370-386).

## Operation semantics

```
eval(ENTRY { L_1..L_p }, rho, in*) = V_p   where V_0 = {} and
V_j = V_{j-1} + {%v_j -> [[op_j]](V_{j-1}(o*), [[attrs_j]]rho)}
[[parameter(k)]]         = in_k
[[constant(lit)]] : t[d*] = fill(d*, round_t(lit))   round_t: nearest-even once from the exact decimal
[[op]] for every other word = the reference operator on the operand tensors (S8.4)
```

- A float literal rounds once to nearest even; a finite literal that rounds to an infinity (a NaN at `f8e4m3fn`) is rejected, and underflow rounds normally.
- A backtick constant outside its element type faults at issue.

Numeric classes fix where "the reference operator" is exact and where it meets a bound:

Every one of the 75 words of S8.1 stands in the table below, once for each type class its row of S8.2 accepts.

| class | at | words | value |
| --- | --- | --- | --- |
| movement | every type | parameter, select, reshape, transpose, broadcast, concatenate, pad, reverse, slice, dynamic_slice, dynamic_update_slice, gather, copy | element bits unchanged; start indices of dynamic_slice, dynamic_update_slice, and gather clamp as StableHLO states |
| literal | every type | constant, iota | constant: `round_t` of the exact decimal, once; iota: the index as an exact value of t |
| integer exact | I and pred | add, subtract, multiply, negate, abs, sign, mulhi, and, or, xor, not, shift_left, shift_right_logical, shift_right_arithmetic, popcnt, count_leading_zeros, compare, maximum, minimum, clamp, power, dot, reduce_add, reduce_multiply, reduce_max, reduce_min, reduce_and, reduce_or, reduce_xor | two's complement, wraparound at bits(t), `pred` as u1; shift by >= bits(t) gives 0 or the sign fill; `dot` and the reductions are exact sums or products in that arithmetic, so order-free; the integer edge cases below |
| integer divide | I | divide, remainder | truncation toward zero; x / 0 = -1 for SI and all ones for UI; min / -1 = min; x % 0 = x; min % -1 = 0 (XLA's values, lowered explicitly) |
| float exact | F | add, subtract, multiply, divide, remainder, negate, abs, sign, sqrt, floor, ceil, round_nearest_afz, round_nearest_even, reduce_precision, is_finite, compare | every operation rounds once to its result type (nearest even); denormals kept; `remainder` takes the dividend's sign and is exact |
| special values | F | maximum, minimum, clamp, reduce_max, reduce_min | IEEE 754-2019 `maximum` and `minimum`: NaN propagates, -0 < +0 |
| convert | every type | convert | float to float and int to float: nearest even, overflow to a signed infinity (NaN at `f8e4m3fn`); float to int: truncate toward zero, saturate, NaN to 0; int to int: wrap; to `pred`: nonzero, NaN included, is true |
| bit cast | every type | bitcast_convert | the bits reinterpreted little-endian (H14); a float NaN gives the canonical quiet NaN bits |
| approximate | F | power, rsqrt, cbrt, exponential, exponential_minus_one, log, log_plus_one, logistic, erf, sine, cosine, tan, asin, acos, atan2, sinh, cosh, tanh, asinh, acosh, atanh | within the per-word tolerance the spec states, with inf, zero, and NaN results matching the IEEE function exactly; the oracle's test bounds (act-oracle docs/operations.md) are the first draft |
| accumulating | F | dot, reduce_add, reduce_multiply | the bounds below |

- A reduction `reduce_<r>` starts from the identity of r at t (utils.py:439-451); a float sum starts from +0.0.
- Integer `power` with an exponent below 0 gives 1 for base 1, 1 or -1 by the parity of the exponent for base -1, and 0 for every other base, 0 included; the oracle checks these values against XLA CPU and lowers them explicitly where XLA differs.
- Integer `dot` is the sum of the products in the wraparound arithmetic of the result type.

- NaN: any NaN equals any NaN; sign and payload are outside the meaning, except that `bitcast_convert` of a float NaN gives the positive canonical quiet NaN bits of that type.
- `dot` and `reduce_add`: `|r - exact| <= gamma_n * sum|terms| + u * |exact|`.
- `reduce_multiply`: `|r - exact| <= gamma_n * |exact|`.
- `gamma_n = n*u / (1 - n*u)`, u the unit roundoff of the result type, n the term count (K for `dot`, the product of the reduced sizes for a reduce).
- Overflow clause: when `sum|terms|` exceeds the largest finite value of the type, the result may also be an infinity or a NaN.
- An ISA that needs a wider accumulator writes a wider result type or explicit `convert` lines.

## Determinism

An issue is a function, with a defined fault where it has no value, when every step above is total:

- every expression satisfies its REQUIRE, so no value depends on an overflow or a zero divisor;
- every region is in bounds and every read element was written;
- every operation is in an exact class, or its bound is tight enough for the comparison the consumer makes.

Two generators agree on a program iff they agree on every exact-class value bit for bit and on every bounded value within its bound.
act-oracle compares with per-word bounds (coverage.md act-oracle row 37); act-backend computes no values and is judged by the shapes and attributes it matches (coverage.md act-backend rows 39, 46).
