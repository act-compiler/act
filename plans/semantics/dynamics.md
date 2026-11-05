# Dynamic semantics

An instruction is a partial function from a machine state, its attribute values, and its operands to a new machine state.
Where the function is undefined the issue is rejected; no case is left undefined.

## Machine state

```
Mem   = {m -> (Z^|A_m| x Z^|U_m|) -> Val_{t_m}}      one array per data model; d0 : Z -> u8
Reg   = {s -> i64}                                  control registers (state variables)
sigma = (Mem, Reg)
Attr  = {x -> i64}  for x in C_i + P_i               each attribute an i64 at issue (instructions.md:123)
```

## Expression evaluation

```
[[k]]rho           = k
[[@c.x]]rho        = rho(x)        [[@a.x]]rho = rho(x)        [[@s.x]]rho = rho.Reg(x)
[[e1 + e2]]rho     = [[e1]]rho + [[e2]]rho  in Z, then REQUIRE in i64
[[e1 / e2]]rho     = floor(a / b)      with b = [[e2]]rho, REQUIRE b != 0        expressions.md:199
[[e1 % e2]]rho     = a - b * floor(a / b)                                        expressions.md:200
[[e1 cmp e2]]rho   = the comparison in Z
[[c1 and c2]]rho   = both hold;  [[c1 or c2]]rho = either holds
```

- REQUIRE failing anywhere in an expression makes the whole instruction issue ill-formed: the issue is rejected (holes.md H4, H5).
- A constraint whose evaluation fails REQUIRE does not hold (coverage.md act-backend row 28, isel_ir2isa_rewrites_rs_generator.py:452-457).
- Both generators compute `/` and `%` this way: act-backend `quotient` and `remainder` (templates quotient.txt, remainder.txt), act-oracle Python `//` and `%` (expression_writer.py:479-481).

## Issue of an instruction

```
issue(i, attrs, sigma) = sigma'   where, with rho_0 = (attrs, sigma.Reg):

1. CONSTRAINTS   for every K_j:  [[K_j]]rho_0 holds, else REJECT               instructions.md:153
2. UPDATES       Reg_1 = Reg_0;  for W_j = (@s.x = e) in order:  Reg_1[x] := [[e]](attrs, Reg_1)   instructions.md:154
3. REGIONS       rho = (attrs, Reg_1)   (see holes.md H3 for the alternative rho_0)
                 for each region R = region(m, e*, f*):  s_j = [[e_j]]rho,  n_j = [[f_j]]rho
                 REQUIRE n_j >= 1 and 0 <= s_j and s_j + n_j <= A_m[j]  (d0: s + n <= |d0|)   instructions.md:137
4. READ          for each In_k:  in_k = Mem_0[m](s* + idx, u*) for idx < n*, u* < U_m,
                 then reshaped row-major to the dims of %In<k>                   semantics.md:44-46
5. COMPUTE       V = eval(B, rho, in*)  by the operation semantics below
6. WRITE         REQUIRE the output regions are pairwise disjoint                instructions.md:138
                 Mem_1 = Mem_0;  for each Out_k:  Mem_1[m](s* + idx, u*) := V(%Out<k>) reshaped row-major
7. RESULT        sigma' = (Mem_1, Reg_1)
```

- Every input is read from `Mem_0`, so an input that overlaps an output sees the pre-state (read before write).
- act-oracle follows this order (instruction_codegen.py:732-737) and writes with `dynamic_update_slice` on the pre-state buffer (utils.py:831-847).
- act-backend models no state (spec_checks.py:562-577) and takes the one output as fresh storage unless `set_inplace` says otherwise (inplace_fix.rs:52-74).

## Operation semantics

```
eval(ENTRY { L_1..L_p }, rho, in*) = V_p   where V_0 = {} and
V_j = V_{j-1} + {%v_j -> [[op_j]](V_{j-1}(o*), [[attrs_j]]rho)}
[[parameter(k)]]         = in_k
[[constant(lit)]] : t[d*] = fill(d*, round_t(lit))         round_t: nearest-even from the exact decimal
[[op]] for every other word = the reference operation on the operand tensors (operation_meaning.md:89)
```

Numeric classes fix where "the reference operation" is exact and where it is a tolerance class:

| class | words | value |
| --- | --- | --- |
| integer exact | add, subtract, multiply, negate, abs, mulhi, and, or, xor, not, shifts, popcnt, count_leading_zeros, compare, select, clamp, maximum, minimum, convert between I and B, bitcast_convert | two's complement, wraparound at bits(t); shift by >= bits(t) gives 0 or the sign fill |
| integer divide | divide, remainder at t in I | truncation toward zero; x / 0 = -1 for SI and all ones for UI; min / -1 = min; x % 0 = x (XLA's values) |
| float exact | add, subtract, multiply, divide, sqrt, negate, abs, convert, compare, maximum, minimum, floor, ceil, round_*, reduce_precision at t in F | IEEE-754 round to nearest even at t; NaN propagates as a quiet NaN; denormals kept |
| float approximate | exponential, log, logistic, erf, sine, ..., power, cbrt, rsqrt, cholesky, triangular_solve, batch_norm_inference, dot and reduce_add and reduce_multiply at t in F | any value within the tolerance the spec states per word; the oracle's test bounds (act-oracle docs/operations.md) are the first draft |
| reduction | reduce_<r> | starts from the identity of r at t (utils.py:439-451); integer reducers exact and order-free; float order implementation-defined inside the tolerance |

## Determinism

An issue is deterministic, that is `issue` is a function, when every step above is total:

- every expression satisfies REQUIRE, so no value depends on an overflow or a zero divisor;
- every region is in bounds and the outputs are disjoint, so no write order is observable;
- every operation is in the integer exact or float exact class, or its tolerance is tight enough for the comparison the consumer makes.

Two generators agree on a program iff they agree on every exact-class value bit for bit and on every approximate-class value within the stated tolerance.
act-oracle today compares with per-word bounds (coverage.md act-oracle row 37); act-backend computes no values and is judged by the shapes and attributes it matches (coverage.md act-backend rows 39, 46).
