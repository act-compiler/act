# Static semantics: operations

One rule per operation family, in the form `V |- op(operands) attrs : t[d*]`.
The reference operator of each word is StableHLO's, or XLA-HLO's where only it has the word (S8.1).
No rule introduces implicit broadcasting or element type promotion: operand types match exactly or the line is ill-typed.
The 75 words of S8.1 are the whole set; `cholesky`, `triangular_solve`, and `batch_norm_inference` are removed in v3 (Q2.8).
The front end checks none of these rules today (S9); act-oracle checks them at generation and at each call (shapes.py:541, oracle_visitor.py:285-293); act-backend checks none (coverage.md:39).
Side conditions marked SHLO are StableHLO's constraints (openxla.org/stablehlo/spec) and are checked against it.

## Type classes

```
F  = {f8e4m3fn, f8e5m2, f16, bf16, f32, f64}
SI = {s2, s4, s8, s16, s32, s64}     UI = {u2, u4, u8, u16, u32, u64}     I = SI + UI
B  = {pred}
```

The classes agree with act-oracle's `TYPES` table (oracle_visitor.py:13-97).

## Inputs and constants

```
parameter(k)              see T-PARAM in typing.md
constant(lit) : t[d*]     lit a number and t in F + I, or lit in {true,false} and t = pred,
                          or lit = `e` and t in I + B; every element is lit at t (S8.4)
                          a number outside t's range is ill-typed (spec_checks.py:94-107 agrees);
                          a float literal is rounded once from its exact decimal, and one that rounds to an infinity is ill-typed
iota() iota_dimension=j : t[d*]      t in F + I, 0 <= j < |d*|
```

## Elementwise

```
V |- %a : t[d*]   t in cls(op)
--------------------------------- UNARY   op in abs, negate, sign, sqrt..atanh, floor..round_nearest_even,
V |- op(%a) : t[d*]                       not, popcnt, count_leading_zeros, is_finite -> pred[d*]

V |- %a : t[d*]   V |- %b : t[d*]   t in cls(op)
------------------------------------------------- BINARY  op in add, subtract, multiply, mulhi, divide,
V |- op(%a, %b) : t[d*]                                   remainder, maximum, minimum, power, atan2,
                                                          and, or, xor, shift_*
cls: arithmetic words F + I (add, multiply, maximum, minimum also B); mulhi, shifts, popcnt I;
     abs and sign F + SI; transcendental and rounding words F; and/or/xor/not I + B
compare(%a, %b) direction=D [type=T] : pred[d*]   %a, %b : t[d*];
     T in {FLOAT, TOTALORDER} for t in F, {SIGNED} for t in SI, {UNSIGNED} for t in UI + B; any other T is ill-typed
     T omitted means FLOAT for t in F, SIGNED for t in SI, UNSIGNED for t in UI + B (oracle_visitor.py:114-118)
select(%p, %a, %b) : t[d*]        %p : pred[d*] or pred[], %a, %b : t[d*]
clamp(%lo, %a, %hi) : t[d*]        %a : t[d*], %lo and %hi : t[d*] or t[]
convert(%a) : t'[d*]               %a : t[d*], any t, t'
bitcast_convert(%a) : t'[d'*]      bits(t) = bits(t')  ==> d'* = d*
                                   bits(t) > bits(t')  ==> d'* = d* ++ [bits(t)/bits(t')]
                                   bits(t) < bits(t')  ==> d* = d'* ++ [bits(t')/bits(t)]   (shapes.py:187-200)
reduce_precision(%a) exponent_bits=e mantissa_bits=m : t[d*]    t in F
```

## Shape and movement

```
reshape(%a) : t[d'*]                        %a : t[d*], prod(d*) = prod(d'*)                     shapes.py:203-208
transpose(%a) dimensions={p*} : t[d'*]      p* a permutation of 0..r-1, d'_j = d_{p_j}
broadcast(%a) dimensions={b*} : t[d'*]      %a : t[d*], |b*| = |d*|, b* distinct and each below |d'*|,
                                            d_j = 1 or d_j = d'_{b_j}                  SHLO broadcast_in_dim C2 to C5
concatenate(%a_1..%a_n) dimensions={j} : t[d'*]   all %a_i : t[d*_i] equal off j, d'_j = sum of d_{i,j}
pad(%a, %v) edge_padding_low=l* edge_padding_high=h* interior_padding=p* : t[d'*]
                                            %v : t[], d'_j = l_j + h_j + d_j + max(d_j - 1, 0) * p_j
reverse(%a) dimensions={j*} : t[d*]
slice(%a) slice={[s_j:l_j:k_j]*} : t[d'*]   0 <= s_j <= l_j <= d_j, k_j >= 1, d'_j = ceil((l_j - s_j) / k_j)
dynamic_slice(%a, %i_1..%i_r) dynamic_slice_sizes={z*} : t[z*]     r = |d*|, 0 <= z_j <= d_j, all %i_j : ti[] with one ti in I
                                                                   starts clamp to 0..d_j - z_j      SHLO dynamic_slice C2 to C5
dynamic_update_slice(%a, %u, %i_1..%i_r) : t[d*]                   %u : t[z*] of the type t, |z*| = |d*|, 0 <= z_j <= d_j,
                                                                   all %i_j : ti[] with one ti in I; starts clamp to 0..d_j - z_j
                                                                   SHLO dynamic_update_slice C1 to C6
gather(%a, %idx) ... : as StableHLO gather                           shapes.py:334-380
copy(%a) : t[d*]                            %a : t[d*]
```

## Linear algebra and reduction

```
dot(%a, %b) lhs_batch_dims=lb* lhs_contracting_dims=lc* rhs_batch_dims=rb* rhs_contracting_dims=rc* : t[d'*]
     %a, %b : t[..], lb and rb same sizes, lc and rc same sizes,
     d'* = batch dims ++ free dims of %a ++ free dims of %b                    shapes.py:393-412
reduce_<r>(%a) dimensions={j*} : t[d'*]     %a : t[d*], d'* = d* without the j*,
     r in {add, multiply, max, min} needs t in F + I (B allowed for add, multiply, max, min by the oracle),
     r in {and, or, xor} needs t in I + B                                       shapes.py:382-391
```

## Result line

```
T-LINE   the type the rule derives equals t[d*] written on the line; else ill-typed (NEW, S8.4)
```
