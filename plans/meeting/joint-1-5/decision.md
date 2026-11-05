# Joint moderator decision, panels 1 and 5

Owner decisions 36 and 37 state no preference; this file rules on evidence and not on the count of positions.
Binding inputs: decisions 16, 27, 31, 34.

## Point A: element-level violations in tensor data (Q1.8, Q1.9 against Q5.4)

Decision: Panel 1.
- Tensor data never faults: a fault depends only on attributes, state, and region text.
- Integer `divide` and `remainder` by zero, `INT_MIN / -1`, and float-to-integer `convert` out of range or NaN take fixed values.
- Divide and remainder values are XLA's, lowered explicitly in the oracle: `x / 0` is -1 (signed) or all ones (unsigned), `min / -1` is `min`, `x % 0` is `x` (dynamics.md integer divide row).
- Float-to-integer `convert` truncates toward zero, saturates, and maps NaN to 0 (Q1.8); a saturating or trapping ISA writes its own `select` or `clamp` in the semantics.
- Panel 5's in-module error flag is not adopted; its other Q5.4 rows (shift guards, little-endian `bitcast_convert`, `compare` default) are not in dispute and stand.

Reasoning chain:
- 1. Decision 16 is met by both options, so it does not decide; decisions 27 and 31 decide.
- 2. The backend input is an XLA-HLO file with no tensor values (act-backend README lines 16 and 57-60), so it cannot prove a divisor nonzero or a float in range.
- 3. Under errors, a correct compile of a legal kernel can raise on data, and Q1.12's "the backend never emits a faulting issue" and its closed-loop test become unsatisfiable.
- 4. XLA and StableHLO define these cases as implementation-defined values and not as errors, so an error rule rejects inputs the source language accepts.
- 5. Real datapaths return a value: x86 `cvttss2si` returns the integer indefinite value when the invalid exception is masked; Arm and NVIDIA saturating converts return a value; tensor units have no data trap (Panel 1 evidence, not fetched here).
- 6. An ABI that faults on data fails scenarios real hardware allows (decision 31): masked softmax, padded tiles, and quantization of NaN all put zero divisors or NaN in a legal run.
- 7. Panel 5's own risk note (zero divisors in padded lanes) is not answered by "declared region only", since the zeros are inside the region.
- 8. A fixed value is exactly checkable on the pinned XLA; a wrong answer cannot hide behind it, since the oracle's value is the specified one (decision 27).
- 9. A hardware value that differs is the ISA author's to declare in the semantics, as decision 31 puts hazards; the default must not make the unguarded program invalid.
- 10. Decision 34 is consistent: addressing expressions are host integers evaluated before any read, so `/` by 0 in an expression still faults.
- 11. The cost argument is neutral: Panel 5 verified that all outputs return from one `execute`, so commit-after-check is a single host test; this ruling does not rest on cost.

Evidence:
- Verified: act-backend README lines 16 and 57-60 (HLO file in, assembly out, no data).
- Verified: act-oracle taidl_to/generic/decorator.py:188-196 (single `executable.execute`), taidl_to/oracle_visitor.py:29,48 (`divide`, `remainder` admitted on integers).
- Verified: act-backend expression_writer.py:94 (host expression divide returns None by zero, i.e. a host fault).
- Verified: StableHLO `divide`, "is an implementation-defined value of the element type of `result`" for zero `rhs` and `min / -1`.
- Verified: StableHLO `convert`, "If the truncated value cannot be represented in the destination type, the behavior is TBD".
- Verified: XLA operation semantics, integer division overflow and by zero "produces an implementation defined value"; float-to-int conversions "currently unspecified".
- Verified: Intel SDM page for `cvttss2si`, integer indefinite value when masked.
- Not fetched (pages truncated or redirected): Arm `SDIV`, PTX `cvt` and `div`.

Dissent:
- Panel 5 (Opus, Sonnet): a fixed value hides the kernel bug the oracle exists to catch.
- Weight: real, but it is a diagnostic wish and not a semantic need; a non-normative lint that reports a zero divisor without changing results is compatible and is left to the oracle plan.
- Fable (panel 5, round 1) held the guarded-value default before conceding on cost; his reason (no ISA in scope traps) agrees with this ruling.

Confidence: medium-high.
Would change the ruling: an in-scope ISA whose hardware raises a precise error on tensor divide by zero that kernels rely on.

## Point B: error bound for float `dot` and `reduce` (Q1.5 against Q5.2)

Decision: Panel 1.
- For `dot` and `reduce_add`: `|r - exact| <= gamma_n * sum|terms| + u * |exact|`.
- For `reduce_multiply`: `|r - exact| <= gamma_n * |exact|`.
- `gamma_n = n*u / (1 - n*u)`, u the unit roundoff of the result type, n the term count; plus the overflow clause of Q1.5.
- Adopted from panel 5 as an oracle rule inside this norm: the oracle widens narrow float `dot` and `reduce` to `f32` explicitly in the module and converts once to the result type.
- Panel 5's "declared accumulator type" is not a norm term.

Reasoning chain:
- 1. Both panels agree the first clause of R8 ("accumulate in the result type") is unobservable and false on the reference platform, so only a bound is a norm.
- 2. The TAIDL semantics is HLO text, where `dot` and `reduce` carry a result type and no accumulator type, so a "declared accumulator type" names a construct that does not exist (mini_trn.py semantics: `bf16[64,64] dot(...)`).
- 3. Panel 5's Opus states that the TAIDL result type is the accumulator, which collapses Q5.2 into Q1.5 where no construct is added.
- 4. Introducing an accumulator declaration would be a new language feature for panel 2, not a norm choice, and Q5.2 leaves its constant `c` unset, so it is not yet checkable.
- 5. Panel 5's form `c K u sum|x_i y_i|` has no term for the final rounding to a narrower result type, so an `f32`-accumulated `bf16` result can miss it for small K; Q1.5's `u * |exact|` term covers that case.
- 6. The result-type bound holds for every order and for every accumulation at least as wide as the result type, so it passes AMX, Trainium, TPU, and tensor cores, and a unit that accumulates at the result type (fp16 FMA).
- 7. A tighter u from a wider accumulator would fail such a narrow-accumulating unit, which an ISA abstraction must not do (decision 31's principle).
- 8. The bound is vacuous when `n*u >= 1` (`bf16`, n >= 256); the author writes a wider result type and converts, which `sh_dot` supports.
- 9. The oracle's explicit `f32` widening removes the dependence on an XLA pass, and keeps the oracle far inside the bound; this is the part of Q5.2 that stands.
- 10. Weakness accepted: a long `bf16` reduction is not judged tightly by the norm; the oracle test may compare against its own `f32` value with a tighter tolerance as a test choice, which is not the language norm.

Evidence:
- Verified: act-oracle taidl_to/generic/utils.py:575 (`DOT_WIDTHS` holds only `i2` and `ui2`), utils.py:597-606 (float dot emitted at its own type with `precision = [HIGHEST, HIGHEST]`).
- Verified: act-oracle utils.py:439 (`reduce_init`, float sum starts from +0.0).
- Verified: StableHLO `reduce`, "The order of reductions is implementation-defined".
- Verified: act-backend examples/mini_trn/mini_trn.py semantics, a `bf16` dot with `bf16` result and no accumulator field.
- Cited by debaters, not re-fetched: Higham, Accuracy and Stability of Numerical Algorithms, sections 3.1 and 4.2; XLA `cpu_compiler.cc` upcast passes; AMX `tdpbf16ps` per-FMA rounding.

Dissent:
- Panel 5 (Fable, Opus, Sonnet): u of the accumulator gives a meaningful bound where the result-type bound is vacuous.
- Weight: right about the vacuous case, but the accumulator is undeclared in the language today; if panel 2 adds one, it can tighten the bound per word without changing the default.
- Opus keeps exact `Fraction` accumulation as a fallback if `f32` widening moves the CPU goldens; no evidence decides it, and the norm covers either.

Confidence: medium-high on the form and the oracle widening; medium on the per-word overflow clause.
Would change the ruling: an accepted language construct that declares an accumulator type per instruction.
