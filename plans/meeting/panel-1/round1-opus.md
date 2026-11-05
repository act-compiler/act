# Panel 1, round 1: Opus (semantics and spec author)

Assumptions for every answer: a good norm is one sentence that both generators meet without extra communication.
A fault is the spec's tool where hardware behavior must not be promised; a fixed value is the tool where every generator can compute it.
R28 holds (every loop unrolled, every expression evaluated on host integers at issue) and R58 holds (every constraint is written by the ISA author).

### Q1.1
Position: option C, every expression of an issue reads the state before the issue, and updates take effect with the outputs.
1. Pre-state is strictly more expressive: a post-state value is the update text substituted into the expression, while an overwritten pre-state value cannot be recovered.
2. So C expresses every guard and address A can, and A cannot express `smstart` guarding `@s.sm == 0` while it sets `@s.sm = 1`.
3. Post-index and pre-index addressing are both writable under C: start `@s.ptr` or `@s.ptr + 64`, each with update `@s.ptr = @s.ptr + 64`.
4. The mixed view (B, D) already hides a bug in the corpus: `load` guards `@s.row + @c.n <= 32` on the old row but writes at the new row.
5. Under C the guard and the write read one value, which a reader expects without reading S5.6.
6. Under C a fault rolls nothing back, since state and regions commit together at the end of the issue.
7. I also make updates read the pre-state (parallel assignment), so no expression sees a partial update; a chained update is rewritten by substitution.
8. The sentence: "Every expression of an issue reads the state before the issue; its updates and its outputs take effect together at its end."
Evidence:
- Step 2: plans/casestudies/sme.md:20-30; reviews/rulings/audit.md:20.
- Step 4: act-oracle/tests/test_state.py:24-28 (constraint line 24, update line 25, output start line 28).
- Step 6: language/spec-draft/execution.md:24-29.
- Step 7: act-oracle/tests/test_state.py:19 is the only chained update found (`@s.row = @s.step` becomes `@s.row = @c.step`).
- External: Sail ISA models read source registers at the start of an `execute` clause and write `X(rd)` as an effect, https://github.com/rems-project/sail.
Confidence: medium (high against the mixed view; medium for C over A).
Would change my mind: a case-study ISA whose start must read a value an update computes and cannot be written by substitution; for step 7 alone, update chains too long to substitute readably.

### Q1.2
Position: option B, each operator's result must lie in signed 64 bits, else the issue faults.
1. Both options are deterministic, so the choice is which sentence both generators meet without hidden work.
2. B is exactly `checked_add`, `checked_sub`, `checked_mul`, which the backend already emits per operator.
3. A forces generated Rust onto i128, bignums, or a bound proof, for a case no example exercises.
4. B costs the oracle one range check per operator on Python ints, host work only after R28.
5. B ties the fault to the parse tree, so the spec says evaluation follows the grammar's left-associative tree (S3 fixes it).
6. The least value stays writable: `-9223372036854775807 - 1` has every intermediate in range.
7. Expression `min / -1` gives 2^63 and faults by the same sentence; no extra row is needed.
8. The sentence: "Every operator of an expression yields a signed 64-bit integer, or the issue faults."
Evidence:
- Step 2: act-backend/act_backend/expression_writer.py:72,94-97.
- Step 3: reviews/rulings/audit.md:33.
- Step 6: language/spec-draft/expressions.md:24.
- External: https://doc.rust-lang.org/std/primitive.i64.html#method.checked_add; WebAssembly defines i64 results per operator, https://webassembly.github.io/spec/core/exec/numerics.html.
Confidence: high.
Would change my mind: a corpus address expression with an in-range final value and an out-of-range intermediate.

### Q1.3
Position: option A, every input is read before any output is written, and an input and an output share storage exactly when their region texts are equal token for token.
1. Read-before-write already fixes the value of every issue: inputs come from the pre-state memory, outputs are written after.
2. So any overlap, exact or partial, equal text or not, has one defined result, and the value needs no further rule.
3. OD17's "reuses the storage" follows for equal regions: the post-state at the region is the output.
4. Text identity is what the backend sees statically, and it is exactly the R21 allocator equality.
5. A fault for other overlaps (B, C) defines nothing new and only forbids; OD16 asks for the simplest norm.
6. Option D makes aliasing dynamic, which the backend cannot use, and changes no value.
7. Hardware that corrupts on partial overlap is described by an ISA-author constraint (R58), so a producer never emits it.
8. `set_inplace` is removed; the `d0`-input, `d2`-output fixture is not in-place under OD17, since the storage differs, and migrates to a copy.
9. The sentence: "An issue reads every input region before it writes any output region; an input and an output with equal region text are one storage."
Evidence:
- Steps 1-2: semantics/dynamics.md:41-48; language/spec-draft/execution.md:28-29.
- Step 4: minutes/rulings.md:45 (R21).
- Step 8: act-backend/tests/test_extraction.py:507; act-backend/examples/mini_trn/mini_trn.py:97-99 (equal text, the real in-place case).
- External: StableHLO `dynamic_update_slice` is pure value semantics with no aliasing notion, https://openxla.org/stablehlo/spec.
Confidence: high on the value rule; medium on token equality versus character equality.
Would change my mind: hardware where an exact text match must not share storage (a double-buffered write), which would need a mark again.

### Q1.4
Position: option C, a read of an element of a buffer other than `d0` that no earlier issue wrote is a fault; the `hbm` contents write all of `d0`.
1. Zero fill (A) promises a value scratchpads do not hold, so every TAIDL description of real hardware would be false there.
2. A fault stays defined without promising hardware behavior, as an out-of-bounds region already does.
3. Every region is a function of host integers, so the oracle tracks written intervals per buffer exactly, at no device cost.
4. Under A a kernel that reads stale storage passes against zeros, hiding the backend bug the oracle exists to catch.
5. Option B gives the backend a duty the spec cannot test; C gives the duty a test.
6. A kernel that reads unwritten padding into a discarded result must write the padding first, which a correct hardware kernel already needs whenever padding reaches a kept value.
Evidence:
- Step 1: act-oracle/taidl_to/templates/templates.txt:469-470 (`sh_zeros` prologue); language/spec-draft/execution.md:17.
- Step 3: reviews/semantics/defined-behavior.md, section "Fault detection cost in the oracle".
- External: WebAssembly zero-fills memory because it promises a portable sandbox, https://webassembly.github.io/spec/core/exec/modules.html; TAIDL models hardware, which makes no such promise.
Confidence: medium.
Would change my mind: a corpus kernel that reads unwritten storage by design with every result kept, or hardware that zeroes its scratchpad at kernel start.

### Q1.5
Position: option C corrected: `dot` and `reduce_add` satisfy `|r - exact| <= gamma_n * sum|terms| + u * |exact|`; `reduce_multiply` satisfies `|r - exact| <= gamma_n * |exact|`; `gamma_n = n*u / (1 - n*u)`.
1. StableHLO leaves reduction order and init positions implementation-defined, so no bitwise norm can be inherited (OD14).
2. "Accumulate in the result type" is a procedure no test observes, so only a bound on the result can be the norm.
3. The summation bound `gamma_{n-1} * sum|x_i|` holds for every order and every accumulation at least as wide as the result type, exactly the freedom StableHLO grants.
4. The dot bound `gamma_n * sum|a_i * b_i|` adds one rounding per product, hence n for dot.
5. A product's error is relative, `(1 + delta)^n`, so the `sum|terms|` form of DB6 is wrong for `reduce_multiply`.
6. The term `u * |exact|` covers the final rounding to the result type after a wider accumulation.
7. When `n*u >= 1` the bound constrains nothing, which is the honest statement for long bf16 reductions; an ISA needing better writes a wider `convert`.
8. n is K for dot and the product of the reduced dimension sizes for a reduce; a NaN or infinite term takes Q1.7's rule instead.
9. The oracle may accumulate wider and stay bit-exact (option D) as its own choice inside the norm.
Evidence:
- Step 1: https://openxla.org/stablehlo/spec (`reduce`: "The order of reductions is implementation-defined"; `dot_general`: `accumulation_type`).
- Steps 3-5: N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM 2002, sections 3.1 and 4.2.
- Steps 2, 7: reviews/rulings/audit.md:21; act-backend/examples/mini_trn/mini_trn.py:91 (bf16 dot, K = 64, n*u = 0.25).
Confidence: medium-high.
Would change my mind: a pinned-XLA bf16 dot violating the bound, which would mean XLA rounds intermediates below the result type.

### Q1.6
Position: option A for rounding; for denormals option E, with keep (C) as the stated norm unless the pinned jaxlib cannot keep them.
1. The float exact class is bitwise, so each operation rounds once to its result type; excess precision makes two conformant oracles differ.
2. XLA's `xla_allow_excess_precision` defaults to true, so the oracle sets it false before jax initializes and a bf16 chain test proves it.
3. IEEE 754 delivers subnormals by default and StableHLO defines its float words by IEEE 754, so keep is the inherited norm under OD14.
4. Flush needs a second sentence (input and output, sign kept); keep needs none.
5. A per-ISA denormal knob is speculative until a second ISA needs flush, and "either" is excluded by OD16.
Evidence:
- Step 2: github.com/openxla/xla, xla/debug_options_flags.cc (`xla_allow_excess_precision`, default true); act-backend/examples/mini_trn/mini_trn.py:101-104.
- Step 3: IEEE 754-2019 clause 7.5; https://openxla.org/stablehlo/spec ("For floats: `maximum` from IEEE-754", likewise per float word).
Confidence: high on rounding; medium on denormals.
Would change my mind: the pinned jaxlib CPU client under flush-to-zero with no switch, making flush the one norm.

### Q1.7
Position: NaN by option A with one fixed bit pattern at `bitcast_convert`; zeros by option C; special values by option E.
1. A NaN result carries no sign or payload in the meaning, and every NaN compares equal to every NaN in conformance.
2. Bits escape only through `bitcast_convert` of a float source, since `d0` holds bytes.
3. So the spec fixes that one point: `bitcast_convert` of a NaN gives the positive canonical quiet NaN bits of its source type.
4. This is WebAssembly's deterministic profile applied at the single observation point, for one select per float-source `bitcast_convert`.
5. Option B buys the same observable result for a select after every float operation.
6. `maximum` and `minimum` follow IEEE 754-2019: NaN propagates and -0 < +0.
7. A ulp tolerance is undefined at an infinity, so special values are exact: the IEEE 754-2019 clause 9.2 cases (`exp(-inf) = +0`, `log(+0) = -inf`, `tanh(+inf) = 1`).
Evidence:
- Step 4: https://webassembly.github.io/spec/core/exec/numerics.html (NaN sign "non-deterministic"; deterministic profile: "only positive canonical NaN outputs are produced").
- Step 6: https://openxla.org/stablehlo/spec (`maximum` from IEEE-754); IEEE 754-2019 clause 9.6.
- Step 7: IEEE 754-2019 clause 9.2.1; act-oracle generic/utils.py:584-612.
Confidence: medium-high.
Would change my mind: a kernel whose result depends on NaN payload other than through `bitcast_convert`, or the pinned XLA giving -0 for `maximum(-0, +0)`.

### Q1.8
Position: option A for every class, plus "to `pred`: nonzero, including NaN, is true".
1. Options B and D make tensor data fault, which Q1.9 rules out.
2. Float to float and int to float round to nearest even, and an overflow delivers a signed infinity (IEEE 754).
3. A type with no infinity (`f8e4m3fn`) delivers NaN, the OCP 8-bit non-saturating rule; option C invents a rule IEEE lacks.
4. Float to int truncates, saturates, and maps NaN to 0, as WebAssembly `trunc_sat` and spec-draft S8.4 do.
5. Int to int wraps, as every integer word does; a saturating ISA writes `clamp` before `convert`.
6. StableHLO calls out-of-range `convert` "TBD", so the norm is ours and is tested on the pinned XLA.
Evidence:
- Step 2: IEEE 754-2019 clause 7.4.
- Step 4: https://webassembly.github.io/spec/core/exec/numerics.html; language/spec-draft/operation_meaning.md:39.
- Step 6: https://openxla.org/stablehlo/spec (`convert`: "the behavior is TBD").
Confidence: medium-high.
Would change my mind: the pinned XLA saturating `f32` to `f8e4m3fn`, making saturation the norm for that pair.

### Q1.9
Position: option A, tensor data never faults; a fault depends only on the issue's attributes, state, and region text.
1. The backend chooses attributes and placements but never sees tensor values, so it can guarantee host-integer preconditions and no data precondition.
2. Under B or C a `divide` on run-time data may fault, so "never emits a faulting issue" (Q1.12) becomes unsatisfiable.
3. Under A every fault is checked on the host before tracing, so the oracle needs no O23 flag and no commit-after-check.
4. One norm by input kind, not by word: "A fault depends only on host integers; every tensor value gives a fixed result."
5. So an expression `/` by 0 faults, and a tensor `divide` by 0 takes XLA's fixed value, both by that sentence.
6. Data indices clamp, as StableHLO defines for `dynamic_slice` and `dynamic_update_slice`.
Evidence:
- Step 3: act-oracle generic/utils.py:651-806; oracle/plan/fidelity.md:75-85 (O23 cost).
- Step 5: language/spec-draft/operation_meaning.md:35-38.
- Step 6: https://openxla.org/stablehlo/spec (`adjusted_start_indices = clamp(0, start_indices, shape(operand) - slice_sizes)`).
Confidence: high.
Would change my mind: hardware raising a precise exception on a tensor divide by zero that a kernel relies on.

### Q1.10
Position: options A and D: a float literal is its exact decimal rounded once to nearest even; one that rounds to an infinity (NaN at `f8e4m3fn`) is rejected; an out-of-range backtick constant faults.
1. One rounding from the exact decimal is the IEEE 754 conversion rule; the oracle's rounding via f64 rounds twice.
2. A finite literal that silently becomes infinite is an author error the front end sees, so it is rejected.
3. Underflow rounds normally, to a subnormal or a signed zero, under Q1.6.
4. Round-trip exactness (C) rejects `0.1`, which no author expects.
5. A backtick constant reads attributes or state at issue, so it cannot be rejected before the run and faults like every host-integer case.
6. An integer literal at a float type rounds once from the exact integer by the same sentence.
Evidence:
- Step 1: IEEE 754-2019 clause 5.12.2; oracle/plan/fidelity.md:15 (O8, `Fraction` rounding).
- Step 5: language/spec-draft/overview.md:45 (S1.3.9).
Confidence: medium-high.
Would change my mind: an ISA that needs an infinity constant with no other way to write it, which argues for an explicit `inf` literal, not for accepting overflow.

### Q1.11
Position: option B, the float sum starts from -0.0, the IEEE additive identity.
1. Under round to nearest `x + (-0) = x` for every x including -0, while `-0 + (+0) = +0`, so only -0.0 is an identity.
2. With -0.0 "the start value is the reducer's identity" is true as written, and a size-1 reduction returns its element.
3. Every reduced dimension has size at least 1 (S1.3.3, S1.3.7), so an empty sum never occurs.
4. The cost is one constant in `reduce_init`, since XLA honors any init value given to `reduce`.
5. The stake is small: `reduce_add` is in the tolerance class (Q1.5), so the zero sign shows only through `bitcast_convert` or a later division.
Evidence:
- Step 1: IEEE 754-2019 clause 6.3.
- Steps 2, 4: language/spec-draft/operation_meaning.md:27; act-oracle generic/utils.py:439-451.
- External: StableHLO requires `body` and `init_values` to form a monoid and names float addition with zero as failing it, https://openxla.org/stablehlo/spec.
Confidence: low-medium.
Would change my mind: a requirement that the oracle match `jnp.sum` bit for bit on signed zeros, which starts from +0.0.

### Q1.12
Position: options A, C, and E: a fault writes no region and no state, every REQUIRE failure is a fault, and the backend never emits an issue that faults.
1. With Q1.9 A every fault is decided on host integers before any read, so atomicity costs nothing for every fault.
2. WebAssembly's `memory.init` checks bounds before it stores any byte, the same check-then-commit shape.
3. With Q1.1 C state commits with the outputs, so no rollback needs specifying.
4. Two words for one outcome ("does not hold", "fault") invite two implementations; fault is the one word, which the backend reads as "never selected".
5. The backend chooses `@a.` values and placements, so it produces those preconditions; "may assume" points the duty the wrong way.
6. A precondition the backend cannot see (state before R27, `|d0|`) is listed in its coverage doc.
7. The program stops at the first fault, and every earlier issue has committed.
Evidence:
- Step 2: https://webassembly.github.io/spec/core/exec/instructions.html (`memory.init` step 8 traps before the stores of step 11).
- Step 4: semantics/dynamics.md:22; minutes/rulings.md:27 (R55).
- Step 5: minutes/rulings.md:14,20 (R4, R10); semantics/holes.md:18 (H11).
- Step 7: language/spec-draft/execution.md:34.
Confidence: high.
Would change my mind: a ruling of Q1.9 B or C, which makes A cost an O23 flag and commit-after-check and makes E unsatisfiable for data faults.
