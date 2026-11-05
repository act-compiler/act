# Panel 1, round 1: Sonnet (oracle implementer)

## Assumptions

- A1: the oracle stays one traced module per kernel, compiled once by XLA, with every address check on the host before tracing (utils.py:651-806, decorator.py:54).
- A2: R28 holds, so every loop is unrolled and a loop variable is a host integer at each issue.
- A3: the oracle runs on XLA CPU and on XLA GPU, in amd64 and arm64 images (README.md:147,163).
- A4: claims about XLA internals that I could not run here are marked "test" and are the cheapest evidence to settle them.
- A5: the StableHLO spec leaves integer divide by zero, signed overflow, and out-of-range convert implementation-defined or TBD (https://openxla.org/stablehlo/spec: divide, convert), so every such value the oracle gives is its own code, not an inherited norm.

### Q1.1

- Position: option D (every constraint reads the pre-state; regions, starts, sizes, and backticks read the post-state).
- 1. A constraint is a guard on the issue, so it must be evaluable before the updates run; an update can itself fault (S1.3.2, S1.3.1).
- 2. Under option A an update fault precedes any constraint test, so a guard such as `@s.sm == 1` can never protect an update that divides or overflows (casestudies/sme.md:18-31, spec-draft execution.md:22-27).
- 3. The oracle already runs constraints first, then updates, then sizes and inputs (instruction_codegen.py:171-173 appends constraints, then updates); option D costs zero code, option A reorders the emitter and the fault precedence.
- 4. After R58 only user constraints remain, so "constraint = precondition on the pre-state" is the whole story for constraints.
- 5. One-sentence norm: a constraint reads the state before the issue; everything the issue does reads the state after its updates.
- 6. A constraint on a new value is written over the update's own expression, so no expressive power is lost.
- Evidence: instruction_codegen.py:31,44,171-173; templates.txt:353-361; execution.md:22-27; sme.md:18-31; holes.md H11.
- Confidence: medium.
- Would change my mind: a case study whose guard needs the post-update value and cannot restate it, or a backend that cannot evaluate constraints before updates.

### Q1.2

- Position: option B (every intermediate outside i64 is a fault, checked per operator in the expression tree as written).
- 1. Exact arithmetic (A) needs the backend to emit i128 or bignums; a chain of three 64-bit products exceeds i128, so "exact" is not bounded by any fixed type (expression_writer.py:72,94 uses checked i64 per operator).
- 2. Option B is one sentence for both generators: checked i64 at each operator, no reassociation, no constant folding that changes an intermediate.
- 3. The oracle today evaluates in Python integers and only warns (utils.py:268-271); a per-step check is one comparison per operator at trace time on host integers.
- 4. The catch: the oracle canonicalizes values into sums of products (utils.py:54-65 `evaluate`), and expansion changes intermediates (a*(b-c) becomes a*b - a*c); the check must walk the original tree, not the canonical form.
- 5. Under R28 each unrolled issue evaluates a closed tree of integers, so the tree walk is simple; O11 is then needed only if the spike fails (fidelity.md O11).
- 6. Cross-generator agreement on a fault is testable only if both define the same intermediates; "final value only" is testable only by emulating exactness in both.
- Evidence: utils.py:54-65,268-271; expression_writer.py:72,94 (per agenda); fidelity.md O11; spec-draft expressions.md:30.
- Confidence: medium-high.
- Would change my mind: a backend measurement showing checked i64 is infeasible in constraints solved by the allocator, or a case study with a legitimate intermediate overflow.

### Q1.3

- Position: option A (alias by identical region text; all other overlap is defined by read-before-write; `set_inplace` removed).
- 1. The oracle loads every input before any output stores (instruction_codegen.py:165-181), so every overlap, exact or partial, already has the read-before-write value at zero cost.
- 2. Under that rule an alias has no value consequence in the oracle: it is an allocator contract (R21) and a statement of OD17, not a semantic switch.
- 3. Faulting on non-identical overlap (B) adds a host interval test and a rule to explain, and protects nothing the oracle can observe; hardware that cannot overlap writes a constraint (R58).
- 4. Option D (alias by evaluated regions) makes aliasing depend on state at issue, so the front end cannot tell an in-place instruction from a non-in-place one; text identity is decidable before any run.
- 5. Equal regions spelled differently are not aliases but are still correct under read-before-write; the backend may not assume storage reuse for them, which is safe.
- 6. Output-output overlap stays a fault; the oracle needs a host check because the later write wins today (instruction_codegen.py:667-681, per holes.md H2).
- Evidence: instruction_codegen.py:165-181; holes.md H1, H2; spec-draft instructions.md:50-53; overview.md:42; mini_trn.py:97-99 (softmax in place).
- Confidence: medium-high.
- Would change my mind: a target whose hardware gives a different value on partial overlap and whose author cannot express disjointness as a constraint over `@a.` addresses.

### Q1.4

- Position: option C (a read of an on-chip element no earlier issue wrote is a fault), tracked with a host boolean mask per buffer.
- 1. Zero fill is a promise real scratchpads do not keep, so option A makes an oracle pass certify kernels that read stale storage (templates.txt:470 `sh_zeros`).
- 2. Every region address is a host integer (A1, A2), so the written set is exact on the host; a data-dependent read-before-write test is never needed.
- 3. Buffers are multi-dimensional, so the right structure is a NumPy bool mask per buffer, not an interval set: a write sets `mask[slice] = True`, a read tests `mask[slice].all()`.
- 4. Cost: O(region elements) host work per issue; at 10^5 issues of 4096 elements this is a few seconds of NumPy, against compile time that is already larger.
- 5. `d0` starts fully written (the caller's `hbm`), and state variables start at their initial value, so neither needs a mask.
- 6. "Written" means written during the kernel, not written by the same logical tensor; a stale read of a reused buffer is not caught, and the spec must say so.
- 7. Any kernel that today relies on zero fill (an accumulator never initialized) fails and must initialize explicitly, which is what the hardware needs.
- Evidence: templates.txt:470; decorator.py (hbm sizing near :120-145); holes.md H8; spec-draft execution.md:15-19.
- Confidence: medium.
- Would change my mind: examples or case studies that read unwritten storage on purpose (run test_examples under the mask); a target whose scratchpad is defined zero at reset.

### Q1.5

- Position: option C, corrected: an order-free bound, with the accumulation type set by the dot's result type; goldens are a test artifact, not the norm.
- 1. Option D (wide accumulation, bit-exact) cannot be a norm: CPU goldens are bit-exact but GPU is held to four bf16 ulps, so the oracle itself has two numerics today (README.md:163).
- 2. Option A ("accumulate in the result type") is unobservable and, at bf16, differs from what XLA does when it widens internally (test); a norm no platform meets bit for bit must be a bound.
- 3. A bound independent of summation order is the only implementable and testable form; the Higham bound is |r - exact| <= gamma_n * sum|terms| with gamma_n = n*u/(1 - n*u) (https://epubs.siam.org/doi/book/10.1137/1.9780898718027).
- 4. DB6's formula is wrong for `reduce_multiply`: a product error is relative to |exact|, not to sum|terms|; the norm needs two forms.
- 5. The bound is vacuous when n*u >= 1 (bf16 u = 2^-8 gives n >= 256), so the spec must say that wider accumulation is the dot's result type (bf16 x bf16 -> f32, then `convert`), which sh_dot already supports because operand and result types are independent (utils.py:604-607).
- 6. Overflow gap: a sum with terms of both signs near the largest value is inf or finite depending on order; the norm needs a clause ("if sum|terms| exceeds the largest finite value the result is any value of the type") or it stays undefined.
- 7. StableHLO `dot_general` carries `precision_config` and `accumulation_type` only as hints (https://openxla.org/stablehlo/spec: dot_general), so the oracle cannot pin an order through the spec either.
- Evidence: README.md:163; utils.py:584-612,603-607; review-response.md:108; fidelity.md O13, O22.
- Confidence: medium (bound form high, overflow clause low).
- Would change my mind: a measurement that XLA CPU accumulates bf16 dot in bf16 (then A is the true behavior), or an owner wish for bit-exact goldens as the norm.

### Q1.6

- Position: rounding option A, set per compile; denormals option C (keep), enforced on both platforms.
- 1. `xla_allow_excess_precision` defaults to true in XLA (https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc), so a bf16 chain such as the softmax at mini_trn.py:101-104 may stay at f32 between lines.
- 2. Setting it through process `XLA_FLAGS` "before jax initializes" is fragile, because a user imports jax first; the oracle already builds `CompileOptions` itself (decorator.py:54), so it can set `debug_options.xla_allow_excess_precision = False` on its own compile (test that the pybind exposes it).
- 3. Fused multiply-add is a second excess-precision path: LLVM contraction can fuse a mul and an add into one rounding (test); the bf16 chain test must include an f32 mul-add.
- 4. The same file lists `xla_cpu_ftz` with a default that flushes denormals on CPU (source fetch, to re-verify), while XLA GPU keeps them by default, so CPU and GPU oracles may already differ.
- 5. Keeping denormals is IEEE's default and the only explainable word; flush "every input and result" cannot be enforced inside library kernels (Eigen or cuBLAS dot) or checked per operation.
- 6. The oracle sets `xla_cpu_ftz=false` per compile and adds a denormal test per exact word on both platforms; an ISA whose hardware flushes states it in its own constraints or documents the gap.
- Evidence: decorator.py:22-23,54; debug_options_flags.cc (above); mini_trn.py:101-104; holes.md H6; fidelity.md O12.
- Confidence: medium (flag names and defaults are test items).
- Would change my mind: a test showing the pinned jaxlib cannot be made to keep denormals on CPU, in which case the single norm becomes "flush" with a software flush on inputs.

### Q1.7

- Position: NaN option A; zero option C (IEEE 754-2019); special values option E (limited to inf, NaN, and zero inputs).
- 1. Option B (canonical NaN after every operation) costs a select per float operation and breaks fusion; NaN bits also differ by architecture (x86 default NaN has the sign set, ARM does not), and the oracle ships on both (README.md:163).
- 2. So any NaN equals any NaN; but `bitcast_convert` of a NaN would then be an arbitrary integer, which is undefined by another name; fix it at that one word: a float NaN read through `bitcast_convert` is the canonical quiet NaN of the type (one select, only on that word).
- 3. NumPy does not settle signed zero: the `maximum` doc states NaN propagation and says nothing about -0 and +0 (https://numpy.org/doc/stable/reference/generated/numpy.maximum.html), so NumPy cannot be the reference there.
- 4. XLA CPU defaults `xla_cpu_enable_fast_min_max` to true with Eigen NaN behavior (debug_options_flags.cc); x86 MAXPS returns the second operand for two zeros, so (-0, +0) is operand-order dependent; the oracle sets the flag false per compile and tests both orders.
- 5. StableHLO says only "maximum from IEEE-754" (https://openxla.org/stablehlo/spec: maximum), which is the 2019 `maximum` if read as the newer standard; the spec text must name IEEE 754-2019 explicitly.
- 6. Special values: exp(-inf) = 0, exp(inf) = inf, log(0) = -inf, log(x<0) = NaN, tanh(+-inf) = +-1, rsqrt(0) = inf are testable by a table of one line per word on both platforms; values near an overflow or underflow threshold stay inside the per-word tolerance.
- Evidence: debug_options_flags.cc; numpy maximum doc; README.md:163; utils.py:584-612.
- Confidence: medium (NaN canonicalization at bitcast, low-medium).
- Would change my mind: a test that XLA already yields positive quiet NaN everywhere, or an owner choice to leave `bitcast_convert` of a NaN as an explicit documented non-determinism.

### Q1.8

- Position: option A for every class, lowered explicitly in the oracle.
- 1. StableHLO leaves float to int out of range TBD (https://openxla.org/stablehlo/spec: convert) and XLA says the precise conversions are unspecified (https://openxla.org/xla/operation_semantics: ConvertElementType), so the oracle must not inherit a value from the backend.
- 2. Tensor data cannot fault (Q1.9), so every case needs a value; saturate with NaN to 0 is hardware-natural (ARM FCVTZS and CUDA saturating converts do it; x86 CVTTSS2SI does not), and wrap for int to int is the two's-complement truncation the oracle already gets.
- 3. Explicit lowering costs a few elementwise operations per float-to-int `convert` (clamp to the integer range as floats, select NaN to 0, then convert) and makes the value independent of the jaxlib version and platform.
- 4. Float to float overflow to infinity is IEEE; at `f8e4m3fn` the result is NaN (no infinity); test against ml_dtypes at the edge values, since saturate (option C) would be the alternative a hardware designer may expect.
- 5. A saturating ISA writes `clamp` before `convert`, which is the existing escape hatch and keeps the language norm uniform.
- 6. Option B and D (faults) need the O23 device flag and are ruled out by Q1.9.
- Evidence: operation_semantics (above); utils.py:393-426 (`mlir_literal` already handles ranges per type); overview.md:48; coverage-table.md float convert row.
- Confidence: medium-high.
- Would change my mind: a pinned-XLA test showing its saturation differs from the lowering at a type pair that the oracle cannot lower (sub-byte).

### Q1.9

- Position: option A (tensor data never faults; fixed values; clamped indices). Option D is the same rule read from the other side.
- 1. The oracle is one module per kernel; a data-dependent fault at issue k needs a flag threaded through every later issue (state' = select(flag, state, new)) so that nothing after k commits, which doubles memory traffic per issue (fidelity.md O23: "the kernel commits its outputs only when no flag is set").
- 2. Hardware integer units do not trap on divide by zero; a value is the faithful behavior.
- 3. The values are XLA's emitter code, not StableHLO's (divide by zero and min / -1 are implementation-defined in both specs: https://openxla.org/stablehlo/spec: divide; https://openxla.org/xla/operation_semantics: Div), so the oracle lowers them explicitly (select on a zero divisor) and tests each fixed value.
- 4. Index clamping is StableHLO's text: adjusted_start_indices = clamp(0, start_indices, shape(operand) - slice_sizes) (dynamic_slice, same page), so no new rule is needed; gather clamps likewise.
- 5. The one-sentence norm is "faults depend only on attributes, state, and region text, never on tensor data"; it explains why expression `/` faults and tensor `divide` does not.
- 6. This makes every fault a host check before tracing (utils.py:651-806), which is what Q1.12 needs.
- Evidence: fidelity.md O23; operations.md S8.1/S8.4; critic.md:46; both spec pages above.
- Confidence: high.
- Would change my mind: an owner wish that a zero divisor be diagnosed in kernels, at the cost of the flag threading above.

### Q1.10

- Position: option A; option D resolved as a fault at issue.
- 1. The oracle already rejects a literal past the largest value, with the tie rule (utils.py:403-406, `abs(value) == bound` rejected unless the type has no infinity), so A is the status quo.
- 2. Option B makes `1e40` at f32 silently infinity; a program that wants infinity writes `inf`, which the oracle already supports (utils.py:416-420).
- 3. Rounding must run once from the exact decimal: use `fractions.Fraction(text)` and emit bits in hex (fidelity.md O8), which also fixes an integer above 2^53 at a float type.
- 4. Option C (reject inexact literals) rejects `0.1`, so it is unusable.
- 5. A nonzero literal that underflows to zero rounds normally; rejecting it would add a rule without a use.
- 6. A backtick constant depends on state, so it is known only at issue; rejection before any run is impossible, and fault is the only form (S1.3.9).
- Evidence: utils.py:393-426; fidelity.md O8; spec-draft semantics.md:77, overview.md:45.
- Confidence: medium-high.
- Would change my mind: an owner wish that out-of-range float literals saturate, as many assemblers do.

### Q1.11

- Position: option A (identity at the type; a float sum starts at +0.0).
- 1. It is the oracle today (utils.py:439-451) and what XLA's reduce does with a +0 init.
- 2. Option B would also need `dot` to start at -0.0, but gemm kernels start at +0 and Q1.5 pins only a bound; the zero sign of a sum falls outside what the bound sees, so B is not enforceable for the words it should cover.
- 3. The norm for reduce words stays exact on the init and only the order is free; the zero-sign sentence reads "an all-negative-zero sum is +0.0".
- 4. Option C needs grammar the language lacks.
- 5. For size-1 and empty reductions say "the element combined with the identity" (SC22).
- Evidence: utils.py:439-451; semantics/rulings.md (reduction initial values); holes.md H7; overview.md:49.
- Confidence: medium-high.
- Would change my mind: an ISA whose reduction hardware returns -0.0 for negative zeros, and a test that XLA honors a -0.0 init end to end.

### Q1.12

- Position: atomicity as a whole-program error (R55 simplified), constraint failure option C, producer option E.
- 1. With Q1.9 A every fault is detected on the host before the module is traced, so a faulting program runs nothing; the observable guarantee is an error naming the issue index and clause id, with no outputs.
- 2. Per-issue rollback (option A as worded) is then unobservable except in a trace, so the spec states the simpler sentence: a fault ends the program with an error, and no output of the program is defined.
- 3. If a later ruling adds a data-dependent fault (Q1.9 B or C), per-issue atomicity forces the flag threading of Q1.9 step 1; that cost belongs to that ruling.
- 4. Precedence: report the lowest issue index; within one issue any applicable clause may be named, since the program is in error either way.
- 5. A REQUIRE failure is always a fault (option C); the backend reading "never selected" is a restatement, so dynamics.md drops "does not hold".
- 6. The backend chooses `@a.` values and placements, so it is a producer (option E); the check has teeth because every backend-emitted program can run on the oracle and must raise nothing, as a closed-loop test.
- Evidence: fidelity.md O23; review-response.md:111; holes.md H5, H11; spec-draft execution.md:27,34.
- Confidence: high on C and E, medium on the whole-program wording.
- Would change my mind: a simulator use case that needs the state after the faulting issue.
