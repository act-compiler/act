# Panel 1 decision: defined behavior and numerics

| Question | Decision (one line) | Confidence | Owner? |
| -------- | ------------------- | ---------- | ------ |
| Q1.1 | Option C: every expression of an issue reads the pre-state; updates and outputs commit together at the end of the issue | medium-high | no |
| Q1.2 | Option B: every operator of an expression yields a signed 64-bit integer, or the issue faults | high | no |
| Q1.3 | `set_inplace` stays as the "must reuse" mark; unmarked exact overlap at issue is legal read-before-write; partial overlap is a fault | medium | no (one item forwarded) |
| Q1.4 | Option C: a read of an on-chip element no earlier issue of the kernel wrote is a fault; `d0` holds the caller's `hbm` | medium-high | no |
| Q1.5 | Two Higham bounds with u of the result type, plus an overflow clause, for `dot`, `reduce_add`, `reduce_multiply` only | medium | no |
| Q1.6 | Rounding A (one rounding per operation); denormals kept, verified by a pinned-jaxlib test per platform | medium-high / medium | no |
| Q1.7 | NaN A with the positive canonical quiet NaN at `bitcast_convert`; zero C (IEEE 754-2019); special values E (exact) | medium-high | no |
| Q1.8 | Option A per conversion class; to `pred` nonzero including NaN is true; a saturating ISA writes `clamp` first | medium-high | no |
| Q1.9 | Option A: faults depend only on attributes, state, and region text; tensor data takes fixed values and indices clamp | high | no |
| Q1.10 | Options A and D: one rounding from the exact decimal; a literal rounding to infinity is rejected; a backtick constant out of range faults | medium-high | no |
| Q1.11 | Option A: the reducer's identity at the type; a float sum starts from +0.0 | medium-high | no |
| Q1.12 | Options A, C, E: per-issue atomicity, every REQUIRE failure is a fault, the backend is a producer that never emits a faulting issue | high | no |

## Q1.1: which state do expressions read

- Decision: option C, one sentence: "Every expression of an issue reads the state before the issue; its updates and its outputs take effect together at its end."
- Updates are a parallel assignment, so no expression sees a partial update; a chained update is written by substitution.
- 1. The mixed view (B, D) is the one the corpus already gets wrong: `load` guards `@s.row + @c.n <= 32` on the old row, updates `@s.row`, and writes its output at `@s.row`, so the guard checks one row and the write lands at another.
- 2. The pre-state is strictly more expressive: a post-state value is the update text substituted into the expression, while an overwritten pre-value cannot be recovered.
- 3. Post-increment addressing, the common hardware form, is start `@s.ptr` with update `@s.ptr = @s.ptr + 64` under C and needs `@s.ptr - 64` under A.
- 4. The SME pair does not discriminate between A and C, since `smstart` has updates and no constraint and `ld1w_zah` has constraints and no update.
- 5. C is one view, so OD16's "easy to explain" holds, and a guard still precedes any update fault because no expression depends on an update.
- Evidence, verified: act-oracle/tests/test_state.py:24-28 (constraint line 24, update line 25, output start `@s.row` line 28); act-oracle/taidl_to/instruction_codegen.py:162-181 (constraints, updates, sizes, inputs, semantics, outputs emitted in that order, so starts read the post-state today); act/plans/casestudies/sme.md:18-31.
- Cost: the oracle moves the update emission after the outputs in instruction_codegen.py:162-181 and makes it parallel; OD10 allows the break.
- Dissent: none in the final positions; Sonnet's D keeps today's emitter order at zero cost but keeps the two-value hazard.
- For the authors: the agenda cites `instruction_codegen.py:732-737`, but the file has 224 lines; cite lines 162-181.
- Confidence: medium-high.

## Q1.2: intermediate overflow in expressions

- Decision: option B, one sentence: "Every operator of an expression yields a signed 64-bit integer, or the issue faults."
- 1. The backend already emits `checked_add`, `checked_sub`, `checked_mul` per operator, so B is the sentence both generators meet without new arithmetic width.
- 2. Option A forces generated Rust and C++ onto i128 or bignums for a case no example exercises.
- 3. The oracle today evaluates in Python integers and only warns, so B costs one range check per operator at trace time.
- 4. The check walks the expression tree as written, never the oracle's canonical sum of products, since expansion changes intermediates.
- 5. Evaluation order follows the grammar's left-associative tree, which the spec states.
- Evidence, verified: act-backend/act_backend/expression_writer.py:72,94-97; act-oracle/taidl_to/generic/utils.py:54-65 (`evaluate` over a canonical form), 268-271 (`warn_width`).
- Dissent: none.
- Confidence: high.

## Q1.3: in-place mark and overlap

- Decision on redundancy: the mark is not shown redundant, so OD22 keeps it.
- 1. Identical region text expresses "must reuse" by construction only when the input and output share one attribute; a two-field encoding whose fields the hardware requires equal has no textual witness, and a constraint `@a.dst == @a.src` names neither the pair nor the buffer and size equality.
- 2. The mark is the one place that names the destructive input, checks buffer and size by text in the front end, and faults at issue when the evaluated starts differ; derived text identity can never fault.
- 3. The backend consumes the mark: `fix_inplace` gives an instruction overwriting a shared operand its own copy, and errors when the operand is pinned in HBM or views shared storage.
- 4. Every corpus use pairs identical text with the mark, so the mark is derivable there, which is why its redundancy was arguable but not shown.
- Meaning of the mark, one norm: "A marked input k and output region 0 are one storage: same buffer and equal sizes by text or the front end rejects, equal starts at issue or the issue faults; after the issue the input region holds the output."
- The marked pair in different buffers is rejected (OD24), so the `load_d2` fixture migrates to a copy.
- Identical input and output text without the mark is rejected by the front end, which asks for the mark, because the overlap is certain and the backend would otherwise destroy a live operand.
- Unmarked overlap, one norm: "An issue reads every input region before it writes any output region; an unmarked input whose evaluated region equals an output region exactly is legal and sees the pre-state, and the implementation may or may not reuse its storage."
- 5. OD22 states this case directly: `r0 = r1 + r2` allows `r2` to equal or differ from `r0`, so a fault on unmarked exact overlap (Opus) contradicts a binding decision and is rejected.
- 6. The value is read-before-write because the oracle loads every input before any output store, at no cost.
- Partial overlap, one norm: "A partial overlap at issue between an input region and an output region, by evaluated addresses, is a fault."
- 7. No shipped ISA defines partial input-output overlap: RVV reserves it outside two positions, AMX raises an undefined-opcode exception, and a streaming systolic matmul writes garbage; the lane-shift example is the very case RVV forbids.
- 8. A fault is a defined outcome under OD16, costs the same host interval test every source already requires for output-output overlap, and spares the author of a systolic ISA a disjointness constraint on every instruction under R58.
- 9. Read-before-write as the default for partial overlap repeats the Q1.4 error: a value the oracle gives for free and the hardware does not keep.
- Evidence, verified: act-backend/act_backend/generic/src/isel/extractor/inplace_fix.rs:6-17,58-80; act-oracle/taidl_to/instruction_codegen.py:162-181; act-backend/tests/test_extraction.py:405-414 (same buffer, same start text, mark), 507 (`load_d2` cross-buffer mark); act-backend/examples/mini_trn/mini_trn.py:97-99; taidl/taidl/instruction.py:163-172; act/plans/language/spec-draft/instructions.md:47-53 (S5.5).
- Evidence, debater-fetched: RVV spec section 5.2 overlap rules (Fable, fetched); AMX `tdpbf16ps` read-modify-write of `tsrcdest` (Fable, fetched); the AMX #UD condition is from memory and not decisive.
- Dissent: Sonnet defines partial overlap by read-before-write as zero-cost and observationally harmless; Opus faults every unmarked overlap, exact included.
- Confidence: medium-high on keeping the mark and on unmarked exact overlap; medium on faulting partial overlap.

## Q1.4: initial contents of on-chip buffers

- Decision: option C: "A read of an element of an on-chip buffer that no earlier issue of the kernel wrote is a fault; `d0` holds the caller's `hbm` contents in full."
- 1. Zero fill promises a value scratchpads do not hold, so an oracle pass would certify a kernel that reads stale storage.
- 2. Every region is a function of host integers under R28, so the oracle tracks the written set exactly on the host with a boolean mask per buffer, at no device cost.
- 3. Backends already emit a zeroing or overwriting first issue (`tilezero`, `zero za`, overwrite mode), so a backend plan item states the duty.
- 4. "Written" means written during the kernel, so a stale read of a reused buffer is not caught, and the spec says so.
- Evidence, verified: act-oracle/taidl_to/templates/templates.txt:469-470 (`sh_zeros` prologue).
- Evidence, debater-cited: NKI `nc_matmul` states accumulate on uninitialized PSUM is undefined behavior (Fable, not fetched this session).
- Dissent: none.
- Confidence: medium-high.

## Q1.5: float accumulation norm

- Decision: for `dot` and `reduce_add`, `|r - exact| <= gamma_n * sum|terms| + u * |exact|`; for `reduce_multiply`, `|r - exact| <= gamma_n * |exact|`; `gamma_n = n*u / (1 - n*u)`, u the unit roundoff of the result type, n the term count (K for `dot`, the product of the reduced sizes for a reduce).
- Overflow clause: "When `sum|terms|` exceeds the largest finite value of the type, the result may also be an infinity or a NaN."
- The accumulation sentence of R8 is dropped; `reduce_max` and `reduce_min` stay exact.
- 1. StableHLO leaves reduction order implementation-defined, so no bitwise norm is inherited (OD14), and "accumulate in the result type" is a procedure no test observes.
- 2. The summation bound holds for every order and every accumulation at least as wide as the result type; the `u * |exact|` term covers the final rounding after a wider accumulation.
- 3. A product's error is relative, so one `sum|terms|` form is dimensionally wrong for `reduce_multiply`, and two forms are needed.
- 4. For `n*u >= 1` the bound constrains nothing, which is the honest statement; the author writes a wider result type, which `sh_dot` supports since operand and result types are independent.
- 5. Sonnet's f32 floor on u is rejected: it hard-codes one hardware width into a language norm and is a test strategy, not a bound derived from the result type.
- 6. A mixed-sign sum near the largest value is finite, infinite, or NaN by order, so the clause must admit NaN, and the oracle may stay bit-exact wider inside the norm.
- Evidence, verified: act-oracle/taidl_to/generic/utils.py:603-607 (`precision = [HIGHEST, HIGHEST]`, result converts down); StableHLO spec, `reduce`: "The order of reductions is implementation-defined".
- Evidence, debater-cited: Higham, Accuracy and Stability of Numerical Algorithms, sections 3.1 and 4.2; AMX `tdpbf16ps` RNE per FMA (Fable, fetched).
- Dissent: Sonnet's f32 floor; Fable's single form, conceded in round 2.
- Confidence: medium.

## Q1.6: rounding and denormals

- Decision: rounding A: "Every float operation rounds once to its result type."
- Denormals: kept, one norm, verified by a per-word test on the pinned jaxlib on both platforms before the spec sentence is final; a unit that flushes writes the flush as a `select` against the smallest normal.
- 1. `xla_allow_excess_precision` defaults to true, so a bf16 chain may stay at f32 between lines; the oracle sets it false on its own compile options and tests a bf16 chain with an f32 mul-add.
- 2. `xla_cpu_ftz` defaults to true ("always flush denormals"), so the CPU oracle flushes today and must set the flag false; the GPU default is a test item.
- 3. A per-class table is two norms, which OD16 excludes; the hardware deviations (AMX, Arm BF16 flush) are per-ISA lines, which Fable conceded.
- Evidence, verified: XLA debug_options_flags.cc (`set_xla_allow_excess_precision(true)`, `set_xla_cpu_ftz(true)`); act-backend/examples/mini_trn/mini_trn.py:101-104.
- Evidence, debater-cited: act-oracle decorator.py:54 (`CompileOptions` built by the oracle; not re-read this session).
- Dissent: none in final positions.
- Confidence: medium-high on rounding, medium on denormals.

## Q1.7: NaN, signed zero, special values

- Decision: NaN A: "Any NaN equals any NaN; sign and payload are outside the meaning, except that `bitcast_convert` of a NaN from a float source gives the positive canonical quiet NaN bits of that type."
- Zero C: `maximum` and `minimum` follow IEEE 754-2019, NaN propagates and -0 < +0.
- Special values E: inf, zero, and NaN results of every approximate word match the IEEE function exactly.
- 1. Default NaN bits differ between x86 and Arm, and the oracle ships on both, so a canonical NaN after every operation (B) costs a select per word for no hardware truth.
- 2. An unstated payload at `bitcast_convert` flows into integer words and indices, so it is undefined behavior by another name; one select on that word closes it, which Fable conceded.
- 3. `xla_cpu_enable_fast_min_max` defaults to true with Eigen NaN behavior, so the oracle sets it false and tests `maximum(-0, +0)` in both operand orders.
- 4. A ulp tolerance is undefined at an infinity, so special values must be exact to be testable.
- Evidence, verified: XLA debug_options_flags.cc (`set_xla_cpu_enable_fast_min_max(true)`).
- Evidence, debater-cited: IEEE 754-2019 clauses 9.2.1 and 9.6; WebAssembly deterministic profile.
- Dissent: none in final positions.
- Confidence: medium-high.

## Q1.8: convert out of range

- Decision: option A per class: float to float and int to float round to nearest even and overflow to a signed infinity, or NaN where the type has none (`f8e4m3fn`); float to int truncates toward zero, saturates, and maps NaN to 0; int to int wraps; to `pred`, nonzero including NaN is true.
- 1. StableHLO calls the out-of-range cases TBD, so the norm is this language's and is tested on the pinned XLA.
- 2. Faults (B, D) make tensor data fault, which Q1.9 rules out.
- 3. Saturation everywhere (C) invents a rule IEEE lacks and would make the oracle's `ml_dtypes` conversions non-conformant; a saturating ISA writes `clamp` before `convert`.
- Evidence, verified: StableHLO spec, `convert`: "the behavior is TBD" for inexact and out-of-range cases.
- Test item: `f32` to `f8e4m3fn` overflow gives NaN in the pinned `ml_dtypes`.
- Dissent: none.
- Confidence: medium-high.

## Q1.9: do tensor data fault

- Decision: option A, one sentence: "A fault depends only on attributes, state, and region text; every tensor value gives a fixed result, and in-semantics indices clamp as StableHLO states."
- 1. A data-dependent fault needs a device flag threaded through every later issue and commit-after-check, doubling memory traffic per issue.
- 2. The backend never sees tensor values, so under B or C "never emits a faulting issue" (Q1.12) is unsatisfiable.
- 3. Integer `divide` and `remainder` by zero are implementation-defined in StableHLO, so the oracle lowers the fixed values explicitly and tests them.
- Evidence, verified: StableHLO spec, `dynamic_slice`: `adjusted_start_indices = clamp(0, start_indices, shape(operand) - slice_sizes)`; `gather`: `full_start_index[d_operand] = clamp(start_index[d_start], 0, dim(operand, d_operand) - slice_sizes[d_operand])`; `divide`: a zero `rhs` is implementation-defined.
- Dissent: none.
- Confidence: high.

## Q1.10: literals

- Decision: options A and D: a float literal is its exact decimal rounded once to nearest even; a finite literal that rounds to an infinity (NaN at `f8e4m3fn`) is rejected; underflow rounds normally; a backtick constant outside its element type faults at issue.
- 1. The oracle already rejects a literal past the largest value with the tie rule, so A is the status quo made single-rounding.
- 2. Rounding via f64 rounds twice; `Fraction` or hex bits round once and fix an integer above 2^53 at a float type.
- 3. Rejecting every inexact literal (C) rejects `0.1`.
- 4. A backtick constant depends on state, so it cannot be rejected before a run.
- Evidence, verified: act-oracle/taidl_to/generic/utils.py:400-406.
- Dissent: none.
- Confidence: medium-high.

## Q1.11: reduction start value

- Decision: option A: the reducer's identity at the type; a float sum starts from +0.0, so an all-negative-zero sum is +0.0; size-1 and empty reductions read "the element combined with the identity".
- 1. Every hardware accumulator starts from all-zero bits, which is +0.0, and XLA starts from +0.0.
- 2. Under Q1.5 the bound sees magnitude and not sign, so -0.0 would be unenforceable on `dot` and the norm would differ between `dot` and `reduce_add`.
- 3. Opus conceded in round 2; the stake is one sign bit visible only through `bitcast_convert` or a division.
- Evidence, verified: act-oracle/taidl_to/generic/utils.py:439-451 (`reduce_init`, "a float sum starts from +0.0, as XLA's does").
- Dissent: none in final positions.
- Confidence: medium-high.

## Q1.12: fault guarantees and producer contract

- Decision: atomicity A: "A fault writes no region and no state; the program stops at the first fault, and every earlier issue has committed."
- Constraint failure C: every REQUIRE failure, in a constraint or elsewhere, is a fault, which the backend reads as "never selected"; dynamics.md drops "does not hold".
- Producer E: the backend never emits an issue that faults, and lists each precondition it cannot see (state before R27, `|d0|`) in its coverage doc.
- 1. Under Q1.9 A every fault is decided on host integers before any read, so atomicity costs nothing; under Q1.1 C state commits with the outputs, so no rollback sentence is needed.
- 2. The backend chooses `@a.` values and placements, so it produces those preconditions; "may assume" points the duty the wrong way.
- 3. The closed-loop test, every backend program runs on the oracle and raises nothing, gives the duty teeth.
- 4. The report names the lowest issue index; within one issue any applicable clause id may be named.
- 5. If a later ruling makes tensor data fault, atomicity moves to option B rather than paying commit-after-check.
- Evidence: minutes/rulings.md R55, R4, R10; holes.md H5, H11; act-oracle fidelity.md O23 (debater-cited, consistent across all six files).
- Dissent: Sonnet's whole-program wording, withdrawn in round 2.
- Confidence: high.

## Escalation to the owner

- Sonnet: none against a decision; OD17's "reuses the storage" adds no observable beyond address equality, so the mark's value is the checked pairing of Q1.3.
- Opus: confirm one consequence of keeping `set_inplace`, that an unmarked input-output overlap at issue is a fault rather than a read-before-write value; this panel rules the exact case legal under OD22 and the partial case a fault, so only the partial case needs the owner's eye.
- Fable: none.
