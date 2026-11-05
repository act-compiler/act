# Panel 1, round 1, Fable (hardware architect)

Lens: what systolic arrays, tensor cores, and vector engines actually do, and a norm that describes them faithfully or says plainly where the ISA author must write the deviation.
Standing assumption: a TAIDL semantics is a specification of architectural state, not a pipeline model, so "what the hardware does" means the architecturally visible result of one instruction.

### Q1.1

Position: option A, one view, the state after all updates, for constraints, starts, sizes, and backticks; an update fault precedes any constraint test.

1. Hardware checks a mode guard at decode on the architectural pre-state, as SME raises an exception when a ZA instruction issues with `PSTATE.ZA` clear, so option D is the literal hardware reading.
2. The case study that motivates the guard separates the mode write and the guarded instruction: `smstart` has updates and no constraint, `ld1w_zah` has a constraint and no update, so under A and D it reads identically.
3. The spec already commits the post-state for shape backticks and starts, and the oracle reads post-state for starts, so A changes the fewest sentences and leaves one view to explain.
4. An update is an explicit expression of the pre-state, so a guard on the pre-value of `@s.x` is written as a guard on the inverse of the update, or with a shadow variable when the update is lossy; the cost is one line in a rare instruction.
5. Two views (D) make the clause-id story harder: a reader must know which role each expression plays before knowing which state it sees.

Evidence: casestudies/sme.md:18-30; language/spec-draft/semantics.md:66; semantics/dynamics.md:34-40; semantics/holes.md H3, H11; minutes/rulings.md R7; developer.arm.com/documentation/ddi0616 (SME, PSTATE.SM and PSTATE.ZA exception checks).
Confidence: medium.
Assumption: the SME exception check on `PSTATE.ZA` is cited from the Arm manual by memory and was not fetched in this session.
Changes my mind: a case-study instruction whose update destroys the pre-value it must also constrain, with no shadow variable available because the ISA has no spare state.

### Q1.2

Position: option B, every intermediate outside signed 64 bits is a fault, met by checked i64 arithmetic in both generators.

1. Address generators and control registers in real hardware are fixed-width datapaths; the machine state already fixes `Reg` as i64, so an expression that leaves i64 midway describes nothing the hardware holds.
2. A backend that must honor option A has to carry i128 or bignums through every generated address computation in Rust and C++, which is a cost for no hardware meaning.
3. Checked arithmetic per operator is already what the backend emits and what Python integers plus a range check give the oracle, so B is the sentence both generators meet today.
4. The one program A accepts and B faults is one whose text multiplies past 2^63 and divides back, which is a bug in an ISA description, not a feature.

Evidence: semantics/dynamics.md:10, 18-27; reviews/semantics/defined-behavior.md DB2; language/spec-draft/expressions.md (S6.2); minutes/rulings.md R56; www.felixcloutier.com/x86/tdpbf16ps (fixed-width tile arithmetic with no wider intermediate state, as the model for any fixed-width hardware path).
Confidence: high.
Changes my mind: evidence that a shipped ISA description relies on an intermediate above 2^63 that the author cannot rewrite.

### Q1.3

Position: option A, alias by identical region text, `set_inplace` removed, every other input-output overlap defined by read-before-write and never a fault.

1. Every accumulator in the case studies (PSUM, AMX tile, ZA, Gemmini accumulator, Tensix `Dst`) is a read-modify-write of one storage, which is exactly OD17's alias plus read-before-write.
2. Hardware aliasing is a property of evaluated addresses, not spelling, so option D is the physical truth; but under read-before-write the value of an evaluated-equal overlap is the same whether or not it is called an alias, so A and D agree on every observable value.
3. What differs between A and D is only what the backend may rely on statically; textual identity is the static witness the allocator can enforce as an equality constraint, so A is the right spelling of D for a producer.
4. Partial overlap on real hardware is often garbage, as with a matmul whose output bank overlaps its moving operand mid-stream, but OD16 forbids leaving it open, read-before-write is the only norm an oracle can implement uniformly, and an ISA whose hardware cannot overlap writes a constraint under R58.
5. Faulting on partial overlap (B, C) would make the oracle reject descriptions of hardware whose overlap is in fact well defined, as with an in-place shift of a vector by one lane.

Evidence: casestudies/strengths.md:3-5; casestudies/trainium2.md:31, 109; casestudies/gemmini.md:54, 106; semantics/dynamics.md:41-49; minutes/rulings.md R5, R58; reviews/semantics/defined-behavior.md DB3; www.felixcloutier.com/x86/tdpbf16ps (`tsrcdest.row[m].fp32[n] := tsrcdest.row[m].fp32[n] + tmpf32`, read-modify-write of one tile).
Confidence: medium-high.
Changes my mind: a hardware accumulate whose observable value is not read-before-write (for example a write-then-read race the vendor documents as the architectural result).

### Q1.4

Position: option C, a read of an on-chip element no earlier instruction of the kernel wrote is a fault; `d0` holds the caller's `hbm` contents.

1. SRAM powers up stale, and vendors say so in their contracts: NKI states that `accumulate=True` on uninitialized PSUM is undefined behavior on all targets, and on NeuronCore-v3 accumulating onto PSUM initialized by a non-matmul instruction is undefined on hardware.
2. R12's zero-fill is therefore a fiction the oracle can keep and the hardware cannot; an oracle pass on a kernel that reads unwritten PSUM hides exactly the bug class NKI warns about.
3. Option C costs one interval set per on-chip buffer on the host and no device work; option A costs nothing today and a silent wrong kernel later.
4. A backend emits a zeroing or overwriting first issue (AMX `tilezero`, SME `zero za`, NKI `accumulate=False`), which the ISA already has, so the backend meets C by construction and needs one plan item to say so.
5. `d0` is the kernel's input and must hold the caller's `hbm`; R12's "every buffer" must be narrowed to on-chip buffers.

Evidence: awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.isa.nc_matmul.html (accumulate on uninitialized PSUM is undefined behavior); minutes/rulings.md R12; reviews/semantics/defined-behavior.md DB4; semantics/holes.md H8; language/spec-draft/semantics.md:46 (`d0` as `u8[<size>]`).
Confidence: high.
Changes my mind: a measured oracle cost of the interval tracking that is super-linear in issues, or a case study whose kernel legitimately reads storage it never wrote.

### Q1.5

Position: option C, `|r - exact| <= n * u * sum|terms|` plus one rounding at the result type, for `dot`, `reduce_add`, `reduce_multiply` only; drop the accumulation sentence.

1. No shipped matrix unit accumulates in a narrow result type: AMX `TDPBF16PS` accumulates FP32 with round-to-nearest-even at each step, Trainium's Tensor Engine accumulates in float32 internally, and TPU's default bf16 matmul accumulates in f32.
2. R8's "accumulate in the result type" is therefore false as a description for every bf16-result case and is a procedure no test observes; only a bound is testable.
3. A bound relative to the exact result rounded to the type (R8) is unmeetable under cancellation by any finite accumulator, while `n * u * sum|terms|` is the classical forward bound every sequential or tree summation at precision u meets.
4. Option C with u of the result type is met by AMX, Trainium, and TPU for f32 results, and trivially for bf16 results produced by rounding an f32 accumulator once.
5. Hopper's fp8 `wgmma` path accumulates with reduced internal precision, and research on tensor cores shows truncation rather than rounding inside the accumulator; such a unit may miss the f32 bound, so the ISA author must be able to state a per-instruction looser tolerance, which is a panel 2 construct and not a reason to weaken the default.
6. Option D (oracle accumulates wider, goldens bit-exact) is a test strategy and not a norm; it would declare the real hardware non-conformant for differing in the last bit.

Evidence: www.felixcloutier.com/x86/tdpbf16ps ("Round to nearest even rounding mode is used when doing each accumulation of the FMA"); awsdocs-neuron.readthedocs-hosted.com nki.isa.nc_matmul ("The accumulation precision internal to Tensor Engine is float32"); docs.jax.dev/en/latest/201/precision.html (DEFAULT on TPU is bf16 inputs with f32 accumulation); arxiv.org/abs/2502.15999 and arxiv.org/abs/2206.02874 (tensor-core accumulator truncation); casestudies/hopper.md:88; casestudies/tpu.md:113; reviews/semantics/defined-behavior.md DB6; minutes/rulings.md R8.
Confidence: medium-high.
Changes my mind: a case-study unit with an f32 result that misses `n * u32 * sum|terms|` on a documented workload, which would push the default bound up rather than change its form.

### Q1.6

Position: rounding, option A; denormals, option E as method with a predicted outcome, and the norm then stated per type class in one table, never "either".

1. Every elementwise engine rounds to its output type per instruction (ScalarE writes bf16, AMX writes f32 with RNE per step); excess precision is a compiler artifact, so option A describes hardware and option B describes XLA's default.
2. The oracle must set `xla_allow_excess_precision=false` before jax initializes and test a bf16 chain, or the exact class is unverifiable.
3. Denormal handling in hardware is per unit, not per machine: AMX always treats input denormals as zero and flushes output denormals, Arm's standard BFloat16 behaviour flushes as if `FZ` and `FIZ` were set, while OCP MX defines subnormals for every FP8, FP6, and FP4 element format and NVIDIA's fp8 conversions preserve them.
4. Option D (flush everywhere) therefore misdescribes every fp8 conversion and would zero a quarter of the e4m3 encodings; option C (keep everywhere) misdescribes every matrix unit.
5. The right split is by class: the `dot` and `reduce` class absorbs input flushing when the bound is stated as `n * u * sum|terms|` with a term for flushed inputs, while the exact class keeps denormals at narrow types; whether f32 exact-class ops keep them is decided by the test on the pinned jaxlib, since the XLA CPU executable may run under a flush-denormal scope.
6. An ISA whose unit flushes writes the flush explicitly (`select` on `compare(abs(x), min_normal)`), which is one line and faithful.

Evidence: www.felixcloutier.com/x86/tdpbf16ps ("Output denormals are always flushed to zero and input denormals are always treated as zero"); df.lth.se/~getz/ARM/SysReg/AArch64-fpcr.html (EBF and alternate BFloat16 behaviour, flush as if FZ and FIZ were 1); OCP MX v1.0 spec section 5.3 (subnormal definition and element formats); reviews/semantics/defined-behavior.md DB7, DB8; semantics/dynamics.md:69; semantics/holes.md H6.
Confidence: medium on rounding, low on denormals.
Changes my mind: the pinned-jaxlib test showing one uniform behavior across f32, bf16, and f8e4m3fn, which would make a single-line norm true of the oracle and leave only the hardware deviations to the ISA author.

### Q1.7

Position: NaN option A, zero option C, special values option E.

1. Hardware canonical NaNs differ (x86 default QNaN has the sign bit set, Arm's does not) and propagation picks different operands, so option B names one pattern no two vendors share; A, any NaN equals any NaN with payload outside the meaning, is the only hardware-faithful norm.
2. `bitcast_convert` of a NaN is then the one place the exact class is not bitwise, and the spec should say so in one sentence rather than invent a payload.
3. IEEE 754-2019 `maximum` and `minimum` (NaN propagates, -0 < +0) is what Arm `fmax` and XLA's `maximum` compute; x86 `maxps` is not, and an ISA built on it writes the `select` form, so option C is checkable and D is not a norm.
4. Transcendental units return exact infinities, zeros, and NaNs at the special points because those are table endpoints, so option E costs hardware nothing and gives tests a fixed answer where a ulp bound gives none.

Evidence: semantics/dynamics.md:69-70; reviews/semantics/defined-behavior.md DB14, DB16; openxla.org/stablehlo/spec (compare WEAKORDER: all NaN representations are treated as equal to each other and greater than +infinity); OCP MX v1.0 spec section 4 ("This specification does not differentiate between quiet and signaling NaNs").
Confidence: medium-high.
Changes my mind: a case-study transcendental unit whose documented `exp(-inf)` or `log(0)` is not the IEEE special value.

### Q1.8

Position: option A per class, with the stated note that a saturating ISA writes `clamp` before `convert`.

1. Int to int wraps on every vector unit; float to int truncates toward zero and saturates with NaN to 0 on Arm, NVIDIA `cvt.rzi.sat`, and XLA's emulation on x86; these two classes are settled by hardware consensus.
2. Float to float on hardware splits: `vcvtneps2bf16` yields infinity, while NVIDIA's fp8 `cvt` must use `.satfinite` and OCP MX conversion saturates; the language cannot describe both with one word, so it picks the IEEE one (infinity, or NaN where the type has none, as `ml_dtypes` does for `f8e4m3fn`) and the saturating ISA writes `clamp` first.
3. Option C (saturate everywhere) would make the oracle's `ml_dtypes` conversions non-conformant and hide overflow in f32-to-bf16 chains that hardware reports as infinity.
4. Options B and D make tensor data fault, which costs a device-side flag (Q1.9) for a case hardware itself never traps on.

Evidence: openxla.org/stablehlo/spec (convert: float-to-int fractional part truncated, out of range TBD); lists.llvm.org MLIR NVVM f8x2 conversion patch (conversion to e4m3x2 or e5m2x2 rejected unless the saturation mode is SATFINITE); language/spec-draft/operation_meaning.md:39; reviews/semantics/defined-behavior.md DB9; language/spec-draft/semantics.md:57 (`f8e4m3fn` has no infinity).
Confidence: high on int classes, medium on float to float.
Changes my mind: the pinned `ml_dtypes` saturating on `f8e4m3fn` overflow, which would move that one type to option C to match the oracle.

### Q1.9

Position: option A, tensor data never faults; in-semantics indices clamp as StableHLO does; integer divide, remainder, and wide shifts take fixed values.

1. No accelerator datapath raises a precise exception on data: SIMD integer divide returns a fixed pattern, matrix units have no trap, and a data-dependent error surfaces only as a flag read by a later sync.
2. A fault on tensor data therefore needs the O23 device flag and commit-after-check in the oracle and an equivalent defined result in every backend target, for a case the hardware itself defines by value.
3. StableHLO already fixes the clamp `clamp(0, start_indices, shape(operand) - slice_sizes)` for `dynamic_slice` and `gather`, so option A inherits a stated reference under OD14.
4. The two-norm reading (D) is the honest statement of A: expressions are host integers and may fault at zero device cost, tensor words are device data and take values; it is one norm once phrased "faults depend only on attributes, state, and region text".

Evidence: openxla.org/stablehlo/spec (dynamic_slice `adjusted_start_indices = clamp(...)`, gather `full_start_index = clamp(...)`); language/spec-draft/operation_meaning.md:36-39; reviews/semantics/defined-behavior.md DB10, fault-detection cost section; semantics/dynamics.md:68.
Confidence: high.
Changes my mind: a case-study ISA whose documented architectural behavior on an out-of-range gather index is a trap that the kernel author relies on.

### Q1.10

Position: option A for literals, option D for backticks: a finite decimal that rounds to infinity (or NaN at `f8e4m3fn`) is rejected, underflow rounds normally, and a backtick constant outside its type faults at issue.

1. A literal is assembler-time text; every assembler rejects an immediate that does not fit rather than silently wrapping or saturating it, because the typo is the common case.
2. Accepting an overflow as infinity (B) turns a mistyped `3e40` into an infinity that propagates through the oracle silently.
3. Rejecting every inexact literal (C) rejects `0.1` at every float type, which no author tolerates.
4. A backtick constant depends on state and so cannot be rejected before a run; faulting at issue is the same sentence as R56 and R55.
5. Single rounding from the exact decimal (Fraction or hex bits) is the IEEE rule and removes the oracle's double rounding via f64.

Evidence: language/spec-draft/semantics.md:74-78; minutes/rulings.md R57; reviews/semantics/defined-behavior.md DB15; OCP MX v1.0 spec section 4 (roundTiesToEven per IEEE 754); www.felixcloutier.com/x86/tdpbf16ps (hardware consumes encoded bits only; the rounding of text is the toolchain's job).
Confidence: medium-high.
Changes my mind: a case-study description that needs an infinite literal and cannot spell `inf`.

### Q1.11

Position: option A, the reducer's identity at the type, a float sum starting from +0.0, so an all-negative-zero sum is +0.0.

1. Every hardware accumulator is initialized by writing all-zero bits (`tilezero`, `zero za`, PSUM overwrite or memset), and all-zero bits are +0.0, so a hardware sum over negative zeros is +0.0 under round-to-nearest-even.
2. Option B would make the oracle differ from every accumulator and from XLA on a sign bit that no workload reads.
3. The spec sentence must say "the element combined with the identity" for size-1 reductions, not "the element itself".

Evidence: www.felixcloutier.com/x86/tdpbf16ps (accumulate into an FP32 tile that `tilezero` fills with zeros); awsdocs-neuron.readthedocs-hosted.com nki.isa.nc_matmul (first matmul into PSUM uses overwrite mode); minutes/rulings.md R3; reviews/semantics/defined-behavior.md DB13; language/spec-draft/operation_meaning.md:25.
Confidence: high.
Changes my mind: a case-study unit whose zero-initialization writes -0.0.

### Q1.12

Position: atomicity option A conditional on Q1.9 option A; constraint failure option C; producer contract option E.

1. On hardware a mid-instruction fault is imprecise, so option B is the literal hardware truth; but once every fault depends only on attributes, state, and region text (Q1.9 A), every fault is host-checkable before any write, and A costs nothing.
2. If Q1.9 falls to B or C, A requires commit-after-check with a device flag and sync per issue, and the panel should then take B rather than pay it.
3. A REQUIRE failure anywhere is one outcome, a fault; the backend's "never selected" is the same contract seen from the producer side, so two words for it (D) only cost clause ids.
4. The backend chooses `@a.` values and placements, so it is a producer of bounds and constraints, not an assumer; a compiler never emits an undefined encoding on the theory that the hardware "may assume" it does not.
5. Preconditions the backend cannot see (state before R27, `|d0|`) are listed in its coverage doc, which is the hardware equivalent of a documented programmer's obligation.

Evidence: minutes/rulings.md R55, R4, R10; reviews/semantics/defined-behavior.md DB5, DB11, fault-detection cost section; semantics/holes.md H11; semantics/dynamics.md:27-28; awsdocs-neuron.readthedocs-hosted.com nki.isa.nc_matmul (programmer obligations stated as "must" with undefined behavior otherwise, the contract TAIDL replaces with a fault).
Confidence: high.
Changes my mind: Q1.9 ruled as B or C, which moves atomicity to option B.
