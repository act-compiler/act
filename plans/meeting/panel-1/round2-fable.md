# Panel 1, round 2, Fable (hardware architect)

Lens unchanged: a TAIDL semantics specifies architectural state, so "what hardware does" is the architecturally visible result of one issue, never a pipeline detail.
Verification note: the AMX operation line and the RVV 5.2 overlap rule were fetched this session; the Intel SDM #UD condition and the RVV `vslideup` restriction are cited from memory and marked so.

### Q1.1

- Strongest opposing argument (Opus, option C): the pre-state is strictly more expressive, since a post-state value is the update text substituted into the expression, while an overwritten pre-value cannot be recovered; the mixed view already hides a bug where `load` guards `@s.row + @c.n <= 32` on the old row but writes at the new row (act-oracle/tests/test_state.py:24-28, read this session).
- Concession: Opus is right on expressiveness, and my round-1 shadow-variable remedy is a workaround for a loss C never incurs.
- Hardware reading: an instruction reads architectural state at issue and writes it back at retire, which is C; post-increment DMA addressing (start `@s.ptr`, update `@s.ptr = @s.ptr + 64`) is written directly under C and needs `@s.ptr - 64` under A.
- Sail models read source registers at the start of the execute clause and write them as effects, which is the same discipline.
- Against Sonnet's D: the test_state.py:24-28 case shows the mixed view lets a guard and a write read two values of one variable, which a reader cannot see without S5.6.
- Cost: the oracle's starts read the post-state today (agenda, `instruction_codegen.py:732-737`), so C moves the starts and not the constraints; the SME `smstart` and `ld1w_zah` pair reads identically under A and C (casestudies/sme.md:18-30).
- Final position: option C, every expression of an issue reads the pre-state, and updates commit with the outputs; changed from A.
- Confidence: medium-high.
- Divides the panel: Sonnet's D keeps the oracle's current emitter order; the panel must weigh zero oracle change against the two-value hazard.

### Q1.2

- All three agree on option B, checked i64 per operator in both generators; Opus's sentence "every operator of an expression yields a signed 64-bit integer, or the issue faults" is the one to adopt, and Sonnet's catch (check the parse tree, not the canonical sum-of-products form, `utils.py:54-65`) is an oracle plan item.
- Final position: option B; unchanged.
- Confidence: high.

### Q1.3

- Strongest opposing argument (the owner's input, and option C in H1 and S5.5): identical region text decides overlap, not the read-after-write question; "may reuse" and "must reuse" are distinct contracts, and the mark is kept unless shown redundant.
- Restating the hardware facts I now rely on.
- Must reuse, one encoding field: AMX `tsrcdest.row[m].fp32[n] := tsrcdest.row[m].fp32[n] + tmpf32` (felixcloutier.com tdpbf16ps, fetched), SME `fmopa` into a ZA tile, Hopper `wgmma` with `scale-d` reading and writing the same accumulator registers, Gemmini `compute_acc` on `accm` (casestudies/gemmini.md:54-58), NKI `nc_matmul` accumulating into its PSUM destination; the destination is the source by construction and no second field exists.
- May reuse, two encoding fields: RVV `vadd.vv vd, vs2, vs1` allows `vd == vs2` when EEW is equal (RVV spec 5.2 "the destination EEW equals the source EEW", fetched), and every register ISA allows `r0 = r1 + r0` by naming `r0` twice; the value is read-before-write and the allocator is free.
- Must not reuse, two fields: RVV widening ops allow overlap only in the highest-numbered part of the destination group and reserve every other encoding (RVV 5.2, fetched); AMX raises #UD when `srcdest`, `src1`, or `src2` coincide (Intel SDM, from memory); `vslideup` forbids `vd` overlapping `vs2` (from memory).
- Partial overlap: RVV reserves it for widening and narrowing ops outside the two allowed positions, AMX #UDs, and a streaming systolic matmul writes garbage; I found no shipped ISA whose partial input-output overlap is architecturally defined, and my round-1 in-place lane shift is exactly the `vslideup` case the one ISA that has it forbids.
- What identical text can and cannot express.
- Identical text expresses must reuse: the region is one storage by definition, OD17 holds, and the allocator gets the R21 equality.
- Distinct attribute text expresses may reuse: `@a.dst` and `@a.src2` may evaluate equal at issue, the value is read-before-write, and the backend decides.
- Identical text cannot express a two-field encoding whose fields must be equal, nor say which input is the destructive one at a glance; a mark does both, and the spec draft's "equal sizes by tokens, fault at issue when starts differ" (S5.5, S1.3.6) is that contract.
- Redundancy verdict: the mark is redundant only if no ISA has a two-field must-equal encoding; I cannot show that absence, and the examples above show the mark reads as the hardware documents read ("srcdest" is a marked operand), so the mark is not shown redundant.
- The read-after-write hazard: TAIDL's buffer model has no physical register file, so "may or may not reuse the source storage" is microarchitectural and invisible; the architectural question is only whether an input region read at issue sees the pre-state, which read-before-write answers for every overlap.
- The remaining design choice is the default for unmarked overlap, and hardware is clear: exact overlap is legal on same-width ops and partial overlap is reserved or #UD everywhere.
- Rebuttal to my own round-1 step 5 and to Sonnet step 3: faulting on partial overlap forbids nothing real, since no example or case study has one, and it spares the author of a systolic or AMX-like ISA a disjointness constraint on every instruction (R58 makes each one a hand-written line).
- OD16 is met: a fault is a defined outcome, and the oracle tests it on host integers per dimension at zero device cost.
- Final position: keep `set_inplace` as the must-reuse mark (storage equality, R21, fault at issue when the evaluated regions differ); an unmarked input whose evaluated region equals an output exactly is may-reuse, legal, read-before-write; any partial input-output overlap is a fault; changed from A.
- Confidence: medium-high on the three cases, medium on fault versus read-before-write for partial overlap.
- Divides the panel: Sonnet and Opus define partial overlap by read-before-write and remove the mark; the decisive evidence is whether any ISA defines partial overlap, and I found none.

### Q1.4

- All three agree on option C, a read of an on-chip element no earlier issue wrote is a fault, with `d0` fully written by the caller's `hbm`; Sonnet's NumPy bool mask per buffer versus an interval set is an oracle implementation detail, and the mask is the right structure for multi-dimensional buffers.
- One added norm sentence from Sonnet step 6 belongs in the spec: "written" means written during the kernel, so a stale read of a reused buffer is not caught.
- Final position: option C; unchanged.
- Confidence: high.

### Q1.5

- Strongest opposing argument (Opus, and Sonnet step 4): DB6's `n * u * sum|terms|` is the wrong form for `reduce_multiply`, whose error is relative to `|exact|`; the Higham form `gamma_n = n*u / (1 - n*u)` is the one that holds for every order, and a final rounding after wider accumulation needs its own `u * |exact|` term.
- Concession: both corrections are right; my round-1 bound was the first-order form of Higham's and omitted the product case.
- Hardware check holds: AMX accumulates FP32 with RNE per FMA (fetched), Trainium accumulates float32 internally, TPU DEFAULT accumulates bf16 in f32, so every f32-result unit meets the bound with `u = u32`, and a bf16 result from one rounding of an f32 accumulator meets it through the `u * |exact|` term.
- Sonnet's overflow gap is real and OD16 forbids "any value of the type"; the defined clause is: when the bound interval around `exact` crosses the largest finite value of the type, an infinity of the sign of `exact` also conforms.
- Sonnet step 5 stands: for `n*u >= 1` the bound is vacuous, and the author writes a wider result type (`bf16 x bf16 -> f32`, then `convert`), which `sh_dot` supports (`utils.py:604-607`).
- Hopper fp8 `wgmma` reduced-precision accumulation may miss the f32 bound; a per-instruction tolerance is a panel 2 construct and not a reason to loosen the default.
- Final position: option C in Opus's two-form Higham statement, plus the overflow clause above, for `dot`, `reduce_add`, `reduce_multiply` only; refined, not changed.
- Confidence: medium-high.
- Divides the panel: the wording of the overflow clause; whether the oracle may stay bit-exact wider (Opus step 9) is uncontested since it lies inside the norm.

### Q1.6

- Rounding: all three agree on option A, one rounding per operation to the result type; Sonnet's point that the flag belongs on the oracle's own `CompileOptions` (`decorator.py:54`) rather than process `XLA_FLAGS` is the right mechanism, and the bf16 chain test must include an f32 mul-add for FMA contraction.
- Strongest opposing argument on denormals (Sonnet, with Opus): keep (C) is IEEE's default, one sentence, and enforceable per compile with `xla_cpu_ftz=false`, while a per-class split is a second sentence and flush "every input and result" cannot be enforced inside library kernels.
- Concession: my per-class split is two norms, which OD16's "one simple norm" excludes; the hardware facts stay (AMX flushes both ways, fetched; Arm BF16 standard behaviour flushes; OCP MX keeps subnormals at every FP8 format), but they are deviations the ISA author writes explicitly, as my own round-1 step 6 already allowed.
- The dot-class bound absorbs an input flush only when terms are normal; an all-subnormal dot on a flushing unit misses it, so a flushing ISA writes the flush on its inputs before the `dot`.
- Final position: rounding A set per compile; denormals C (keep) as the one norm, with option E's pinned-jaxlib test as the verification step on both platforms; changed from the per-class split.
- Confidence: medium-high on rounding, medium on denormals.
- Divides the panel: nothing material once the test is a plan item.

### Q1.7

- Strongest opposing argument (Opus and Sonnet): leaving `bitcast_convert` of a NaN non-bitwise makes it an arbitrary integer, which is undefined behavior by another name under OD16; fix it at that one word with the positive canonical quiet NaN of the source type, one select on that word only.
- Concession: correct; my round-1 "say so in one sentence" named the hole and did not close it.
- Hardware consequence the spec must state: x86-style units produce a negative default NaN and Arm a positive one, so a hardware kernel that bitcasts a NaN differs from the oracle unless the ISA author documents the deviation or the backend inserts the canonicalizing select.
- Zero: IEEE 754-2019 `maximum` and `minimum` (option C) is what Arm `fmax` and XLA compute; Sonnet's `xla_cpu_enable_fast_min_max=false` per compile is the oracle item.
- Special values: option E, a table of one line per word on both platforms, as Sonnet lists.
- Final position: NaN A with canonical positive quiet NaN at `bitcast_convert`; zero C; special values E; refined on the bitcast point.
- Confidence: medium-high.
- Divides the panel: nothing material.

### Q1.8

- All three agree on option A per class, lowered explicitly in the oracle, with Opus's addition that to `pred` nonzero including NaN is true, and the note that a saturating ISA writes `clamp` before `convert`.
- Final position: option A; unchanged.
- Confidence: high on integer classes, medium on float to float at `f8e4m3fn` pending the pinned `ml_dtypes` test.

### Q1.9

- All three agree on option A: faults depend only on attributes, state, and region text; tensor data takes fixed values; indices clamp as StableHLO states.
- Final position: option A; unchanged.
- Confidence: high.

### Q1.10

- All three agree on options A and D: single rounding from the exact decimal, a literal that rounds to infinity (NaN at `f8e4m3fn`) is rejected, underflow rounds normally, and an out-of-range backtick constant faults at issue.
- Final position: options A and D; unchanged.
- Confidence: medium-high.

### Q1.11

- Strongest opposing argument (Opus, option B): -0.0 is the only IEEE additive identity, so "the start value is the reducer's identity" is true as written and a size-1 sum returns its element; StableHLO names float addition with +0 as failing the monoid law.
- Rebuttal: every hardware accumulator starts from all-zero bits, which is +0.0 (`tilezero`, `zero za`, PSUM overwrite mode), and XLA's `jnp.sum` starts from +0.0; under B every such unit and the oracle's current `reduce_init` (`utils.py:439-451`) are non-conformant on a sign bit observable only through `bitcast_convert` or a division.
- Sonnet step 2 adds that `dot` would need a -0.0 start too, which no gemm kernel has.
- Opus step 5 concedes the stake is small, so the question is which side pays: B costs hardware conformance, A costs one wording fix, "the element combined with the identity" (SC22).
- Final position: option A; unchanged.
- Confidence: medium-high.
- Divides the panel: Opus's identity argument, which is mathematically right and hardware-wrong.

### Q1.12

- Strongest opposing argument (Sonnet): with Q1.9 A every fault is decided on the host before tracing, so the oracle runs nothing, per-issue atomicity is unobservable, and the spec should say only that a fault ends the program with an error and no output is defined.
- Rebuttal: a hardware precise exception, a simulator, and the oracle in trace mode all observe the state at the faulting issue; "every earlier issue has committed and the faulting issue writes no region and no state" (R55, Opus step 7) is one sentence, costs nothing under Q1.9 A, and keeps the debugging value; the oracle may run nothing as its own choice inside that norm.
- Agreement: all three take C (every REQUIRE failure is a fault, the backend reads it as never selected) and E (the backend is a producer of `@a.` values and placements, and lists unseen preconditions such as state before R27 and `|d0|` in its coverage doc).
- Under Q1.1 C state commits with the outputs, so no rollback sentence is needed (Opus step 3).
- Final position: atomicity A, constraint failure C, producer contract E; unchanged.
- Confidence: high.
- Divides the panel: Sonnet's whole-program wording versus per-issue atomicity; the difference is observable only in a trace or on hardware.

### Escalation to the owner

- None; the Q1.3 position above follows the owner's input, and no owner decision is contested.
