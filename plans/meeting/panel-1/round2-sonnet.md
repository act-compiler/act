# Panel 1, round 2: Sonnet (oracle implementer)

Evidence verified this round:

- Constraints, updates, sizes, inputs, semantics, outputs are emitted in that order (act-oracle taidl_to/instruction_codegen.py:165-181).
- In act-oracle tests/test_state.py:24-28 the constraint reads `@s.row` and the output start `@s.row` follows the update at line 25.
- XLA debug_options_flags.cc sets `xla_allow_excess_precision` true, `xla_cpu_enable_fast_min_max` true, and `xla_cpu_ftz` true ("always flush denormals").
- The `xla_gpu_ftz` default is not in that file and stays a test item.

### Q1.1

- Strongest opposing argument (Opus, with Fable's A as the rival): every expression reads the state before the issue, and updates commit with the outputs.
- It is strictly more expressive: a post-state value is the update text substituted, while an overwritten pre-state value cannot be recovered.
- Fable's option A (post-state everywhere) fails on `smstart`: guard `@s.sm == 0` and update `@s.sm = 1` cannot coexist when the guard reads the post-state.
- Fable's remedy, a guard on the inverse of the update or a shadow variable, fails for a lossy update and costs spare state.
- Concession: my option D is the mixed view, and test_state.py:24-28 shows its defect (guard on the old row, write at the new row).
- Option C removes the defect by one sentence: every expression of an issue reads the state before the issue, and updates take effect with the outputs.
- Oracle cost under C: starts and sizes move to the pre-state and updates become parallel assignment, a breaking change that OD10 allows.
- Fault order under C is a reporting choice only, since no expression sees another's effect.
- Final position: option C (changed from D).
- Confidence: medium-high.
- Still divides the panel: Fable's A against C, on whether a post-state guard is ever needed that substitution cannot write.

### Q1.2

- All three agree on option B: every operator result outside signed 64 bits faults, by checked i64 in both generators.
- Implementation note: the oracle checks the tree as written, never the canonical sum of products (utils.py:54-65).

### Q1.3

- Strongest opposing argument (Opus and Fable, with R5): read-before-write fixes the value of every issue, so aliasing is only identical region text, and `set_inplace` is redundant.
- Owner input: overlap alone does not settle the read-after-write question, and an implementation may or may not reuse the source storage.
- Step 1: the read-after-write question is answered by a rule, not by overlap: an issue reads every input region from the pre-state, then writes every output region.
- Step 2: under that rule, for r0 = r1 + r0 and r0 = r1 + r2, the state after the issue is the same whether the implementation reuses storage or writes a fresh array and copies back.
- Step 3: "may reuse" is therefore an implementation freedom with no observable effect, and no region text needs to express it.
- Step 4: "must reuse" is a program obligation about addresses, so it is what text and marks must express.
- Step 5: identical text expresses "must" by construction: one attribute used in input and output names one storage at every issue, with no choice left.
- Step 6: differing text expresses "may": r2 equals r0 or differs from it, and both give the read-before-write value.
- Step 7: text cannot express "must" for two differently spelled starts (two encoding fields, `@a.dst` and `@a.src`, that hardware requires equal), nor can it assert intent.
- Step 8: a user constraint `@a.dst == @a.src` states the equality but does not say which input pairs with which output (OD17) and does not check buffer and size.
- Step 9: so the mark is derivable (buffer and size equality by text, plus start equality at issue) but is the one place that names the pair and fails fast.
- Step 10: the hazard inside one instruction, hardware that streams a write before a later read, is a property of the unit; the ISA author writes a constraint (R58), and the oracle implements read-before-write.
- Step 11: the oracle loads all inputs before any output store (instruction_codegen.py:165-181), so every overlap already has this value at no cost.
- Concession: I withdraw "remove `set_inplace`".
- Final position: keep `set_inplace` as a checked declaration with a fixed desugaring: the flagged input equals output region k in buffer and size by text (front-end rejection otherwise) and in starts at issue (fault otherwise); after the issue the input region holds the output (OD17).
- Final value rule: any other overlap, identical text included, is read-before-write with no fault and no undefined case; overlapping output regions are a fault.
- Final position, one line: read-before-write for all overlap; the mark is a checked declaration, redundant for value and not redundant as assertion and pairing.
- Confidence: medium-high on the value rule, medium on keeping the mark.
- Still divides the panel: Opus and Fable would remove the mark and derive alias from text; the question is whether a checked pairing is worth a flag.

### Q1.4

- All three agree on option C: a read of an on-chip element no earlier issue wrote is a fault, and `d0` holds the `hbm` contents.
- Residual detail: the oracle keeps a bool mask per buffer, not an interval set, because buffers are multi-dimensional.

### Q1.5

- Strongest opposing argument (Fable): one bound `n * u * sum|terms|` plus one rounding for all three words, with the ISA author loosening it per instruction where hardware (Hopper fp8) misses it.
- Rebuttal on `reduce_multiply`: the error of a product is relative to the exact result, so `sum|terms|` is meaningless there; Opus's `gamma_n * |exact|` is the tight form.
- Concession to Opus: add `u * |exact|` for the final rounding when accumulation is wider than the result; my round-1 form omitted it.
- Adopted form: `dot` and `reduce_add` satisfy `|r - exact| <= gamma_n * sum|terms| + u * |exact|`; `reduce_multiply` satisfies `|r - exact| <= gamma_n * |exact|`.
- Refinement from Fable's hardware evidence (AMX, Trainium, and TPU accumulate in f32): take u in gamma_n as that of the wider of the result type and f32.
- That refinement removes the vacuous case (n * u >= 1 at bf16 for n >= 256) that "any value is allowed" would leave undefined by another name.
- It is testable: XLA CPU bf16 dot and reduce are expected to accumulate in f32, a test item on the pinned jaxlib.
- GPU `dot` must run at highest precision, since TF32 would miss the f32 u (RA5).
- Overflow clause stays: if `sum|terms|` exceeds the largest finite value, the result is any value of the type, or the program states a fixed rule; no panel member covered it.
- Final position: Opus's two bounds with the f32 floor on u; the oracle may accumulate wider inside the norm.
- Confidence: medium.
- Still divides the panel: the f32 floor (mine), the overflow clause, and Fable's per-instruction loosening (panel 2).

### Q1.6

- Rounding: all three agree on option A, with `xla_allow_excess_precision=false` on the oracle's own compile options and a bf16 chain test.
- Denormals, strongest opposing argument (Fable): hardware flushes per unit (AMX, Arm bf16) and keeps for fp8, so the norm should be a per-class table fixed by a test.
- Rebuttal 1: the language norm describes the language; a unit that flushes writes the flush as a `select` on `abs(x)` against the smallest normal, which Fable's own step 6 grants.
- Rebuttal 2: a per-type table is not "one simple norm easy to explain" (OD16).
- Rebuttal 3: the oracle must set `xla_cpu_ftz=false`, since its default is true and the CPU oracle flushes today (verified above); that flag is the single switch that makes keep true.
- Concession: Fable's option E is the right gate: the test on the pinned jaxlib decides between keep and the fallback, and a flush fallback needs a software flush on both inputs and results.
- Final position: keep denormals, enforced by `xla_cpu_ftz=false` and the GPU equivalent, gated on a per-word test (unchanged, now with the flag verified).
- Confidence: medium.
- Still divides the panel: Fable's per-class table versus one keep norm.

### Q1.7

- All three agree on NaN option A, zero option C, and special values option E.
- Strongest opposing point (Fable): `bitcast_convert` of a NaN is a documented non-bitwise point with no fixed payload.
- Rebuttal: the integer result of the bitcast is observable, so an unspecified payload is undefined by another name (OD16); Opus's rule fixes it: a NaN read through `bitcast_convert` gives the positive canonical quiet NaN of the source type, one select on that word only.
- `xla_cpu_enable_fast_min_max` defaults true (verified above), so the oracle sets it false and tests (-0, +0) in both operand orders.
- Final position: NaN A with canonical bits at `bitcast_convert`; zero C with the XLA flag off; special values E.
- Confidence: medium-high.
- Still divides the panel: Fable's open payload at `bitcast_convert` against Opus and me.

### Q1.8

- All three agree on option A per class; a saturating ISA writes `clamp` before `convert`.
- Adopted from Opus: to `pred`, nonzero including NaN is true.
- Test item: `f32` to `f8e4m3fn` overflow gives NaN in the pinned ml_dtypes.

### Q1.9

- All three agree on option A: tensor data never faults, indices clamp as StableHLO does, and integer divide and shifts take fixed values lowered explicitly in the oracle.

### Q1.10

- All three agree on options A and D: one rounding from the exact decimal, a literal that rounds to infinity (NaN at `f8e4m3fn`) is rejected, underflow rounds normally, and an out-of-range backtick constant faults at issue.

### Q1.11

- Strongest opposing argument (Opus, option B): only -0.0 is the IEEE additive identity, so "the start is the identity" is true as written, a size-1 reduction returns its element, and the cost is one constant in `reduce_init`.
- Rebuttal 1 (Fable): hardware accumulators are initialized by writing all-zero bits, which is +0.0, so A describes every unit and the oracle today (utils.py:439-451).
- Rebuttal 2: under Q1.5 the bound sees the magnitude of a zero sum and not its sign, so B gives the sign no enforcement on `dot`, and the norm would differ between `dot` and `reduce_add`.
- Concession: B reads more simply; A needs the sentence "the element combined with the identity", and the stake is a sign bit visible only through `bitcast_convert` or a later division.
- Final position: option A (unchanged).
- Confidence: medium.
- Still divides the panel: Opus's B against Fable and me; the evidence is balanced and the answer is a value judgment.

### Q1.12

- Strongest opposing argument (Opus, matching R55): a fault writes no region and no state, the program stops at the first fault, and every earlier issue has committed.
- My round-1 wording, "no output of the program is defined", reads as undefined and conflicts with OD16.
- Concession: adopt Opus's statement; the oracle meets it by tracing the prefix before the faulting issue when the host check fails, which costs only the error path.
- Fable's conditional (atomicity A holds only if Q1.9 is A) is agreed; with Q1.9 A every fault is host-checked, so atomicity is free.
- A REQUIRE failure is always a fault (option C), and dynamics.md drops "does not hold".
- The backend is a producer of bounds and constraints (option E); the closed-loop test (every backend program runs on the oracle with no fault) gives the duty teeth.
- Report the lowest issue index; within one issue any applicable clause id may be named.
- Final position: A (per-issue atomicity, earlier issues committed), C, E (changed in wording from a whole-program error).
- Confidence: high.
- Still divides the panel: nothing substantive; Fable's B fallback applies only if Q1.9 moves.

## Escalation to the owner

- None. OD16 and OD17 hold as written.
- Note on OD17: "the output reuses the marked input's storage" adds no observable beyond address equality, so the mark's value to the owner is the checked pairing in Q1.3.
