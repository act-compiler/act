# Panel 1, round 2: Opus (semantics and spec author)

Rebuttals to round1-sonnet.md and round1-fable.md, under the owner input on Q1.3 and owner decisions 16 and 17.

### Q1.1

- Strongest opposing argument (Fable, option A): one state view is easiest to explain, the spec and oracle already read the post-state for starts, and the SME guard does not discriminate because `smstart` has updates and no constraint while `ld1w_zah` has a constraint and no update.
- Second opposing argument (Sonnet, option D): a constraint is a guard and must be tested before an update can fault, and the oracle already emits constraints first at zero cost.
- Concession: Fable is right that the SME pair does not discriminate; act/plans/casestudies/sme.md:20 shows `smstart` with updates and no constraint, and the guard sits on `ld1w_zah` alone.
- Rebuttal to D: the mixed view is the one the corpus already gets wrong.
- act-oracle/tests/test_state.py:24-28 guards `@s.row + @c.n <= 32` on the old row, updates `@s.row`, and writes the output at `@s.row`, so under D the guard checks one row and the write lands at another.
- Rebuttal to A: Fable's own step 1 says hardware checks a guard at decode on the architectural pre-state, which is option C, not A.
- Post-increment addressing, the common hardware form, uses the old pointer; under C it is start `@s.ptr` with update `@s.ptr = @s.ptr + 64`, while under A it needs `@s.ptr - 64`.
- C also meets Sonnet's point: with every expression on the pre-state, constraints can be tested first, so a guard still precedes any update fault.
- C is one view, so Fable's explainability argument supports it as much as A.
- Final position: option C, every expression of an issue reads the pre-state; updates and outputs take effect together at the end of the issue. (unchanged)
- Confidence: medium.
- What still divides the panel: C versus A (Fable) versus D (Sonnet); the test_state.py:24-28 rewrite is the concrete case that separates them.

### Q1.2

- All three agree on option B: every operator yields a signed 64-bit integer, or the issue faults.
- Adopted from Sonnet: the check walks the expression tree as written, never the oracle's canonical sum of products (act-oracle generic utils.py:54-65), since expansion changes intermediates.
- Final position: option B. (unchanged) Confidence: high.

### Q1.3

- Strongest opposing argument (Sonnet and Fable, option A, which I also held): the oracle already loads every input before any store, so every overlap has the read-before-write value at no cost; a fault defines nothing new; hardware that cannot overlap writes a constraint under R58.
- Fable adds that an in-place shift by one lane is a well-defined partial overlap that a fault would reject.
- Owner input, binding: overlap alone does not settle the read-after-write question; `r0 = r1 + r0` may or may not reuse the source storage; `r0 = r1 + r2` allows `r2` to equal `r0` or not.
- Concession: my round-1 option A conflated two facts that region text cannot separate.
- Region text says where an input and an output are; it cannot say whether the hardware reads the input before it writes the output.
- Identical text fixes one address, so it can only say "same location"; it cannot say "may reuse" (the owner's renamed `r0`) versus "must reuse" (OD17).
- Different text can still evaluate to one location at issue (the owner's `r2 == r0`), so text identity misses the very case where the hazard arises.
- The "may" and "must" distinction is real in practice: JAX buffer donation is a permission the runtime may decline, while OD17's mark is a guarantee (jax.jit `donate_argnums` documentation, cited from memory and not fetched this session; the spec-draft glossary.md:46 already lists "donation" as the rejected synonym).
- So `set_inplace` is not redundant: it is the one place the ISA author states that this hardware reads this input before it writes into the same storage.
- Rebuttal to "read-before-write at zero cost": Q1.4 rejected zero fill on the same ground, a value the oracle gives for free and the hardware does not keep; universal read-before-write on any overlap is the same promise, and Fable's own step 4 says partial overlap on real hardware is often garbage.
- Rebuttal to "write a constraint": after R58 the default would still be the unsafe promise, and every ISA author must remember a disjointness constraint per input; the default should be the safe outcome under OD16.
- Rebuttal to the lane shift: it is expressible as a marked full-region in-place issue whose semantics performs the shift, so no partial overlap is needed.
- The fault is cheap and consistent with Q1.9: regions are host integers, so it is the same host interval test as the output-output overlap fault every source already has (SC4, H2) and Sonnet's Q1.4 mask.
- The norm, in two sentences:
  - "A marked input shares output region 0's storage, is read before that output is written, and holds the output after the issue."
  - "Any other overlap at issue between an input region and an output region, by evaluated addresses, is a fault."
- Consequences:
  - The marked pair keeps S5.5: buffer and sizes equal by text, starts equal at issue or a fault (spec-draft instructions.md:49, conformance.md:30).
  - Identical text without the mark always overlaps, so the front end rejects it and asks for the mark (H1).
  - act-backend/examples/mini_trn/mini_trn.py:97-99 already carries the mark and is unaffected.
  - The backend's "may reuse" of a dead input's storage within one issue becomes illegal unless the input is marked, which is the hazard-safe reading.
  - O10 is answered by OD17: the marked input holds the output after the call.
- Final position: option C made precise with option D's evaluated check: keep `set_inplace` as the read-before-write guarantee; every unmarked input-output overlap at issue is a fault. (changed from A)
- Confidence: medium-high on keeping the mark; medium on faulting every partial overlap rather than a front-end rejection where the overlap is static.
- What still divides the panel: Sonnet and Fable hold A; the deciding question is whether the spec may promise read-before-write for hardware whose author did not state it.

### Q1.4

- All three agree on option C: a read of an on-chip element no earlier issue wrote is a fault; `d0` holds the caller's `hbm`.
- Adopted from Sonnet: a boolean mask per buffer rather than intervals, and the stated limit that a stale read of a reused buffer is not caught.
- Final position: option C. (unchanged) Confidence: medium-high.

### Q1.5

- Strongest opposing argument (Fable): keep DB6's single form `n * u * sum|terms|` plus one rounding for all three words, met by AMX, Trainium, and TPU.
- Rebuttal: for `reduce_multiply` the error is relative, `(1 + delta)^n`, so `sum|terms|` has the wrong dimension; Sonnet's step 4 reaches the same split independently (Higham, Accuracy and Stability of Numerical Algorithms, sections 3.1 and 4.2).
- Concession to Sonnet's step 6: the bound leaves order-dependent overflow undefined, since a mixed-sign sum near the largest value is finite in one order and infinite or NaN (inf - inf) in another.
- Added clause: "When `sum|terms|` exceeds the largest finite value of the type, the result may also be an infinity or a NaN."
- On Fable's flushed-input term: an input-flushing unit is a per-instruction looser tolerance (panel 2), as Fable's own step 5 places fp8 `wgmma`, not a change to the default.
- Final position: `dot` and `reduce_add` within `gamma_n * sum|terms| + u * |exact|`, `reduce_multiply` within `gamma_n * |exact|`, plus the overflow clause. (refined)
- Confidence: medium-high on the forms; medium on the overflow clause.
- What still divides the panel: one form (Fable) versus two (Sonnet, Opus).

### Q1.6

- All three agree on rounding option A.
- Adopted from Sonnet: set `xla_allow_excess_precision` and the denormal flag on the oracle's own compile options (act-oracle decorator.py:54) rather than through process `XLA_FLAGS`, and include a mul-add in the bf16 chain test.
- Strongest opposing argument on denormals (Fable): keep everywhere misdescribes matrix units, since AMX and Arm BFloat16 flush; the norm should be a per-class table.
- Rebuttal: matrix units sit in the tolerance class, not the exact class, and Fable's own step 6 gives the faithful escape, an explicit `select` flush in the ISA's semantics; the exact class keeps denormals, which is one sentence.
- Final position: rounding A; denormals kept in every exact-class word, confirmed by a per-platform test on the pinned jaxlib. (unchanged)
- Confidence: high on rounding; medium on denormals.
- What still divides the panel: Fable's per-class table versus one keep sentence plus the explicit flush idiom.

### Q1.7

- All three agree on NaN option A, zero option C, special values option E.
- Strongest opposing argument (Fable): `bitcast_convert` of a NaN should be stated as the one non-bitwise point rather than given an invented payload.
- Rebuttal: an unstated bit pattern flows into integer words, `compare`, and `dynamic_slice` indices, so two conformant generators diverge without bound; Sonnet's x86 versus Arm default-NaN sign point shows the divergence is real on the oracle's own two images.
- Adopted from Sonnet: the oracle sets `xla_cpu_enable_fast_min_max` false per compile and tests `maximum(-0, +0)` in both operand orders.
- Final position: as round 1, with the positive canonical quiet NaN at float-source `bitcast_convert`. (unchanged) Confidence: medium-high.
- What still divides the panel: canonical bits at `bitcast_convert` (Sonnet, Opus) versus a stated non-bitwise point (Fable).

### Q1.8

- All three agree on option A per class; I keep "to `pred`: nonzero, including NaN, is true".
- Final position: option A. (unchanged) Confidence: medium-high.

### Q1.9

- All three agree on option A: faults depend only on attributes, state, and region text; tensor data takes fixed values and indices clamp.
- The Q1.3 overlap fault obeys the same sentence, since it reads evaluated region addresses only.
- Final position: option A. (unchanged) Confidence: high.

### Q1.10

- All three agree on option A for literals and a fault at issue for an out-of-range backtick constant.
- Final position: options A and D. (unchanged) Confidence: medium-high.

### Q1.11

- Strongest opposing argument (Fable, with Sonnet): every hardware accumulator starts from all-zero bits, which is +0.0, and XLA and `jnp.sum` start from +0.0, so -0.0 would make the oracle differ from all of them on a sign no workload reads.
- Concession: the evidence fits; `tilezero`, `zero za`, and PSUM overwrite all yield +0.0, and my own round-1 step 5 put the stake below a reason to diverge.
- The spec sentence reads "the element combined with the identity" (SC22), which stays true with +0.0.
- Final position: option A, a float sum starts from +0.0. (changed from B)
- Confidence: medium-high.
- What still divides the panel: nothing.

### Q1.12

- All three agree on constraint failure option C and producer contract option E.
- Strongest opposing argument (Sonnet): with every fault decided on the host before tracing, per-issue rollback is unobservable, so the spec should say only that a fault ends the program with an error naming the issue and clause.
- Rebuttal: the two are compatible; per-issue atomicity (option A) defines the machine, and Sonnet's sentence is what the oracle reports.
- Keeping A costs nothing under Q1.9 A and keeps the state at the fault defined for a simulator or debugger, the use case Sonnet named as the one that would change its mind.
- "No output of the program is defined" should read "the program returns an error and no outputs", so no undefined value is named under OD16.
- Final position: options A, C, and E. (unchanged) Confidence: high.
- What still divides the panel: wording only.

## Escalation to the owner

- None against a decision.
- Q1.3 asks the owner to confirm one consequence of keeping `set_inplace`: an unmarked input-output overlap at issue, exact or partial, is a fault rather than a read-before-write value.
