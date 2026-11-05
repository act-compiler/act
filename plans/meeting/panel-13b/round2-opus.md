# Panel 13b, round 2, Opus (lens: formal soundness and owner decision 16)

Spec paths are under plans/language/spec-draft; meeting paths are under plans/meeting.
The test applied to every rule: its cases partition the inputs, each case has one stated outcome, and no two clauses can demand different results (OD16).
Positions changed in this round: Q13b.6 and Q13b.15 to reject, Q13b.7 to `operator.index`, Q13b.12 to option B, and the tie reading of Q13b.14.

### Q13b.1 (G3): metadata for `d0`

- Agreement: all three give `d0` no `meta=`; the text is Sonnet's, "`d0` carries no generator metadata of its own; an entry about hbm goes in the accelerator's `meta`".
- Soundness: the sentence closes the one open case, a `meta=` aimed at hbm, which no call can carry today (declarations.md:40).
- Confidence: high; nothing divides the panel.

### Q13b.2 (G7): what the decorator returns

- Strongest opposing argument (Fable): layers.md:46 has `encode` take the instruction "as `@acc.instruction` gives it", so the function already serves; a stray call reaches `read`, which S3.2 raises (grammar.md:84); a handle class is the extra type OD41 excludes.
- Rebuttal 1: grammar.md:84 raises only for `read`, `write`, `require`, and `update`, so a legal body that calls none of them, such as a no-op or one that only builds `ops.iota` values (semantics.md:12-13), returns silently when called.
- So G7's "raises at the author's line" is false for some legal instructions unless the returned object's `__call__` raises, which is a handle.
- Rebuttal 2: returning the function leaves `acc.instruction(name='a')(f)` then `acc.instruction(name='b')(f)` with two instructions under one `encode` key, which needs one more S9 row.
- Fable's family point, a name rebound each trip (grammar.md:51-55), holds for both choices equally, so it decides nothing.
- Final position: the decorator returns a handle with `name`; calling it raises a trace error; `acc.encode` takes the handle.
- Confidence: medium-high, up from medium; still divides: OD41's "nothing fancy" against a call rule that is total.

### Q13b.3 (G12): text-decidable rejections

- Strongest opposing argument (Sonnet, Fable): row (b) should need equal start and equal size expressions, as G12 wrote it.
- Rebuttal: S1.3.4 faults on any overlap of two outputs (overview.md:48), and equal starts in every dimension put the start element in both regions unless S1.3.3 faults first (overview.md:47), so every issue faults, as in the applied G13.
- Agreement: rows (a) and (d); the plain start plus plain size past a plain access dimension, `d0` exempt (Sonnet step 2 states it too); and (c) narrowed to unequal plain sizes, where Fable's `2 * m` against `m + m` is one more reason.
- Soundness of a closed list: each row is a sufficient condition for "faults at every issue", so a rejection removes no valid issue (OD40), and no row claims completeness.
- Final position: as round 1, with (b) on equal starts alone.
- Confidence: medium-high; still divides: only the width of (b), where both forms are sound and mine rejects more.

### Q13b.4 (G21): addressing attributes into state

- Strongest opposing argument (Fable): D5's reason is "sizes and shapes must be address-free" (plans/language/plan/decisions.md:12), and an update is neither; spelling an address `Comp` misstates its kind; the check is two passes over recorded trees.
- Concession: the kind misstatement is a real cost, and D5's stated reason does not name updates.
- Rebuttal 1: S6.1 lets a state variable appear in every place, a state update included (expressions.md:12), so `update(b2, b1 + 4)` carries an address from `b1` to `b2`.
- The addressing mark is therefore a least fixpoint over every update of the specification, not two passes, and is the cross-instruction analysis I named in round 1.
- Rebuttal 2: the mark is non-local; adding one set-base instruction turns a size that reads that state, in another instruction already accepted, into an S9 error on a line the author did not change.
- Rebuttal 3 (Sonnet step 4): rejecting now and allowing later is a pure relaxation, so every accepted v3 specification keeps its meaning; the reverse order breaks specifications.
- The oracle meaning is the same under both spellings, since a `Comp` may stand in every place (expressions.md:12).
- Final position: keep D5 and add the S5.4 sentence; record Fable's rule, stated as a fixpoint, as the relaxation for B16.
- Confidence: medium-high; still divides: kind honesty now against a relaxation later.

### Q13b.5 (G23): what a run yields

- Agreement: all three yield the `hbm` contents after the last issue, or the fault report.
- Soundness rider: execution.md:43 commits every issue before the first fault, so the sentence should say that this committed `hbm` is not part of a faulting run's result, and no generator must preserve partial results for conformance.
- Final position: G23's sentence, plus "a faulting run yields the fault report alone".
- Confidence: medium-high; nothing divides the panel beyond the rider.

### Q13b.6 and Q13b.15 (G26, SG31): a `read` with no `write`

- Strongest opposing argument (Fable, and my round 1): prefetches read memory and write nothing, R11 was a rule of the string surface, and rejection fails a real instruction (OD40).
- Concession to Sonnet: panel 2 binds this panel, and Q2.9 states R11: "an instruction may omit semantics only when it has no regions; a semantics defines at least `%Out0`" (panel-2/decision.md:125).
- A read with no write has a region and defines no output, so R11 rejects it; panel 11's surface change (panel-11/decision.md:92) is not on the agenda's binding list and repeals nothing.
- New evidence against my round 1: the only observable effects of such an instruction are S1.3.3 and S1.3.11 faults (Sonnet steps 3-4), and a real prefetch raises neither: "The PREFETCHh instruction is merely a hint and does not affect program behavior" (felixcloutier.com/x86/prefetchh, read this session).
- So modelling a prefetch as a `read` makes TAIDL fault on a prefetch of unwritten memory that the hardware allows, which OD31's last clause forbids.
- Final position: reject with an S9 row citing S7.1 and R11; a prefetch is an instruction with no `read` and no `write` plus its encoding constraints, so semantics.md:13 becomes "changes only state, or nothing".
- Confidence: medium-high; still divides: Fable's reading that R11 lapsed with the string surface, a question of binding scope for the moderator.

### Q13b.7 (SG5): `bool` and NumPy integers

- Strongest opposing argument (Fable): `operator.index` is Python's own integer protocol and returns an exact `int`; `np.int64` from `np.arange` or a CSV is no mistake, so rejecting it fails without genuine cause (OD40); the tree still holds `int`.
- Concession: my round-1 step 2 does not separate the rules, since NumPy wraps first (probed, NumPy 2.5.3: `np.array([2**62]) * 4` gives 0 silently) and `int(x)` returns the same wrapped value.
- Probe: without `__array_ufunc__ = None`, `np.int64(4) + n` reaches `n.__radd__` as a Python `int`, while `size=[np.int64(4)]` reaches the front end as `np.int64`.
- Under the exact-type rule one value is then accepted in one place and rejected in the other; under `operator.index` both are accepted, so consistency favors Fable.
- Probe: `operator.index(np.bool_(True))` raises `TypeError`, so excluding Python `bool` alone excludes both bools.
- Probe: without the guard `np.arange(2) + n` builds an object array element by element; with it the tracer receives the array and rejects it, so the guard is needed under either rule.
- Final position: a plain integer is any value other than a `bool` that `operator.index` accepts, and its value is the `int` returned; the `Model` holds that `int`; S4.3 and S4.4 follow; symbolic types set `__array_ufunc__ = None`; `True` and `False` are values only at `pred`.
- Confidence: medium; still divides: Sonnet's single exact-type rule matching the front end today (accelerator.py:39, :55), though Sonnet named this rule as acceptable.

### Q13b.8 (SG6): plain `bool` and empty conditions

- Strongest opposing argument (Sonnet): folding `True` breaks S6.3, where every comparison is evaluated with no short circuit and a fault in any is a fault (expressions.md:36), so `any_of(True, a // b == 1)` must still evaluate `a // b`.
- Concession: my round-1 step 4, "`any_of(True, c)` adds nothing", drops the faults of `c` and is unsound under expressions.md:36.
- Rebuttal 1: the repair is one sentence: a plain `bool` is a condition leaf with that value that never faults, and every other leaf still evaluates.
- Then `any_of(True, a // b == 1)` faults where `b` is 0 and holds elsewhere, exactly as S6.3 reads.
- Rebuttal 2: rejection makes one helper fail by argument kind; `def fits(s): return s <= 64` yields a `bool` for a plain `s` and a condition for a symbolic one (expressions.md:24), so `require(fits(x))` fails or passes on whether `x` is an attribute, without genuine cause (OD40).
- False: rejected only when the plain leaves alone make a `require` false at every issue, as in `require(False)` or `require(all_of(False, c))`; `any_of(False, c)` stays.
- Final position: fold as a leaf, reject the always-false case above, and reject an empty `require()`, `any_of()`, or `all_of()` as a `TypeError` at the line, the arity rule of grammar.md:26; G19 follows this answer.
- Confidence: medium-high; still divides: Sonnet's "no plain `bool` at all".

### Q13b.9 (SG7): plain float constants

- Strongest opposing argument (Fable): a Python float is binary64 before the surface sees it; hex bits through `bitcast_convert` already spell any exact constant; double rounding needs a tie that no vendor table produces.
- Rebuttal 1: panel 1 Q1.10 binds: a float literal is its exact decimal rounded once, and "`Fraction` or hex bits round once and fix an integer above 2^53 at a float type" (panel-1/decision.md:144-146).
- Hex bits give an exact pattern, not a decimal rounded once: the author rounds by hand, the step Q1.10 gives the language.
- Rebuttal 2: Fable's text is silent on `int`, and Q1.10 makes an `int` above 2^53 round once, so the front end needs an exact rounder anyway.
- Example: `2**53 + 2**29 + 1` at `f32` is `2**53 + 2**30` rounded once and `2**53` through binary64; a `Fraction` feeds the same rounder.
- Norm: one sentence covers all three, "a plain number is its exact value rounded once, and a `float`'s exact value is its binary64 value", the one simple norm OD16 asks for.
- Final position: as round 1, with no `str`, which all three now hold.
- Confidence: high on `int`, medium-high on binary64, medium on `Fraction`; still divides: Fable's view that `Fraction` has no use beyond hex bits.

### Q13b.10 (SG9): signatures and `iota` order

- Agreement: `ops.iota(element_type, shape, iota_dimension=...)`, and every target may be passed by keyword.
- Strongest opposing argument (Fable): a default `iota_dimension=0` costs nothing.
- Rebuttal and final position: StableHLO makes `iota_dimension` a required attribute (openxla.org/stablehlo/spec, iota), a default silently picks an axis at rank 2 or more, and a missing keyword is a `TypeError` at once (grammar.md:26), so it has no default.
- Confidence: medium-high; still divides: only the default, a spelling choice.

### Q13b.11 (SG11): symbolic axes

- Agreement: all three make an axis attribute a plain `int`, with SG11's list plus the `transpose` permutation and the `concatenate` and `reverse` dimensions.
- Soundness: S7.4 infers every result shape, and S6 has no selection over integer expressions, so a symbolic axis leaves the inferred shape unstateable (Sonnet step 1).
- Confidence: high; nothing divides the panel.

### Q13b.12 (SG15): result type of `dot`

- Strongest opposing argument (Sonnet, Fable): StableHLO C13 ties only lhs to rhs, XLA has `preferred_element_type`, the oracle and backend already carry an independent result type, and option A makes the backend see through two converts.
- Concession, on binding text: panel 1 Q1.5 step 4 says `sh_dot` supports a wider result type "since operand and result types are independent" (panel-1/decision.md:87), and joint 1-5 step 8 says "the author writes a wider result type" (joint-1-5/decision.md:63); option A contradicts that premise.
- Soundness rider: B needs its meaning stated, or a mixed `s8` dot is ambiguous, since SG16 wraps every integer result to its type and the products' type is unsaid.
- Proposed S8.4 text: "With `element_type=T`, the exact products are summed exactly; at an integer T the sum wraps to T; at a float T the joint 1-5 bound applies with u of T and n = K; the operands and T are all integer or all float."
- Then `dot(a, b, element_type=T)` equals `dot(convert(a, T), convert(b, T))` whenever both converts are exact, so a backend rewrite either way is sound.
- Final position: option B with the sentence above; the running example becomes `ops.dot(a, b, ..., element_type=f32)`.
- Confidence: medium-high; nothing divides the panel once the meaning sentence lands.

### Q13b.13 (SG22): which IEEE function

- Agreement: `pow` for `power`, `rSqrt`, `rootn(x, 3)`, and the mathematical limit for `erf` and `logistic`.
- Strongest opposing argument (Sonnet, and SG22's text): key the bound to the exact y, "the bound applies when y is finite and nonzero", which avoids a jump in tolerance at half the least subnormal.
- Rebuttal: that text is unsatisfiable at overflow; at `f32`, `exponential(88.8)` has y = 3.6758e38, finite and above the largest `f32` 3.4028e38 (computed this session), so the IEEE result is +inf.
- operation_meaning.md:66 then demands r = +inf exactly, while SG22's bound demands r within k ulps of a finite y, which +inf fails; no r conforms, and `f8e4m3fn` overflow to NaN fails the same way.
- The current text keys both clauses to the IEEE result (operation_meaning.md:66, :72), which partitions the inputs, and Sonnet's tolerance jump is no contradiction, which is all OD16 forbids.
- Final position: keep :66 and :72 as keyed now, and add "within the bound, r may be 0 or a subnormal".
- Confidence: high on the overflow defect, medium-high on exact 0 below half the least subnormal; still divides: whether a tiny nonzero y may give a subnormal r.

### Q13b.14 (SG26): `f8e8m0fnu` rounding and `f4e2m1fn` NaN

- Strongest opposing argument (Sonnet, Fable): ml_dtypes' generic converter keeps the implicit 1, so every tie carries up.
- Concession, verified in ml_dtypes main `float8.h` this session: lines 1616-1619 OR the implicit bit into `normalized_mantissa` ("Normal number (or E8M0)"), and lines 1667-1668 round that value with `RoundBitsToNearestEven`.
- With 0 target mantissa bits the kept bit is the implicit 1, always odd, so 1.5 * 2^k rounds up for every k; my even-exponent reading is wrong for main.
- Remaining risk: this converter is main, and the ml_dtypes the pinned jax requires may differ, so the L17 gate stays.
- Adopted from Fable step 3: ml_dtypes' `mxfloat.h` calls NaN to `f4e2m1fn` implementation-defined, so under OD16 the spec must fix the value itself, not inherit it from the pin.
- Final position: "nearest by value, 1.5 * 2^k rounds up" and "NaN gives +0" are TAIDL's own values; L17 tests the oracle against them; where the pin differs, the oracle lowers the row explicitly.
- Confidence: medium-high; nothing divides the panel beyond the pinned-version test.

## Escalation to the owner

- None.
