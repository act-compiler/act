# Panel 13b, round 2, Fable: hardware architect across real accelerator ISAs

Code paths are relative to /home/user; spec paths are relative to act/plans/language/spec-draft.
Checks run this round: ml_dtypes float8.h and mxfloat.h (main), the PTX ISA cvt section (ptx-90-parser mirror), Python 3 operator.index and NumPy 2.5.3 probes, act-backend egraph.rs and spec_checks.py.

### Q13b.1 (G3): generator metadata for hbm

Opposing argument: none; all three put hbm settings in the accelerator's `meta`.
Final position: Sonnet's sentence for S4.3, "`d0` carries no generator metadata of its own; an entry about hbm goes in the accelerator's `meta`".
Confidence: high.

### Q13b.2 (G7): what the decorator binds

Opposing argument (Opus, Sonnet): a handle object names exactly one instruction, so `acc.encode` resolves by identity and ownership with one `isinstance`, and a family loop rebinds the Python name each trip, so the name never identifies a member; a bare function decorated twice gives two instructions for one object, and calling it raises an unrelated `read` error from inside the body.
Rebuttal, in part:
1. Each trip of a loop executes `def` anew and creates a distinct function object (probed: two trips, two ids), so function identity does identify a family member; a registry keyed by function object gives `encode` the same one-lookup ownership test a handle gives.
2. Decorating one function object twice is the shape of "one name twice" (grammar.md:59) and is rejected by `check` the same way; Opus's step 2 is a rejection row, not an ambiguity.
Concession: a call on the returned function runs the body on plain values and the first `read` raises G18's message from inside the body; the S3.2 row should name the call ("an instruction function is not called; the tracer runs it once") so the author's frame is the first in the traceback.
Final position: the function object unchanged, plus the S3.2 row and a `check` rejection of a function decorated twice.
Confidence: medium.
Divides the panel: whether a one-method class is cheaper than two `check` rows; both are small, and the moderator decides on OD41's "nothing fancy".

### Q13b.3 (G12): rejection rows

Opposing argument (Opus): row (b) needs only equal start expressions, since both sizes are at least 1 and so overlap at the start index; and a plain start plus a plain size past a plain access dimension is a fifth decidable row, `d0` exempt.
Concession: both are right.
1. Two outputs of one buffer whose start trees are equal in every dimension cover the start index twice at every issue (instructions.md:51, :55), so size equality is not needed; the narrower row of round 1 misses the equal-start, unequal-plain-size pair.
2. A plain `start + size` over a plain access dimension is decided from the recorded trees alone and faults at every issue (instructions.md:55); `d0`'s dimension is a run-time byte count (declarations.md:43), so it stays an issue-time fault there.
Final position: five rows: plain start below 0 or plain size below 1; plain start plus plain size over a plain access dimension, not `d0`; two outputs of one buffer with equal start trees; an exact-or-apart pair with equal start trees and some size pair of two unequal plain integers; one input tied to two outputs.
The F5 corpus row keeps a symbolic size and expects a fault; a plain-size twin expects rejection.
Confidence: medium-high.
Divides the panel: nothing of substance.

### Q13b.4 (G21): an address into state

Opposing argument (Opus, Sonnet): `Addr` and `Comp` carry the same values, so `set_base(b: Comp)` has the oracle meaning the `Addr` spelling would have; the backend excludes every instruction that updates state (spec_checks.py:370-378), so the `Addr` spelling buys no v3 feature; relaxing D5 later is free, while allowing now and restricting later breaks accepted specifications; the flow check is a whole-specification analysis no rule has yet.
Concession: the ordering argument decides it.
1. `check_state` raises on any `update` before any shape search (spec_checks.py:370-378, read), so my "Comp lies about its kind" never reaches the allocator in v3; the lie is harmless until B16, and B16 can then choose the spelling with its solver in view.
2. A relaxation never changes the meaning of an accepted specification; a restriction does, so the asymmetry favors keeping D5 now.
3. Fidelity is kept: Ethos-U base registers and AMX `ldtilecfg` are spelled today as `Comp` into state, and the oracle evaluates them as integers either way.
Final position: keep D5; add Opus's and Sonnet's S5.4 sentence, "an address held in state is loaded from a computational attribute; an addressing attribute never reaches state"; the backend plan's B16 row names the revisit.
Confidence: medium-high.

### Q13b.5 (G23): what a run yields

Opposing argument: none; all three say hbm contents or the fault report.
Opus's addition stands: a state value is tested through a later store whose start reads it, so no corpus check is lost.
Final position: G23's sentence, with the fault report naming the issue index from 0.
Confidence: high.

### Q13b.6 (G26) and Q13b.15 (SG31): a `read` with no `write`, one answer for both

Opposing argument (Sonnet): panel 2 Q2.9 is binding and says a semantics defines at least `%Out0` and an instruction omits semantics only without regions (panel-2/decision.md:125, rulings.md:21), so a read with no output is rejected; such an instruction's only purpose would be to fault, since state cannot read tensor data; rejection is reversible under Q2.3.
Rebuttal:
1. R11 answers "may the semantics text be omitted", a question of the string surface where the text had to name every region and `check_values` reconciled them (taidl/taidl/instruction.py:142-161); panel 11 removed that reconciliation (panel-11/decision.md:92), and in the traced surface a body with a `read` and no `write` has not omitted anything.
2. Panel 2 itself recorded the input-only case as undecided: its Sonnet wrote "whether an input-only instruction with no output region may omit semantics" under "what still divides the panel" (panel-2/round2-sonnet.md:147), and the decision's dissent line does not settle it (decision.md:127).
3. "Its sole purpose would be to fault" is wrong on the hardware: the class is an instruction whose destination lies outside the modeled buffers, such as a PTX `cp.async.bulk` into another CTA's `shared::cluster` memory, a Trainium collective send out of SBUF, or a debug dump to the host; the read is real, and reading SBUF nobody wrote before a send is a real bug that S1.3.11 catches.
Concession: my round-1 example was the wrong one; an architectural prefetch (PTX `cp.async.bulk.prefetch.tensor`, x86 `PREFETCHh`) never faults, so it is spelled with no regions, not as a `read`.
4. Cost stays nil: the oracle loads inputs before any store and a module with no outputs is the trivial case (panel-9/decision.md:105); the backend excludes other than one output (spec_checks.py:22-29).
Final position: legal; SG31's sentence in S5.2 and S7.1, with S1.3.11 named; no S9 row.
Confidence: medium-high.
Divides the panel: whether R11's "with regions it is rejected" binds the traced surface; the moderator reads rulings.md:21 and panel-2/round2-sonnet.md:147 and decides.

### Q13b.7 (SG5): `bool` and NumPy integers

Opposing argument (Opus, Sonnet): exactly `int` is one rule across S4.3, S4.4, and S6.2; `np.int64(4) + n` would run NumPy's `__add__` first; a NumPy scalar can reach generated source through `repr`; `int(x)` is a correctable fix under OD40.
Rebuttal, in part:
1. NumPy defers to a symbolic right operand (probed: `np.int64(4) + S()` reaches `S.__radd__` with a plain `int`), which Sonnet's own probe confirms, so Opus's step 3 does not hold and `__array_ufunc__ = None` is unneeded.
2. Under `operator.index` the recorded tree holds exact `int` (probed: `type(operator.index(np.int64(4))) is int`), so `repr` never sees a NumPy scalar.
Concession: `operator.index(True)` returns `1` (probed), so the rule must exclude `bool` by type before indexing, as round 1 wrote; and "exactly `int`" is the smaller spec text.
Final position: a plain integer is a `bool`-excluded `__index__` value, normalized at the edge; if the moderator weighs one rule over friction, exactly `int` with the `int(x)` message is acceptable to me.
Confidence: medium-low.
Divides the panel: friction for table-driven families versus one type rule; no evidence decides it.

### Q13b.8 (SG6): a plain `bool` condition

Opposing argument (Sonnet): folding `True` breaks S6.3, since `any_of(True, a // b == 1)` must still evaluate `a // b` (expressions.md:38); `assert` and `if` already cover plain checks.
Concession and rebuttal:
1. Sonnet is right inside `any_of`: a plain `True` there makes the other arms dead and a plain `False` is a no-op arm, so a plain `bool` inside `any_of` is rejected.
2. As a conjunct of `require` or `all_of`, dropping a plain `True` removes no comparison, so S6.3's evaluation rule is untouched; that is the family-loop case `require(n * 64 <= 4096)`.
3. Opus's `-O` point stands: `assert` vanishes under `python -O`, so sending the author there loses the check silently.
Final position: a plain `True` conjunct of `require` or `all_of` is dropped; a plain `False` anywhere, and a plain `bool` inside `any_of`, are rejected; an empty `require`, `any_of`, or `all_of` raises at the author's line; G19 is rewritten to match.
Confidence: medium-high.
Divides the panel: whether any folding is worth a rule; Sonnet holds none.

### Q13b.9 (SG7): float literals

Opposing argument (Opus, Sonnet): panel 1 Q1.10 step 2 names `Fraction` as the single-rounding spelling (panel-1/decision.md:146, binding), and an `int` must not pass through `float()`, which today's oracle does (utils.py:404-407).
Concession: both points stand.
1. Q1.10 is binding and names `Fraction` or hex bits; `Fraction` is stdlib and exact, so admitting it is no new mechanism.
2. An `int` is its exact value rounded once; Opus's `2**53 + 2**29 + 1` at `f32` shows the double rounding.
Final position: a `float` is its exact binary64 value, an `int` and a `fractions.Fraction` their exact values, each rounded once; no `str`; L17 tests Opus's `bf16` case `1 + 2**-8 + 2**-40`.
Confidence: medium-high.

### Q13b.10 (SG9): signatures of the creation words

Opposing argument (Opus): StableHLO C1 makes `iota_dimension` required, so it has no default.
Concession and final position: the SG9 table with `ops.iota(element_type, shape, iota_dimension)`, no default, every target passable by keyword.
Confidence: medium-high.

### Q13b.11 (SG11): symbolic axes

Opposing argument: none; all three make an axis a plain `int`.
Final position: adopt SG11 with Opus's list, `transpose`, `concatenate`, `reverse` included; sizes, bounds, paddings, and bit counts stay expressions.
Confidence: high.

### Q13b.12 (SG15): the result type of `dot`

Opposing argument (Opus): every widening `convert` in S7.3 is exact, so option A, `dot(convert(a), convert(b))` at `f32`, has the same exact value, n, and u as a mixed `dot`; it adds no keyword, matches `reduce_add`, and costs the backend one sound rewrite.
Rebuttal:
1. The reference form is B: an XLA kernel from `jnp.dot(a, b, preferred_element_type=f32)` arrives as `dot_general` with `bf16` operands and an `f32` result, and the backend's dot payload already carries the result type (egraph.rs:67-68, read; Sonnet's utils.py:584-610 for the oracle); A makes the backend rewrite every such kernel before matching.
2. The hardware is B: AMX `tdpbf16ps`, PTX `mma.sync` with `.f32.bf16.bf16.f32`, Trainium and TPU matrix units take narrow operands and produce a wide result in one operation; a semantics that converts the operands first describes a unit no vendor built, which the backend then has to see through.
3. Exactness is not the issue: Opus's step 2 shows A is sound, not that it is the shape kernels and units have.
4. Joint 1-5 is untouched by either option (joint-1-5/decision.md:57, :80); the result type is the bound's type in both.
Final position: option B, `ops.dot(a, b, ..., element_type=None)` defaulting to the operand type; the running example becomes `ops.dot(a, b, ..., element_type=f32)`.
Confidence: medium-high.
Divides the panel: whether one backend rewrite is cheaper than one keyword; Sonnet's egraph evidence says no.

### Q13b.13 (SG22): which IEEE function

Opposing argument: none on `pow`; Opus and Sonnet differ on the zero clause.
Opus: when the correctly rounded value of y at t is an infinity, a zero, or NaN, r equals it; otherwise the ulp bound rules.
Sonnet: exact 0 only when y is exactly zero; otherwise the bound.
Position: Opus's form; it is the "rounds once" rule (operation_meaning.md:64) applied to the exact value, so `exp(-200)` at `f32` is exactly 0 and a unit that flushes a subnormal result still fails, which is what line 65's `select` idiom is for.
Final position: `pow`, `rSqrt`, `rootn(x, 3)`, the mathematical limit for `erf` and `logistic`; Opus's sentence replaces operation_meaning.md:66 and :72.
Confidence: high on `pow`; medium-high on the clause.
Divides the panel: the zero clause's edge; the moderator picks a sentence.

### Q13b.14 (SG26): `f8e8m0fnu` rounding and the `f4e2m1fn` NaN

Opposing argument (Opus): bit-level ties-to-even on a 0-mantissa format would pick the even exponent field, so 3 would round to 2, not 4, and operation_meaning.md:53's "tie to the greater" may be wrong.
Rebuttal, from the code:
1. In float8.h `ConvertImpl`, `RoundBitsToNearestEven` computes `bias = ((bits >> roundoff) & 1) + (1 << (roundoff - 1)) - 1`; for a normal source the bit at `roundoff` is the implicit leading 1, so the bias is a constant half and an exact tie carries into the exponent, giving the greater power of two; the exponent field's parity is never inspected.
2. So 1.5 gives 2 and 3 gives 4; the draft's row is right once "by value" is added, and Sonnet's reading (lines 1647-1668) agrees.
3. PTX offers no nearest mode at all: `cvt.frnd3{.satfinite}.ue8m0x2.f32` with `.frnd3 = { .rz, .rp }` (PTX ISA cvt section), so the pinned ml_dtypes and XLA are the only reference and the L17 test is required, not optional.
4. NaN row: mxfloat.h's `float4_e2m1fn::quiet_NaN()` is `FromRep(0b1000)`, which is -0 in that layout, and the generic path returns it for a positive NaN; so the pinned ml_dtypes gives -0 where the draft says +0, and `maximum` tells them apart (operation_meaning.md:68), so the oracle lowers +0 as an explicit `select` and the L17 test checks the sign.
Final position: nearest by value, tie to the greater; keep "NaN gives +0" as TAIDL's fixed value, lowered explicitly; L17 tests both rows against the pinned stack, sign included.
Confidence: medium-high on the tie; medium on the NaN row until the pinned version is run.
Divides the panel: whether TAIDL fixes the `f4e2m1fn` NaN result itself or follows the pin; MX leaves it implementation-defined, so either is defensible, and the pin's -0 is a quirk no hardware document names.
