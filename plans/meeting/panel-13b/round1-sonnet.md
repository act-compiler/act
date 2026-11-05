# Panel 13b, round 1, Sonnet (oracle and backend implementer)

Paths: spec = plans/language/spec-draft; oracle = /home/user/act-oracle/taidl_to; backend = /home/user/act-backend/act_backend; front end = /home/user/taidl/taidl.
Lens: what the generated oracle and backend must do on each answer, and what it costs them to do it.
External checks run this session: StableHLO spec.md (raw, main), XLA operation_semantics, ml_dtypes float8.h (main), jax.lax.broadcasted_iota docs, NumPy 2.5.3 probes.
Escalation to the owner: none; no answer below contradicts an owner decision.

## Q13b.1 (G3): generator metadata on d0

Position: no `d0_meta`; add "`d0` carries no generator metadata of its own; an entry about hbm goes in the accelerator's `meta`".
1. OD39 says metadata "may" sit at accelerator, buffer, or instruction level; it does not require every buffer to carry it (owner-decisions.md:33).
2. The `meta` value is a free-form namespace mapping (generator_metadata.md:10-11), so `meta={'allo': {'hbm': {...}}}` already spells an hbm partition; the idiom does not fail.
3. Panel 2's admission bar needs an idiom failure plus two public sites before new syntax (panel-2/decision.md:66-75); there is no site.
4. Neither generator reads metadata (generator_metadata.md:28 neutrality), so the oracle and backend cost of either choice is zero.
5. A `d0_meta` keyword would make `Accelerator` carry per-buffer data that `acc.buffer` carries elsewhere, a second spelling for one concept.
Confidence: high.
Would change my mind: a real act-allo entry that cannot sit in the accelerator mapping because it needs buffer identity.

## Q13b.2 (G7): what `@acc.instruction` returns

Position: a small handle object (class `Instruction`) with a `name`, and a `__call__` that raises a coded error; `acc.encode` takes the handle.
1. `acc.encode(instr, ...)` must reject an `instr` that is not an instruction of `acc` (layers.md S11.3, "is rejected (S9)"); a handle with an owner field makes that one `isinstance` plus identity test.
2. With a bare function, the same test needs a side registry keyed by function identity, and a family loop redefines the name `copy` each trip (grammar.md:50-55), so the Python name alone never identifies a member.
3. A returned plain function runs its body on plain values when called; `read` outside the tracer raises something unrelated (G18), so G7's "raises at the author's line" is not given by "unchanged".
4. Neither generator sees the handle: both read the `Model` from `check` (conformance.md:21), so the cost lands only in the front end, about one small class.
5. The handle gives the corpus and error messages a stable `.name` for family members.
Confidence: medium.
Would change my mind: a reviewer showing the registry approach is smaller in the tracer than a class with `__call__`.

## Q13b.3 (G12): which always-false cases `check` rejects

Position: reject cases (a), (b), (d) as G12 states them; restrict (c) to plain, unequal sizes; extend (a) to a plain start plus plain size over a plain access dimension (not `d0`).
1. The rule for a row is "decidable from the recorded expressions alone" (conformance.md:10 failure policy; instructions.md:123), and each rejected row must be a genuine, correctable cause (OD40).
2. (a) A plain start below 0 or plain size below 1 faults at every issue (S1.3.3, instructions.md:54); the plain overflow case `start + size > dim` has the same certainty and G12 omits it; `d0` is excluded because its size is a run-time byte count (declarations.md:41).
3. (b) Two outputs of one buffer with equal start trees and equal size trees evaluate equal at every issue, and sizes are at least 1, so S1.3.4 faults every time.
4. (d) An input tied to two outputs forces `same(In, Out1)` and `same(In, Out2)` (panel-9/decision.md:20), hence Out1 equals Out2, an S1.3.4 fault at every issue.
5. (c) Equal start trees do not make the sizes decidable: sizes `n` and `m` can evaluate equal under a written `require(n == m)`, and panel 9 states that texts differing but evaluating equal satisfy `same` (panel-9/decision.md:67).
6. So (c) rejects only when every start pair is equal and some size pair is two different plain integers; a symbolic size pair stays an issue-time fault, as F5 and `keep_off` model it (panel-9/decision.md:47).
7. Oracle and backend are unchanged: both already fault or never select at issue (instruction_codegen.py:162-181); the gain is that the backend solver never searches an instruction that cannot run.
8. Cost: four S9 rows, four corpus rows, and no new generator code.
Confidence: medium-high on (a), (b), (d); medium on the narrowing of (c).
Would change my mind: a real ISA that writes a plain-size mismatch under equal starts on purpose and never issues it.

## Q13b.4 (G21): an address held in state

Position: keep the rejection (D5); an address reaches state only through a computational attribute; add that sentence to S5.4.
1. D5 rejects `@a.` in an update because the allocator picks addressing values (language/plan/decisions.md D5; backend plan coverage.md B16).
2. A computational attribute may appear in every place (expressions.md:10-12), so the "set base address" instruction is spelled with a `Comp` attribute and a later start reads the state.
3. The backend excludes any instruction that updates state (spec_checks.py:370-378) and plans updates only after v3 (backend/plan/decisions.md State row), so allowing `Addr` in `update` buys the backend nothing in v3.
4. Allowing it later is a pure relaxation: no accepted specification changes meaning, whereas allowing now and restricting later breaks specifications.
5. When the backend does model state updates, an allocator-chosen value flowing across instructions is a new solver capability (a variable alive across issues); that design belongs with that work, not with the v3 language.
6. The oracle cost is nil either way: attributes are host integers at trace time (decorator.py:203-208).
7. A `Comp` attribute used as an address in the backend is excluded loudly: no shape or payload reads it (spec_checks.py:225-236), so the failure is a printed reason, not a silent wrong placement.
Confidence: medium-high.
Would change my mind: a cited in-scope ISA whose address-into-state instruction the backend must select in the first release.

## Q13b.5 (G23): what a run yields

Position: add G23's sentence: a run yields the `hbm` contents after the last issue, or the fault report.
1. The generated oracle returns slices of the final `d0` value: `kernel_epilogue(f'%d0.{...}', outputs, ...)` and `return` of those results (templates.txt:44-56), and `executable.execute` returns only those (decorator.py:191).
2. On-chip buffers are SSA values inside the module and are never returned, so a corpus case cannot assert them in the oracle today.
3. State lives in the Python-side dict `api.state` at trace time (decorator.py:43), a generator-interface fact, not a language result.
4. Faults are decided on host integers when the kernel body is traced (decorator.py:203-208, Q1.9), so "or the fault report" matches how the oracle fails.
5. The backend closed loop (Q1.12) compares hbm results only, so the sentence matches both consumers.
6. A state-only effect stays testable through hbm, because state steers later starts (expressions.md:14).
Confidence: high.
Would change my mind: a corpus need to assert final on-chip contents; then add a generator-level debug dump, not a language result.

## Q13b.6 (G26) and Q13b.15 (SG31): an instruction with a read and no write

Position: reject it (S9 row), citing S7.1; one answer for both IDs.
1. Panel 2 Q2.9 is binding: a semantics defines at least `%Out0`, and an instruction omits semantics only when it has no regions (panel-2/decision.md:120-126, from R11 in minutes/rulings.md:21).
2. A read with no write has a region and no output, which that rule rejects; allowing it reverses a binding decision.
3. Under Q1.9 tensor values never fault, so every computed value without a `write` is unobservable; the only effects would be S1.3.3 and S1.3.11 faults and state updates.
4. State expressions cannot read region values (expressions.md:4, S6.1 table), so the read cannot feed state; its sole purpose would be to fault.
5. The oracle would emit a module piece with no root value for the instruction; the backend already excludes anything with other than one output (spec_checks.py:22-28).
6. Rejecting is reversible: a real prefetch or touch instruction at two public sites can relax it under Q2.3.
7. The "no lazy or dead value" line of execution.md:35 is unobservable for such a case, so it costs nothing to drop it there.
Confidence: medium-high.
Would change my mind: a public in-scope ISA with a read-only instruction whose fault behavior matters.

## Q13b.7 (SG5): what a plain integer is

Position: a plain integer is exactly `int`; `bool` and NumPy integers are rejected with a message that says `int(x)`.
1. The front end already enforces exactly `int` for dimensions and state (accelerator.py:39, :55) and the draft says so (declarations.md:33, :52); one rule everywhere is simpler than a second class.
2. `bool` is an `int` subclass, so `x + True` or `size=[True]` would silently be 1; exact-type testing is the only reliable guard.
3. NumPy 2 prints `np.int64(4)` as its repr (probed: numpy 2.5.3), and the generators build source text with `repr` in places (instruction_codegen.py:39-41); an un-normalized scalar can reach generated code.
4. The oracle's kernel-call side accepts NumPy integers and rejects bool (decorator.py:65-67), which is the right split: kernel arguments come from NumPy arrays, ISA specification literals do not.
5. Tracing is not the obstacle: a symbolic object defines `__radd__` and NumPy defers to it (probed), so rejection is a policy choice, not a necessity.
6. `ops.constant`: `True` and `False` are values at `pred` only (SG5 text); at `pred` also `0` and `1` (semantics.md:74).
7. The cost of the friction is one error message; `np.prod` users write `int(np.prod(...))`.
Confidence: medium.
Would change my mind: a case study whose shapes are routinely NumPy results and a simple normalization rule (`operator.index`, bool excluded) that the whole panel prefers.

## Q13b.8 (SG6): a plain bool or an empty condition list

Position: reject a plain `bool` in `require`, `any_of`, `all_of`, and reject calls with no condition; no folding. The spelling for a plain check is `assert` or `if` outside the instruction.
1. Folding `True` as "adds no constraint" breaks S6.3: every comparison of a condition is evaluated and a fault in any is a fault (expressions.md:36), so `any_of(True, a // b == 1)` must still evaluate `a // b`.
2. Making folding sound needs a rule about which operands of a folded call still run; that is new semantics for a case no author needs.
3. S3.1 already gives Python control flow on plain values at definition time, so `if n <= 64:` builds the family member (grammar.md:47-48) and `assert` states a plain precondition.
4. A folded `False` is an instruction that no issue can satisfy, which the spec would then reject anyway; rejecting the `bool` itself is one rule instead of two.
5. Generators then never receive a constant condition; the oracle emits one check per constraint (instruction_codegen.py:32-46) and the backend would otherwise handle trivially true rows.
6. Arity: S6.3 says "one or more" (expressions.md:34); an empty call is a typo the architect can correct.
Confidence: medium-high.
Would change my mind: a family-generation pattern where `assert` cannot replace the plain `bool` and the rest of the panel proves a sound folding rule.

## Q13b.9 (SG7): float literals

Position: a plain `float` is its exact binary64 value rounded once to the element type; `fractions.Fraction` is accepted and rounded once from its exact value; no `str` and no `Decimal`.
1. The front end never reads source text (grammar.md:4), so a Python `0.1` is already binary64 when it arrives; Q1.10's "exact decimal" (panel-1/decision.md:144) cannot be met for it.
2. The oracle also starts from a binary64: `mlir_literal` takes a Python `float` and emits `repr(float(value))` (utils.py:393-409); one rounding from that value is the honest sentence.
3. Q1.10 step 2 wants a single rounding and names `Fraction` or hex bits as the means (panel-1/decision.md:146); a `Fraction` is stdlib, exact, and `Fraction('0.1')` parses a decimal exactly.
4. `str` would add a string literal form to a surface whose design rule is typed Python objects (OD41); `Fraction` covers the exact-decimal need.
5. Double rounding only differs for decimals within 2^-53 relative of a midpoint of the narrow type, so the binary64 rule is correct in practice; the `Fraction` path exists for exactness, not for common use.
6. Cost: an exact `Fraction` rounder over (exponent, mantissa) with subnormals and the `f8e4m3fn` top-exponent rule, about 30 lines in the shared shape module, tested against `ml_dtypes` on a sweep; it reuses `float_largest` (utils.py:380-390).
Confidence: medium.
Would change my mind: evidence that no ISA needs an exactly rounded non-binary64 literal (then drop `Fraction` and state binary64 only).

## Q13b.10 (SG9): signatures of the target words

Position: add SG9's table with one change: `ops.iota(element_type, shape, iota_dimension=d)`.
1. `constant` is fixed as `(value, element_type=s64, shape=[])` (resolutions.md:36), element type before shape; `iota` has the same two targets and should keep that order.
2. jax's `broadcasted_iota(dtype, shape, dimension)` puts dtype first (https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcasted_iota.html), and jax is the oracle's pinned reference (operation_meaning.md:10).
3. Single-target words keep the target after the operands: `convert(x, element_type)`, `bitcast_convert(x, element_type)`, `reshape(x, shape)`, `broadcast(x, shape, dimensions=...)` (semantics.md:71).
4. `iota_dimension` is an operation attribute and so a keyword (operations.md S8.2 row `iota`); the oracle lowers it from the attribute dict (oracle_visitor.py:366).
5. `bitcast_convert` takes the element type only: the result shape is inferred from the width ratio (S7.4 "a shape is written only where an operation takes one").
6. A target may also be passed by keyword, as panel 11 lists six words with targets (panel-11/decision.md, "Result types are inferred" bullet).
Confidence: medium-high.
Would change my mind: a panel reading that SG9's shape-first order matches an owner paper's examples.

## Q13b.11 (SG11): symbolic axes

Position: yes, an attribute that names an axis is a plain `int`, with SG11's list; sizes, bounds, paddings, and bit counts may stay integer expressions.
1. S7.4 says the front end infers every result shape (semantics.md:61); for `transpose`, `reduce_*`, `concatenate`, `dot`, `gather`, the result dimension at position k is operand dimension `perm[k]`, and S6 has no indexing or selection of an integer expression, so no tree can state that shape for a symbolic axis.
2. Sizes work: `slice`, `pad`, and `dynamic_slice_sizes` results are plain sums, products, and `//` of the entries, all expressible in S6.2.
3. Today's oracle can run a symbolic axis only because the author writes the result shape by hand and `check_shapes` tests it at the call (oracle_visitor.py:300-316); a test uses `lhs_contracting_dims={`@c.k`}` (tests/test_templates.py:292). That is a v3 break, allowed by OD10.
4. The backend reads a computational attribute from a matched payload only as one integer position (isel_ir2isa_rewrites_rs_generator.py:41, :204-217, egraph.rs:62-77); it never builds a pattern whose topology changes with the value.
5. The oracle already rejects an expression in a scalar operation attribute (oracle_visitor.py:417), so a plain axis loses it nothing; `iota_dimension` is already emitted from a literal (oracle_visitor.py:366).
6. With R28 every attribute is a host integer at trace time, so the oracle can lower symbolic sizes and bit counts per issue.
Confidence: high.
Would change my mind: an ISA whose axis varies per issue and whose result shape does not depend on it (only `broadcast`), shown at two sites.

## Q13b.12 (SG15): result element type of `dot`

Position: option B, restated: `ops.dot` takes an optional `element_type=` target, defaulting to the operand type; do not take option A.
1. StableHLO constrains only `element_type(lhs) = element_type(rhs)` (C13) and builds the sum with `constant(0, element_type(result))`, so the reference leaves the result type free (stablehlo spec.md dot_general, lines 2813-2960); XLA's `DotGeneral` has `preferred_element_type` (https://openxla.org/xla/operation_semantics).
2. The oracle already emits `dot_general` with the author's result type independent of the operands (utils.py:584-610), and the shared shape rule says "any element types" (shapes.py:393-411).
3. The backend dot e-node carries its result type in the payload (egraph.rs:67-68, "then its type"), so an explicit result type is the form it already matches; option A would need it to see through two converts.
4. Joint 1-5 binds the bound to "the result type" and says an author "writes a wider result type" (joint-1-5/decision.md:63; panel-1/decision.md:87); option A moves that work to operand converts and makes the bound's u depend on a wider operand type that no longer appears in the dot line.
5. Integer accumulation `s8 x s8 -> s32` has no option-A spelling that matches an XLA `dot` with `preferred_element_type`; the convert form is a different IR shape.
6. Panel 11's list of target words omits `dot` (panel-11/decision.md); adding one optional keyword is the smallest change.
7. Joint 1-5 would be undone only by an accumulator declaration (joint-1-5/decision.md:80); `element_type=` is the result type, not an accumulator.
Confidence: medium-high.
Would change my mind: an IR-matching test showing the backend matches `dot(convert(a), convert(b))` as cheaply as a mixed-type `dot`.

## Q13b.13 (SG22): which IEEE power

Position: `pow` for `power`, as SG22 proposes, with the zero clause applied to the exact value.
1. StableHLO says floats use "`pow` from IEEE-754" (stablehlo spec.md, power section, line 5104 onward).
2. IEEE 754-2019 clause 9.2 defines three powers; `pow(1, y)` is 1 for any y including NaN and `pow(x, +-0)` is 1, whereas `powr(1, inf)` is NaN.
3. A NumPy probe shows `power(1.0, nan)` and `power(1.0, inf)` both give 1.0, the `pow` family, which is what the pinned oracle's XLA and NumPy references compute.
4. Choosing `powr` or `pown` would make the oracle's own XLA result a violation of the norm.
5. Zero clause: Q1.7 makes zero results exact (panel-1/decision.md:111) while S1.3.19's bound needs a nonzero IEEE result (operation_meaning.md:72); with y the exact value, bound it when y is nonzero and require exact 0 only when y is exactly zero, so no discontinuity sits at half the least subnormal.
6. `erf` and `logistic` have no IEEE function, so "the mathematical limit" is the only definition available.
Confidence: medium-high on `pow`; medium on the zero clause.
Would change my mind: pinned-XLA evidence that `power` special cases differ from C `pow` on the CPU or GPU backend.

## Q13b.14 (SG26): `f8e8m0fnu` and `f4e2m1fn` conversion rows

Position: take SG26's text (nearest by value; 1.5 * 2^k rounds up) and gate the rows with an L17 test against the pinned ml_dtypes and XLA before the spec sentence is final.
1. Read in ml_dtypes main, `float8.h` generic `ConvertImpl`: the mantissa keeps its implicit leading 1 and `RoundBitsToNearestEven` runs on the dropped bits (lines 1647-1668), so a source mantissa of exactly one half rounds up to the next power of two.
2. That is nearest by value with the tie to the greater, matching operation_meaning.md:50 once "by value" is added.
3. The same code returns NaN for NaN, infinity for infinity into a type without infinity (NaN there), NaN for a negative into an unsigned type when not saturating, and NaN for zero into a type with no zero (lines 1568-1604), matching the draft's `f8e8m0fnu` row.
4. Not verified: `float4_e2m1fn` NaN to +0 and saturation, because that header was not fetched, and main may differ from the version the pinned jax requires.
5. The pin matters: SG29 asks to name the ml_dtypes types and the version the pinned jax requires; the conformance test should read it from there.
Confidence: medium.
Would change my mind: a failing pinned-version test on any row.

## Q13b.15 (SG31): same question as Q13b.6, same answer: reject a read with no write (S9 row), by panel 2 Q2.9.
