# Panel 13b, round 1, Opus (lens: semantics and numerics author)

Questions are the debate rows in order: instructions.md (G) first, then semantics.md (SG).
Spec paths are under plans/language/spec-draft; line numbers are the draft as read at 10:00 UTC.

### Q13b.1 (G3): metadata for `d0`

Position: `d0` carries no buffer-level metadata; a generator that needs hbm metadata reads it from its namespace of `Accelerator(meta=...)`.
1. Owner decision 39 allows metadata "at instruction, buffer, or accelerator level"; the accelerator level already reaches the generator's entry point, so nothing is unspellable (declarations.md:18-20).
2. A `d0_meta=` keyword adds a knob for one buffer before any generator has a use; engineering.md says abstraction comes at the second use site.
3. Allo partitioning targets on-chip arrays; off-chip bank placement is a link-time connectivity choice, which is accelerator-wide (Allo docs, https://cornell-zhang.github.io/allo/).
4. Proposed text for S4.3: "`d0` carries no generator metadata; a generator reads hbm settings from the accelerator's `meta`."
Confidence: medium-high.
Would change my mind: a generator plan item that needs per-buffer hbm metadata which accelerator `meta` cannot key.

### Q13b.2 (G7): what the decorator returns

Position: the decorator returns an instruction handle (name, attributes, call site); calling it raises at the author's line naming `acc.encode`.
1. Python binds the decorator's result to the name (PEP 318, https://peps.python.org/pep-0318/), so the language may choose either object.
2. Returning the function makes `acc.encode(f, ...)` ambiguous: `acc.instruction(name='a')(f)` then `acc.instruction(name='b')(f)` is legal Python and gives two instructions for one function object.
3. A handle names exactly one instruction, so `encode` resolves by identity and a family member keeps its `name=` (grammar.md:48).
4. Calling the unchanged function would re-run `read` and `write` outside a trace; G18 would then raise from inside the body, far from the author's mistake.
5. Proposed text: "The decorator returns the instruction's handle, which `acc.encode` takes; calling it raises."
Confidence: medium.
Would change my mind: a front-end design where encode keys by function and one function may be decorated only once, enforced by `check`.

### Q13b.3 (G12): text-decidable cases `check` rejects

Position: adopt rows (a), (b), and (d), narrow (c), add one row; S9 lists cases and never promises to find every always-false constraint.
1. Decision 40 rejects problems the architect can correct; an instruction that faults at every issue is one, as conformance.md:75 already does for a plain dimension below 1.
2. (a) A plain start below 0 or a plain size below 1: the start is the evaluated address itself, so OD34's "evaluated address" rule is met.
3. Added: a plain start plus a plain size past a declared access dimension; `d0` is exempt since its size is the run's byte count (declarations.md:41).
4. (b) Two outputs of one buffer whose start expressions all equal: the sizes are at least 1 (instructions.md:54), so they always overlap; size equality is not needed, by the same logic as G13.
5. (c) as written is wrong: size expressions `n` and `m` that differ as text are equal at issues where `n == m`, so `same` can hold.
6. (c) narrowed: equal start expressions and, in some dimension, unequal plain sizes; then `apart` never holds and `same` never holds.
7. Panel 9's F5 corpus row expects a fault (panel-9/decision.md:118); it must use a symbolic size to stay a fault, plus a plain-size row expecting rejection.
8. (d) One input tied to two outputs forces both outputs onto one storage (OD17), which S1.3.4 faults at every issue.
9. A general "false at every issue" rule is undecidable once `*` joins two symbols (nonlinear integer arithmetic), so the list stays closed, as StableHLO lists each verifier constraint by number (https://openxla.org/stablehlo/spec).
Confidence: medium-high.
Would change my mind: reading panel 9 step 9 as a closed list of rejections that this panel may not extend.

### Q13b.4 (G21): addressing attributes into state

Position: keep D5 (plans/language/plan/decisions.md:12); S5.4 says an address held in state comes from a `Comp` attribute.
1. `Addr` and `Comp` carry the same signed 64-bit values; the kind only says where a value may appear and what the backend searches (instructions.md:38-39).
2. So a `set_base(b: Comp)` with `update(base, b)` and a start `base + off` has the same oracle meaning the `Addr` spelling would have.
3. The `Addr` alternative needs a cross-instruction flow check (state used only in starts and constraints), a whole-specification analysis that no rule has yet.
4. The backend excludes every instruction that reads or updates state (panel-11/decision.md, backend checks rule), so the `Addr` spelling buys no compiler feature in v3.
5. Real descriptors pass a base address as a plain host value: `cuTensorMapEncodeTiled` takes `globalAddress` as an argument, not an allocated operand (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html).
6. Revisit with backend state support (B16) and the data-dependent addressing study (L23), which P11 names (P11-state-kinds.md:7).
Confidence: medium-high.
Would change my mind: a backend plan item in v3 that allocates through state.

### Q13b.5 (G23): what a run yields

Position: a run yields the `hbm` contents after the last issue, or the fault report; final state and on-chip buffers are a generator's debug interface, outside conformance.
1. The kernel contract is hbm in, hbm out: today's oracle returns only the declared outputs of `executable.execute` (act-oracle taidl_to/generic/decorator.py:191-195).
2. The backend input and output are HBM tensors, so a backend's correctness is judged only on hbm (Q1.12 producer rule, execution.md:159).
3. State and buffer values stay testable: a corpus case ends with a store whose start reads the state, so the value lands in hbm.
4. Making final state normative would bind every simulator to expose it and every backend to preserve it at kernel end, a duty no plan item states.
5. Precedent: an NKI kernel's results are its output tensors in HBM (https://awsdocs-neuron.readthedocs-hosted.com, NKI programming model).
Confidence: medium.
Would change my mind: a corpus row that cannot express its check through a store.

### Q13b.6 (G26): a `read` with no `write`

Same row as SG31; see Q13b.15.

### Q13b.7 (SG5): `bool` and NumPy integers

Position: a plain integer has type exactly `int`; a `bool` or a NumPy integer is rejected naming `int(x)`; the symbolic types set `__array_ufunc__ = None`.
1. S4.3 and S4.4 already say "type exactly `int`" (declarations.md:33, :52); one rule across the spec beats two.
2. The S6.2 range check (expressions.md:111) relies on Python's unbounded `int`: a NumPy integer has already wrapped before the tracer sees it, so `np.int64(2**62) * 4` reaches the front end wrong (NumPy overflow wraps, https://numpy.org/doc/stable/user/basics.types.html).
3. `np.int64(4) + n` calls NumPy's `__add__` first and may build an object array; `__array_ufunc__ = None` makes NumPy defer to the tracer, which then rejects.
4. Rejection is correctable (`int(x)`), so decision 40 allows it.
5. The kernel-side oracle API accepts NumPy integers (act-oracle decorator.py:65-67), but that is kernel input, not an ISA specification.
Confidence: medium-high.
Would change my mind: a corpus specification that builds shapes from NumPy arrays where `int()` is a real burden.

### Q13b.8 (SG6): plain `bool` and empty conditions

Position: a plain `True` adds no constraint; a plain `False` is rejected as an instruction no issue satisfies; `require()`, `any_of()`, `all_of()` with no condition raise at the author's line.
1. A family loop folds `n <= 64` to a `bool` (grammar.md:47), and conformance.md:66 rejects it even when true: that fails without a genuine cause (decision 40).
2. The G19 row tells authors to use `assert` instead, but Python removes `assert` under `-O` (https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement), so the family check silently vanishes.
3. `False` folded is the same "fault at every issue" class as Q13b.3, so rejection keeps one principle.
4. Inside `any_of` and `all_of` the plain values fold by the usual logic, so `any_of(True, c)` adds nothing.
5. An empty call is a call-signature error, like a Python `TypeError`, so it raises at the line, not in `check`.
6. G19 and SG6 contradict each other; the moderator should decide both rows by this question.
Confidence: medium.
Would change my mind: evidence that silent `True` from an `is` comparison is a common author mistake.

### Q13b.9 (SG7): plain float constants

Position: a `float` is its exact binary64 value, an `int` and a `fractions.Fraction` their exact values, each rounded once to the element type; no `str`.
1. The tracer never sees source text (grammar.md:4), so "exact decimal value" (semantics.md:75) cannot be implemented for a `float`.
2. Panel 1 Q1.10 step 2 (binding) names `Fraction` as a single-rounding spelling, so admitting it keeps Q1.10's intent: `Fraction('0.1')` is the exact decimal.
3. An `int` must not pass through `float()`: today's oracle does (act-oracle taidl_to/generic/utils.py:404-407), so `2**53 + 2**29 + 1` at `f32` rounds to the f64 tie `2**53 + 2**29`, then to `2**53`, while one rounding gives `2**53 + 2**30`.
4. Eigen's `bfloat16(double)` narrows through `float`, a second rounding; L17 should test `1 + 2**-8 + 2**-40` at `bf16`, whose single rounding is `1.0078125`.
5. A `str` duplicates `Fraction` and clashes with `float('nan')` as the special-value spelling (semantics.md:75).
Confidence: medium-high on binary64 and `int`; medium on `Fraction`.
Would change my mind: a pinned-stack test showing `Fraction` cannot reach the module without passing through f64.

### Q13b.10 (SG9): operation signatures and `iota` order

Position: `ops.iota(element_type, shape, iota_dimension=d)`; every target may also be passed by keyword; adopt the rest of the SG9 table.
1. `ops.constant(value, element_type=s64, shape=[])` (semantics.md:72) puts the type before the shape; `iota` should match.
2. HLO text writes the type before the dimensions (`s32[4,8] iota()`), the form the `Model` printer emits.
3. JAX's `lax.broadcasted_iota(dtype, shape, dimension)` uses the same order (https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcasted_iota.html).
4. StableHLO makes `iota_dimension` a required input in `[0, rank)` (C1, https://openxla.org/stablehlo/spec), so it has no default.
5. Keyword targets follow G8: every parameter is positional-or-keyword.
Confidence: medium-high.
Would change my mind: a corpus convention already shipping `iota(shape, type)`.

### Q13b.11 (SG11): symbolic axes

Position: adopt SG11; an axis attribute is a plain `int`, and `transpose` `permutation`, `concatenate` and `reverse` dimensions join the list.
1. StableHLO axes are static attributes; a symbolic axis makes the result's dimension order a per-issue fact no static shape module (OD25) can hold.
2. A hardware mode bit that picks an axis becomes two instructions with different fixed encoding bits (S11.4, `fixed=`), so nothing real is lost.
3. Sizes, bounds, paddings, and bit counts change values, not structure, so they may stay expressions under S1.3.7 and S1.3.8.
4. operation_meaning.md:19 then shrinks to "a plain axis outside `[0, rank)` is rejected".
Confidence: medium-high.
Would change my mind: an in-scope ISA whose axis field cannot be split into per-value instructions.

### Q13b.12 (SG15): result type of `dot`

Position: option A: `dot`'s result type is its operands'; a wider result converts both operands first; the running example becomes `ops.dot(ops.convert(a, f32), ops.convert(b, f32), ...)`.
1. StableHLO ties the operands (C13, https://openxla.org/stablehlo/spec); TAIDL must pick a result type, and A picks the one with no new keyword.
2. Every widening `convert` inside S7.3 is exact: `bf16`, `f16`, both `f8`, `f4e2m1fn`, and `f8e8m0fnu` all embed in `f32` (2^-127 is an `f32` subnormal); `s8` embeds in `s32`.
3. So A loses no meaning: `dot(convert(a), convert(b))` at `f32` has the same exact value, the same n = K, and the same u as a mixed `dot` at `f32` under the joint 1-5 Point B bound.
4. The same idiom covers `reduce_add`, which has no target either; B would give `dot` a keyword `reduce_add` lacks.
5. Joint 1-5 Point B is unaffected: neither option declares an accumulator.
6. The running example today rounds the product sum to `bf16` before `convert` (instructions.md:23), a bound of u = 2^-8, not the AMX `f32` behavior its name implies.
7. Backend cost: one sound rewrite, `dot(a, b) : T` mixed equals `dot(convert_T a, convert_T b)`, sound because the converts are exact.
Confidence: medium.
Would change my mind: backend evidence that the convert-wrapped pattern defeats matching of `preferred_element_type` kernels.

### Q13b.13 (SG22): which IEEE function

Position: name StableHLO's functions: `pow` for `power`, `rSqrt` for `rsqrt`, `rootn(x, 3)` for `cbrt`, and the exact mathematical limit for `logistic` and `erf`; key exactness to the correctly rounded result.
1. StableHLO's `power` is IEEE `pow`, `cbrt` is `rootn(x, 3)`, and `logistic` is `1 / (1 + exp(-x))` (https://openxla.org/stablehlo/spec); the reference rule (operation_meaning.md:9) already points there.
2. `pow` and `powr` differ at `pow(1, inf)`, so naming one is required for the exact clause to be testable.
3. Replace operation_meaning.md:66 and :72 by: "When the correctly rounded value of y at t is an infinity, a zero, or NaN, r equals it; otherwise r meets the ulp bound, which may give 0 or a subnormal."
4. That settles the underflow edge: below half the least subnormal r is exactly 0; above it the bound rules.
5. XLA's CPU `pow` special cases are a pinned-stack test item (L17), such as `pow(-0, -inf)` and `pow(nan, 0)`.
Confidence: medium-high.
Would change my mind: a pinned-XLA test showing `pow` special cases fail, which would force a fixed table instead.

### Q13b.14 (SG26): `f8e8m0fnu` rounding and `f4e2m1fn` NaN

Position: "nearest by value", so the midpoint is 1.5 * 2^k; the tie direction and the `f4e2m1fn` NaN result are what the pinned `ml_dtypes` and XLA `convert` give, fixed by an L17 test before the text is final.
1. Round to nearest even on a format with 0 mantissa bits rounds by value at the dropped bit, so the midpoint is 1.5 * 2^k.
2. Bit-level ties-to-even would then pick the even exponent field, so 1.5 goes to 2 but 3 goes to 2, not "the greater"; operation_meaning.md:50 may be wrong.
3. `ml_dtypes` is absent from this machine, so neither tie nor NaN rule was run here; that is why the text waits for the test.
4. The OCP MX v1.0 specification leaves NaN to FP4 to the implementation (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), so the pinned stack decides.
Confidence: medium on "by value"; low on the tie until tested.
Would change my mind: the pinned-XLA test result, either way.

### Q13b.15 (SG31, also G26): a `read` with no `write`

Position: legal; its effects are its faults, S1.3.11 and S1.3.3 included, and its state updates; its semantics values are computed and discarded.
1. Real ISAs have such instructions: Hopper `cp.async.bulk.prefetch.tensor` reads global memory into L2 and writes no architectural destination (https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk_prefetch_tensor.html); OD15 includes TMA.
2. Rejection would fail a scenario real architectures allow (decision 31) and reject without a genuine cause (decision 40).
3. State updates cannot depend on tensor data (no new state kind, P11-state-kinds.md:7), so the semantics values are unobservable; tensor values never fault (execution.md:152).
4. The unwritten-read fault still applies, which is the one observable reason to model the read.
5. The backend already excludes instructions with other than one output (panel 11 backend checks), so no backend rule changes.
Confidence: medium-high.
Would change my mind: a showing that every such instruction is better modelled as constraints alone, with the unwritten-read check lost on purpose.

## Escalation to the owner

- None.
