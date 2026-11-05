# Panel 13b, round 1, Fable: hardware architect across real accelerator ISAs

Code paths are relative to /home/user; spec paths are relative to act/plans/language/spec-draft.
Questions follow the debate rows of the two gap reports, instructions first (G3, G7, G12, G21, G23, G26), then semantics (SG5, SG6, SG7, SG9, SG11, SG15, SG22, SG26, SG31).

### Q13b.1 (G3): generator metadata for hbm

Position: `d0` carries no generator metadata; what a generator needs to know about hbm is an entry of the accelerator's `meta`.
1. OD39 names three metadata levels, instruction, buffer, and accelerator, and hbm is one per accelerator (declarations.md:18, :40), so the accelerator level already addresses it.
2. The one named use, an Allo partition on hbm, is a per-accelerator datum, since Allo partitions the external memory interface once per design (https://cornell-zhang.github.io/allo/, memory customization `.partition`).
3. A `d0_meta=` keyword is a second spelling for one level of OD39 with one use site, which the second-use-site rule excludes (CLAUDE engineering rules; panel-11/decision.md:111 reasoning 9).
4. S12.3 neutrality holds either way: the front end stores entries without reading them (generator_metadata.md:19).
Confidence: medium-high.
Would change my mind: a second generator whose per-buffer schema must treat `d0` like other buffers, so that a special accelerator-level key duplicates its buffer schema.

### Q13b.2 (G7): what the decorator binds

Position: the decorator returns the function object unchanged, which `acc.encode` takes; calling it outside the tracer raises at the author's line through the S3.2 row of G18.
1. `acc.encode(instr, ...)` "takes the instruction function" (layers.md:41), so the function is already the handle the surface uses.
2. A family rebinds the same Python name each trip (grammar.md:51-55), so a handle object would be lost on every trip but the last; the `Model` keys instructions by declared name (declarations.md:13), not by object.
3. A called body reaches `read` outside the tracer, which the G18 row raises; no second mechanism is needed.
4. Returning the function unchanged is Python's registration idiom; PEP 318 lets a decorator return any object, and `atexit.register` returns its argument so the name keeps meaning (https://docs.python.org/3/library/atexit.html).
5. A handle is an object type with one method and no second use, which OD41's "nothing fancy" excludes.
Confidence: medium-high.
Would change my mind: a generator that needs per-instruction methods on the decorated name, such as `gemm_acc.encode(...)`, which panel 11 ruled out (panel-11/decision.md:95).

### Q13b.3 (G12): four new rejection rows

Position: add three rows as proposed, and narrow the fourth to sizes that are both plain and unequal.
1. Panel 9 keeps static rejection "only where text decides" (panel-9/decision.md:91), and OD40 rejects only what the architect can correct.
2. A plain start below 0 or a plain size below 1 faults at every issue (instructions.md:54), so it is a typo-shaped defect; reject.
3. Two outputs with equal start and size expressions overlap at every issue (S1.3.4), decided by structural equality (expressions.md:18); reject.
4. One input tied to two outputs makes both outputs `same` as the input and so `same` as each other, the S1.3.4 fault at every issue; reject.
5. Equal start expressions with differing size expressions is not text-decidable: `2 * m` and `m + m` differ structurally and evaluate equal, so panel 9's F5 corpus row expects a fault, not a rejection (panel-9/decision.md:118-119); reject only when both sizes are plain `int` and unequal.
6. StableHLO verifies its constraints statically only on static shapes and defers the rest (https://openxla.org/stablehlo/spec, "Constraints"), the same split.
Confidence: medium-high.
Would change my mind: a ruling that S6.1 equality is by value for plain-foldable trees, which would make step 5 decidable.

### Q13b.4 (G21): an address into state

Position: allow an addressing attribute in `update`; a state variable any update assigns from an expression holding one is an addressing state variable, and may appear only in a start or a constraint, checked by `check`.
1. Base-address registers are the norm in command-stream NPUs: Vela emits `NPU_SET_IFM_BASE0` and similar register writes before each operation for Ethos-U55 (https://git.mlplatform.org/ml/ethos-u/ethos-u-vela.git/tree/ethosu/vela/register_command_stream_generator.py), and AMX `ldtilecfg` loads configuration from memory (joint-unwritten/decision.md:37).
2. The plans route descriptors through "plain integer state, attributes, and region reads" (plans/priorart/plan/P11-state-kinds.md:7; plans/casestudies/gaps.md:21), so a set-base instruction is in scope and needs a spelling.
3. D5's reason is "sizes and shapes must be address-free for B's allocator" (plans/language/plan/decisions.md:12); an update is neither a size nor a shape, so the rule's reason does not reach it.
4. Spelling the address as `Comp` lies about its kind: the backend reads every computational attribute from a shape (act-backend/act_backend/spec_checks.py:222-234), which an address never names, and a future state-aware backend (R27, rulings.md:51) would then search a shape value for an address.
5. The check is two passes over recorded trees, no flow analysis: mark each state variable assigned from an `Addr`-bearing tree, then apply the S6.1 addressing-attribute row to its reads.
6. The backend excludes state instructions today (conformance.md:95), so nothing lands in the allocator now; the oracle evaluates the state as an integer either way.
Confidence: medium.
Would change my mind: a B16 design that models state and needs addressing state kept out of updates for its solver, stated by the backend track.

### Q13b.5 (G23): what a run yields

Position: a run yields the `hbm` contents after the last issue, or the fault report; state and on-chip buffers are observable only through a generator's own interface.
1. The oracle's kernel returns `outputs` of `executable.execute`, the declared hbm outputs, and nothing else (act-oracle/taidl_to/generic/decorator.py:142, :191-195).
2. On every in-scope accelerator a kernel's architectural result is memory: NKI kernels return HBM tensors and PSUM contents are undefined at kernel start (https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/, joint-unwritten/decision.md:38).
3. A corpus case therefore asserts on hbm and on the fault report; a state assertion is a generator test, which panel 9's corpus already treats so (panel-9/decision.md:115).
4. A fault report that names the issue index from 0 is consistent with input region numbering from 0 (instructions.md:47).
Confidence: medium-high.
Would change my mind: a conformance need to assert final state across generators, which would make state part of the run result.

### Q13b.6 (G26): a `read` with no `write`

Position: legal; its effects are its faults, S1.3.11 included, and its state updates (same answer as Q13b.15).
1. Prefetch instructions read memory and write no architectural destination: PTX `cp.async.bulk.prefetch.tensor` and x86 `PREFETCHh` (https://docs.nvidia.com/cuda/parallel-thread-execution/, section cp.async.bulk.prefetch.tensor).
2. Such an instruction is modeled as a `read` whose bounds fault (S1.3.3) is the architectural effect, exactly what the hardware checks.
3. R11 rejected regions without semantics under the string surface, where the semantics text had to name every region (taidl/taidl/instruction.py:142-161); the traced surface has no reconciliation step (panel-11/decision.md:92).
4. Rejecting needs an S9 row the architect cannot correct without dropping a real instruction, against OD40 and conformance.md:8.
5. The oracle loads inputs before any store (panel-9/decision.md:105), so a module with no outputs is the trivial case.
Confidence: medium-high.
Would change my mind: a generator that cannot emit an instruction with zero outputs; the backend already excludes it as "other than one output" (conformance.md:95).

### Q13b.7 (SG5): `bool` and NumPy integers as plain integers

Position: a plain integer is any object with `__index__` other than `bool`; `bool` is a value only at `pred`.
1. An architect generates families from tables, `for n in np.arange(1, 65, 8)` or a size list read from a vendor CSV, so `np.int64` reaches `size=` routinely.
2. `operator.index` is Python's own definition of an integer-like value and returns exact `int` since 3.10 (https://docs.python.org/3/library/operator.html#operator.index); NumPy and PyTorch shapes accept it.
3. OD40 rejects only what the architect can correct; `np.int64(4)` is not a mistake, while `True` as a size or a state value almost always is (declarations.md:52 already rejects it).
4. The recorded tree still holds exact `int`, so S6.2 and the `Model` are unchanged.
Confidence: medium.
Would change my mind: a generator contract that pickles or hashes the raw given objects, where a NumPy scalar would differ from an `int`.

### Q13b.8 (SG6): a plain `bool` condition

Position: `True` folds to no constraint; `False` is rejected as an instruction no issue can satisfy; `require`, `any_of`, and `all_of` with no argument are rejected.
1. A comparison of two plain integers is a `bool` before any tracer runs (https://docs.python.org/3/reference/expressions.html#comparisons), and family loops produce such comparisons, `require(n * 64 <= 4096)`.
2. Rejecting `True` sends the author to `assert`, a second mechanism beside the one of instructions.md:62.
3. A `False` constraint faults every issue; the only hardware case for an always-illegal word is a reserved encoding, which S1.3.18 already handles by matching no instruction.
4. Empty `any_of()` would be false in logic and empty `all_of()` true, a trap no architect intends; both are typo-shaped and rejected under OD40.
Confidence: medium-high.
Would change my mind: an in-scope ISA whose documented instruction is defined to trap unconditionally, needing an always-false constraint.

### Q13b.9 (SG7): float literals

Position: a plain `float` is its binary64 value rounded once to the element type; no `str` or `Fraction` form.
1. A Python float literal is binary64 before the surface sees it (https://docs.python.org/3/tutorial/floatingpoint.html), and the front end never reads source text (grammar.md:4).
2. Hardware documents give constants as hex bit patterns, which the surface already spells exactly: `ops.bitcast_convert(ops.constant(0x3DCD, u16), bf16)`.
3. JAX and PyTorch convert `0.1` to `bf16` through binary64 too, so the oracle, the author's notebook, and the kernel agree.
4. Double rounding differs from direct rounding only when the binary64 value lands on a narrow-type tie, a case no vendor constant table produces in decimal.
Confidence: medium-high.
Would change my mind: an in-scope ISA document that specifies a constant in decimal with more than 17 significant digits.

### Q13b.10 (SG9): signatures of the creation words

Position: add the table, with `ops.iota(element_type, shape, iota_dimension=0)`; a target may be passed by keyword.
1. `ops.constant(value, element_type=s64, shape=[])` puts the type before the shape (D43, plans/language/plan/decisions.md:50); `iota` should not reverse it.
2. HLO text writes `s32[4,5] iota(), iota_dimension=0`, type before dimensions, and `jax.lax.broadcasted_iota(dtype, shape, dimension)` has the same order (https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcasted_iota.html).
3. XLA's client `Iota(builder, shape, iota_dimension)` orders differently only because its `Shape` bundles type and dimensions.
4. Keyword passing costs nothing and matches S7.5's operation attributes.
Confidence: medium-high.
Would change my mind: nothing of substance; this is a spelling choice.

### Q13b.11 (SG11): symbolic axes

Position: an attribute that names an axis is a plain `int`; a flag-chosen layout is an instruction family.
1. The one hardware case, the `wgmma` transpose flags, is listed as the failure of the mode idiom (plans/casestudies/gaps.md:24); the hardware encodes it as fixed bits, which TAIDL writes as a family (gaps.md:25).
2. A symbolic axis changes the result's dimension order per issue; StableHLO's `permutation` and `dimensions` are constant array attributes (https://openxla.org/stablehlo/spec, `transpose`), so no module holds it.
3. The backend reads computational attributes from shape text (act-backend/act_backend/spec_checks.py:222-234; isel_ir2isa_rewrites_rs_generator.py:134-137), and the oracle emits `iota_dimension` as a literal (act-oracle/taidl_to/oracle_visitor.py:366; plans/oracle/plan/bugs.md:39).
4. A size, bound, padding, or bit count may stay symbolic, since it changes a dimension's extent, not its order.
Confidence: medium-high.
Would change my mind: a second ran public site whose per-issue axis choice no family of plain-axis members covers.

### Q13b.12 (SG15): the result type of `dot`

Position: option B; `ops.dot(a, b, ..., element_type=None)` takes a result type, defaulting to the operands' type.
1. Every in-scope tensor core widens: AMX `tdpbf16ps` widens bf16 pairs with `make_fp32` and accumulates into fp32 (https://www.felixcloutier.com/x86/tdpbf16ps), `tdpbssd` gives s32 from s8, NVIDIA `mma` gives f32 or s32, Trainium and TPU give f32 from bf16.
2. XLA's `DotGeneral` has `preferred_element_type` for "higher/lower precision output types used for accumulation" (https://openxla.org/xla/operation_semantics), and StableHLO's C13 constrains only `element_type(lhs) = element_type(rhs)` (https://github.com/openxla/stablehlo/blob/main/docs/spec.md); the result type is already free in the reference.
3. Joint 1-5 rests its bound on u of the result type and says HLO `dot` "carries a result type and no accumulator type" (joint-1-5/decision.md:57); option B is that result type and adds no accumulator construct, so joint-1-5/decision.md:80 is not triggered.
4. Gap rank 3, "accumulation precision cannot be stated" (plans/casestudies/gaps.md:20), closes without a new word.
5. Option A's `dot(convert(a), convert(b))` is exact for products but a JAX kernel arrives as `f32 dot(bf16, bf16)`, so the backend needs a rewrite splitting converts out of a dot, where B matches as written.
6. The running example `convert(dot(a, b), f32)` (instructions.md:23) rounds the sum to bf16 first, which no listed unit does; B fixes the example to `ops.dot(a, b, ..., element_type=f32)`.
Confidence: medium-high.
Would change my mind: the backend's instruction selection cannot carry a mixed-type `dot` pattern, which I did not verify in act-backend.

### Q13b.13 (SG22): which IEEE function

Position: `pow` for `power`, `rSqrt` for `rsqrt`, `rootn(x, 3)` for `cbrt`, the mathematical limit for `erf` and `logistic`; the zero-result clause as proposed.
1. StableHLO `power` states "For floats: pow from IEEE-754" (https://github.com/openxla/stablehlo/blob/main/docs/spec.md), and the reference binds TAIDL (operation_meaning.md:9).
2. `pow(1, inf)` is 1 and `powr(1, inf)` is NaN in IEEE 754-2019 clause 9.2, so the choice is observable and must be named.
3. CUDA `powf` and glibc `pow` follow the C `pow` special cases, which match IEEE `pow`; no accelerator has a `powr` unit.
Confidence: high on `pow`; medium-high on the rest.
Would change my mind: a pinned XLA whose `power` lowering differs from `pow` at a special case.

### Q13b.14 (SG26): `f8e8m0fnu` rounding and the `f4e2m1fn` NaN

Position: nearest by value, a tie to the greater; keep the `f4e2m1fn` NaN to +0 rule as the oracle's explicit `select`; add the L17 tests against the pinned XLA `convert`.
1. `ml_dtypes` has no e8m0-specific converter; the generic `ConvertImpl` applies `RoundBitsToNearestEven` on the mantissa, which with zero mantissa bits is nearest by value with the carry rounding a tie up (https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/include/float8.h); the draft's rule matches it.
2. No hardware rounds to nearest: PTX `cvt` to `.ue8m0x2` admits only `.rz` and `.rp` (https://docs.rs/crate/ptx-90-parser/0.2.2/source/docs/instructions/cvt.md, mirror of the PTX ISA), so the pinned XLA is the only reference and the test is required.
3. `mxfloat.h` says "Conversion from NaNs is implementation-defined (by MX specification)" for `float4_e2m1fn` (https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/include/mxfloat.h), so +0 is TAIDL's own fixed value and the oracle must lower it explicitly, as it lowers the integer edge cases (operation_meaning.md:42).
Confidence: medium.
Would change my mind: the pinned XLA `convert` to `f8e8m0fnu` rounding a tie down, in which case the spec follows the pin.

### Q13b.15 (SG31): a `read` with no `write`

Position: legal, as Q13b.6 argues; the proposed sentence of SG31 is the text, with S1.3.11 named.
1. The evidence is Q13b.6 steps 1 to 5; one sentence in S5.2 and S7.1 each, no S9 row.
Confidence: medium-high.
Would change my mind: as Q13b.6.
