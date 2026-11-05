# Panel 5 round 1: Opus (numerics and verification)

Lens: the oracle is the ground truth for the backend, so every behavior it has must be stated, reproducible, and checkable against something independent of it.
Shared assumptions:
- Loop trip counts are compile-time attributes of an instruction, so after unrolling every addressing value is a plain integer.
- Decision 16 binds every run-time case to an error or one defined value; "implementation-defined" is not an allowed answer for the oracle.
- The StableHLO spec leaves reduce order, dot precision, and integer divide at zero implementation-defined (https://openxla.org/stablehlo/spec, `reduce`, `dot_general`, `divide`).

### Q5.1

Position: D; unroll every loop and set the compile-time bound now, before the spike runs.
1. A per-call check over a symbolic loop variable must enumerate iterations anyway (`U:68-72`), so the symbolic algebra buys no checking power, only a second semantics to verify.
2. Unrolled values are concrete integers, so the i64 rule of Q5.3 and the overlap rule of Q5.6 become plain comparisons with no symbolic proof.
3. Unrolling is already the path for state updates and varying shapes (`T:101-104`), so A and D make one path the only path; C keeps two paths whose equivalence nobody tests.
4. A spike with no fail point cannot fail; the bound must exist before measurement or it will be fitted to the result (`overhaul.md:32`, `decisions.md:21`).
5. Proposed bound: the attention tests and every example compile in at most 2x their current time, and no single instruction emits more than a stated op count; the owner may change the numbers, not their existence.
Evidence: `U:68-72`; `T:101-104,109-132`; `oracle/plan/decisions.md:21,58`; `overhaul.md:32-33`; XLA unrolls loops itself as an optimization with a trip-count limit (https://github.com/openxla/xla/blob/main/xla/service/while_loop_unroller.h).
Confidence: medium-high.
Would change my mind: a TAIDL example whose trip count makes unrolled compile time exceed the bound by an order of magnitude; then C, with a test that the `while` and unrolled paths agree.

### Q5.2

Position: C for the oracle plus B for conformance; the oracle computes the exact result rounded once, and the spec states a K-dependent tolerance that hardware must meet.
1. A says "accumulate in the result element type"; for a `bf16` result that is `bf16` accumulation, which XLA CPU does not do and the oracle's own test rejects (`tests/test_mini_trn.py:91-92` asserts one rounding of an `f64` product).
2. So A is a norm the oracle already violates; a ground truth that breaks its own written norm cannot arbitrate a backend.
3. Under decision 15 a tensor-core `mma` accumulates in its declared accumulator type, typically `f32`; the TAIDL result type is that accumulator, so "exact then round once" models it and gives a reproducible reference.
4. Products of two `bf16` or `f16` values are exact in `f32`, and products of two `f32` values are exact in `f64`; summing them in `f64` gives a result whose error is far below half an ulp of the narrow type except at near-ties, so widening is a cheap, near-exact reference.
5. Where widening is not exact (an `f64` dot, or a near-tie), the oracle cannot claim bit-exactness, so the spec needs a tolerance anyway: B states it as a function of K and the type, in the style of the classical bound gamma_K = K u / (1 - K u).
6. D (fixed order) forbids the tree reductions every accelerator uses, and the StableHLO spec states the order is implementation-defined, so a fixed order would make conforming hardware non-conforming.
7. E makes XLA's choices the norm; those change between releases (Q5.10), so the norm would drift with a jax pin.
Evidence: `tests/test_mini_trn.py:91-92`; `U:603-604` (`HIGHEST` per dot); `README.md:163` (CPU bit-exact, GPU within 4 ulps); StableHLO spec `reduce`: "The order of reductions is implementation-defined" and `dot_general`: "the semantics of these enum values is underspecified" (https://openxla.org/stablehlo/spec); critic note on the norm table (`reviews/oracle/critic.md:44`).
Confidence: medium.
Would change my mind: a measured case where `f64` widening of a `bf16` dot differs from the exactly rounded result often enough to break goldens; then the oracle reduces with exact `Fraction` or integer accumulation for narrow types.

### Q5.3

Position: B; every intermediate is i64, and leaving i64 at any step is an error, checked on concrete values after unrolling.
1. The backend already rejects any intermediate past i64 (`act-backend/act_backend/expression_writer.py:72,94`, Rust `checked_add` and friends).
2. Under A, the oracle accepts programs the backend rejects; a differential reference that disagrees with the implementation about which programs are errors produces false alarms on exactly the edge cases verification exists for.
3. Hardware address units are fixed width, so i64 intermediates are the model closer to the target.
4. C makes an overflowing constraint silently false; that is a silent fallback which masks a wrong program, against the shared rule to fail loudly and against decision 16's "defined error".
5. D keeps a warning, which decision 16 does not allow for a run-time violation.
6. B needs the spec to fix the evaluation tree of an expression (as parsed, left to right), since i64 checking is order-sensitive; that is a one-line spec rule.
7. After Q5.1's unrolling, the check is a Python loop over integers with a bound test per operation, with no symbolic reasoning.
Evidence: `expression_writer.py:72,94`; `T:353-363` (constraints never warn); `U:287`; Rust `i64::checked_add` returns None on overflow (https://doc.rust-lang.org/std/primitive.i64.html#method.checked_add).
Confidence: medium.
Would change my mind: an owner ruling that the backend moves to exact intermediates too; then A, provided both sides change together.

### Q5.4

Position: C; a per-row choice in one norm table, with errors detected in-module by flag (A's mechanism) and defined values by guard (B's mechanism).
1. Integer `divide` and `remainder` at zero and at `INT_MIN / -1` are errors: no value is right, and a value would hide a bug; StableHLO leaves the result implementation-defined.
2. Float-to-integer `convert` out of range is "TBD" in StableHLO; I propose an error for NaN and out-of-range, since saturation silently changes results.
3. Shifts past the width get a defined value (zero for left and logical right, sign fill for arithmetic right), a common hardware result that is cheap to guard with `select`.
4. `bitcast_convert` byte order is a defined value: little-endian, stated once.
5. `compare` default is a front-end rule, not a run-time case.
6. Atomicity is cheap in this oracle: the module is functional, so outputs are new values, and the oracle writes them back only when every flag is clear; the cost is one `reduce_or` per guarded op and a host check.
7. D and E leave rows open, which decision 16 forbids.
Evidence: StableHLO `divide`: "the corresponding element of result is an implementation-defined value"; `convert`: "the behavior is TBD" (https://openxla.org/stablehlo/spec); `U:651-806` (all checks before tracing); `oracle/plan/fidelity.md:75-84` (O23); `decisions.md:50`.
Confidence: medium.
Would change my mind: a measured case where the write-back is not separable from the run (for example a donated buffer); then commit-after-check needs a copy, and its cost must be sized.

### Q5.5

Position: A, with the range check on the exact value (D, exact form).
1. `float(text)` then a narrow type rounds twice (`OV:383`, `U:408-409`), and double rounding gives a wrong result at ties; an integer above 2^53 at a float type has the same fault.
2. Parsing with `Fraction` and rounding once to nearest-even gives the IEEE result by construction.
3. Hex bits remove any dependence on the pinned MLIR parser's decimal reading; the oracle already emits non-finite values by bits (`U:411-420`), so this extends an existing path.
4. B depends on parser behavior that can change with a jax bump (Q5.10); a ground truth should not.
5. C documents a defect instead of fixing it, and contradicts `docs/operations.md:103` ("rounds to the nearest value").
6. The range check must compare the exact value with largest-plus-half-ulp, with the tie rule for types without infinity, which `U:400-406` already encodes; only its input must become exact.
Evidence: `OV:383`; `U:400-420`; `docs/operations.md:103`; MLIR accepts hexadecimal float literals as exact bit patterns (https://mlir.llvm.org/docs/Dialects/Builtin/, FloatAttr).
Confidence: high.
Would change my mind: nothing within the numerics lens; only a cost argument that hex output harms readability enough to matter, which a comment with the decimal fixes.

### Q5.6

Position: B plus C; keep the mark as the declared pairing, check its windows at the call, and raise on any other overlap and on overlapping outputs.
1. Decision 17 speaks of "the marked input", so the mark is part of the binding text; A removes it.
2. A backend test marks a `d0` input in place for a `d2` output (`act-backend/tests/test_extraction.py:507`), so pairing by identical region text cannot express a case already in use.
3. Inputs load before stores (`IC:169-181`), so identical-window aliasing does not change values; only partial overlap and overlapping outputs change results.
4. Overlapping outputs today resolve as "the later write wins" (`IC:667-681`), an emission-order accident, not a stated norm.
5. After unrolling, windows are concrete, so overlap is an interval test per call, and D's normalization is free.
6. `set_inplace` has one flag per input (`taidl/instruction.py:163-172`); the spec must name which output a mark pairs with, or the oracle cannot check it.
Evidence: `IC:169-181,667-681`; `taidl/instruction.py:163-172`; `test_extraction.py:507`; XLA's input/output aliasing is an explicit declared pairing, not inferred (https://openxla.org/xla/aliasing).
Confidence: medium.
Would change my mind: a panel 1 or 2 ruling that reads decision 17 as satisfied by derived aliasing and removes the cross-buffer case from the backend test.

### Q5.7

Position: B plus C; NumPy references over the instruction record with exact literals, golden module-text tests per norm row, and no new interpreter dependency now.
1. A reference earns its place only if it is independent of what it checks.
2. The StableHLO interpreter evaluates the emitted module text, so a lowering bug that writes the wrong module evaluates "correctly" in both; it checks XLA against the interpreter, not TAIDL against the module.
3. A NumPy reference built from the instruction record is independent of the lowering, which is where the oracle's bugs live (D1-D17).
4. The interpreter's own float check uses a single tolerance of 0.0001 for every type, which is looser than one `bf16` ulp near 1 only by luck and does not scale with K; it cannot serve as the accuracy contract.
5. It is a new dependency, which needs owner approval, and its StableHLO version may differ from the jax pin.
6. C covers what B cannot: module-text goldens pin the lowering of each norm row (Q5.4) exactly.
Evidence: `tests/test_elementwise.py:7`; `tests/test_reduce.py:82-91`; `oracle/plan/decisions.md:44,46`; StableHLO testing guideline G6: "a one-size-fits-all tolerance value of 0.0001" (https://github.com/openxla/stablehlo/blob/main/docs/reference.md).
Confidence: medium-high.
Would change my mind: evidence that the interpreter runs from a pip-installable wheel at the jax pin's StableHLO version; then it is a cheap third opinion for XLA-level bugs, still not the first reference.

### Q5.8

Position: D, with A as the end state and E adopted independently.
1. A copied table drifts silently (`TYPES` copies S8.3 row by row, D7); a single source is the only way two generators agree by construction.
2. Importing the front end pinned to the same tag (A) keeps one source and fails loudly on a version mismatch; vendoring (C) reintroduces the copy.
3. B leaves region and literal checks in two places, so the front end and the oracle can disagree on validity.
4. The audit sizes the merged item L or larger, against R62's M; splitting it lets the module move land and be measured before the checks move.
5. E is independent of location and improves verification: a spec error and a coverage gap are different failures and must be distinguishable in tests.
6. Decision 16's per-call checks stay in the oracle; only static checks move.
Evidence: `semantic_generator.py:41-56`; `OV:251-289,373-389`; `taidl/docs/conformance.md:59`; LLVM TableGen exists to keep one description that generates every consumer (https://llvm.org/docs/TableGen/).
Confidence: medium.
Would change my mind: a front-end release cadence that cannot match the oracle's tag; then C with a test that the vendored copy equals the source.

### Q5.9

Position: B, then C, then A; fidelity first, delete before moving, then restructure; reject E; build the debug handler once per install (F).
1. A byte-identical gate is only useful against a baseline that will not change for fidelity reasons; O8 and O21 change bytes, so they land first (`overhaul.md:31`).
2. Deleting the loop algebra before splitting `utils.py` avoids moving about 400 lines that then die.
3. A's runtime move then makes the 260 lines of `API_FILE` visible to ruff and tests, which is a verification gain.
4. E loses the emitted text that the module goldens of Q5.7 pin, and gives jax control of lowering choices the norm table must own.
5. D keeps per-word templates that the critic and architect both identify as untested text.
6. F: a per-generation C++ build adds a moving part per run with no fidelity value.
Evidence: `T:26-293`; `overhaul.md:31-39`; `decisions.md:39`; `reviews/oracle/critic.md:37`; `generator.py:72-84`.
Confidence: medium.
Would change my mind: a fidelity item that needs the runtime split to land (for example O23 needing O3); then that step moves earlier, alone.

### Q5.10

Position: A, B, D, E, and F; excess precision off, CPU normative, GPU held to stated bounds, and a jax bump treated as a fidelity event.
1. `xla_allow_excess_precision` defaults to true and lets XLA "increase the output precision of an instruction", so a `bf16` chain (`examples/mini_trn/mini_trn.py:101-104`) may skip intermediate rounding.
2. A reference whose rounding depends on an optimizer flag is not ground truth; A sets it false and tests per-op rounding on both platforms.
3. B: the matmul-precision global does not affect a module compiled from text, and `HIGHEST` stays per dot (`U:603-604`); removing `jax.device_put` makes the x64 global unnecessary, so C becomes moot.
4. D: the CPU goldens are bit-exact and the GPU is already held to 4 ulps (`README.md:163`); stating that split makes it a contract, and every GPU deviation gets a row.
5. E: the `f16` solve is widened on every platform (`U:498-511`) on an untested claim; measure first, then keep or drop.
6. F: XLA changes its transcendental approximations between releases, so each bump regenerates and diffs every golden and remeasures the ulp table.
Evidence: `decorator.py:22-23,188`; `U:498-511,603-604`; `pyproject.toml:18-19`; XLA `debug_options_flags.cc`: "Allow xla to increase the output precision of an instruction", default true (https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc); `reviews/oracle/critic.md:43,47-48`.
Confidence: high for A, D, F; medium for B and E.
Would change my mind: a measurement showing excess precision has no effect on CPU for the pinned XLA; A still stays, since the GPU pipeline uses the flag.
