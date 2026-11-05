# Panel 5 round 2: Sonnet (oracle maintainer)

Paths: `U` is `act-oracle/taidl_to/generic/utils.py`, `OV` is `taidl_to/oracle_visitor.py`, `IC` is `taidl_to/instruction_codegen.py`, `D` is `taidl_to/generic/decorator.py`.
Checks run this round, read-only: `U:574-581`, `U:595-610`, `U:498-511`, `IC:93-113`, `D:180-196`, `act-backend/tests/test_extraction.py:500-512`, `grep start_loop` over `examples/` and `tests/`.
Limit: jax is not installed in this sandbox, so no flag or compile-time claim below was executed.

### Q5.1

Strongest opposing argument (Opus, Fable): the bound must be set before the spike and expressed relative to today's compile time of the attention tests and examples, because a bound fitted afterwards cannot fail.
Rebuttal and concession:

- My round-1 numbers (1024 trips in 60 s, 64 trips in 5 s) were invented, not measured.
- I withdraw them as evidence; they stay only as an example of the shape a bound takes (a count and a time).
- Opus's relative form is better in kind, but it has a hole: no file under `examples/` calls `start_loop`, so a bound over "every example" says nothing about loops.
- Loops appear only in tests (`tests/test_constraints.py:70,123,137-138` use trips of 2 to 4), so the relative bound would pass vacuously.
- The bound therefore needs two parts: a relative limit on the existing loop tests, and one synthetic stress kernel with a fixed trip count and instruction count.
- Fable's point that XLA unrolls or specializes small whiles itself does not decide anything here, because the cost in question is oracle-side trace plus XLA compile of a long straight-line module.
- All three of us agree on the fallback order: if the spike fails, C (unroll only where needed) with a test that both paths agree (Opus), not B.

Final position: D. Unroll every loop; the bound is a relative limit on existing loop tests plus a synthetic stress kernel, both set by the owner before the spike.
Confidence: medium.
Still divides the panel: nothing on the option; the numbers are an owner value judgment, and the stress kernel's size has no evidence behind it yet.

### Q5.2

Strongest opposing argument (Opus and Fable, option C): "accumulate in the result type" is violated by the oracle's own reference platform, so the norm should be exact then rounded once, met by the oracle widening narrow dots and reduces itself.
Fable adds that the oracle already widens dots through `dot_type` and `.wide` (`utils.py:597-610`), so C is a small step.
Rebuttal:

- That last claim is false as read: `DOT_WIDTHS` holds only `i2` and `ui2` (`U:575`), so `dot_type` widens 2-bit integers and nothing else.
- A `bf16` dot today is emitted at `bf16` with `HIGHEST` (`U:603-606`) and gets its f32 accumulation from XLA CPU's upcast pass, which is XLA's choice and not the oracle's.
- "Exact then round once" is also not what an `f64` sum delivers: bf16 products are exact in f32, but a sum of K of them in `f64` is exact only while the exponent spread stays within 53 bits.
- So the oracle can promise "accumulated wide, rounded once, within a bound of the exact value", not "exact"; the word exact would be a false claim in the spec.
- A norm that says exact cannot be met by an f32-accumulating tensor core either (decision 15), so conformance for hardware is the tolerance in any case.
Concession:

- Fable's point that the oracle's value should not depend on a CPU pass that can change under a jax bump is right.
- The oracle should name its accumulator in the module: convert narrow float dot and reduce operands to `f32`, accumulate, convert once to the result type.
- That is an implementation rule of the oracle and a test target, while the spec norm stays B.
- Widening to `f64` or `Fraction` for narrow types is not needed; `f32` matches the declared accumulator of tensor cores.

Final position: B as the spec norm (tolerance as a function of K and the accumulator type, owned by the spec, checked in `taidl.testing`); the oracle accumulates narrow floats at `f32` explicitly; R8 clause one is dropped.
Changed: partly; I add explicit `f32` widening in the oracle, adopted from the C camp, and keep B for the norm.
Confidence: medium.
Still divides the panel: whether the spec says "exact then round once" (Opus, Fable) or "within a bound" (me); the difference matters for float `f32` and `f64`, where no one claims exactness.

### Q5.3

All three agree on B (every intermediate checked in i64 on concrete values after unrolling).
Residual: Opus asks the spec to fix the evaluation tree (as parsed, left to right) because i64 checking is order-sensitive; I adopt that as a one-line spec rule, and Fable's offer to fold S6.2 in as a consequence for constraints is the same outcome as a raised `broken_constraint` with the overflow named.
Final position: B.
Changed: no.
Confidence: medium-high.
Still divides the panel: nothing; only the wording of S6.2.

### Q5.4

Strongest opposing argument (Fable): the default row should be a defined value written by a `select` guard, because XLA CPU and GPU already return a stable total value for integer divide by zero, O23's commit-after-check adds device-host traffic, and no ISA in scope hardware-traps.
Rebuttal:

- The stable value is XLA's lowering (`-1`, `INT_MIN`), not a property of the accelerators under test, so adopting it as the norm models XLA and not the ISA.
- A defined value for divide by zero hides the kernel bug the oracle exists to find; the failing kernel runs on and gives a plausible tensor.
- The traffic claim is unsupported: the flag is one `i1` scalar output, and HBM already leaves only as the return of `execute` (`D:191-196`), so commit-after-check is a host-side test before returning, not a per-instruction protocol.
- Atomicity is free here because the module is functional and the wrapper returns outputs only when the flag is clear.
Concession:

- Fable's guard rule is right for rows where a total result is the common hardware answer: shifts past the width (zero, or sign fill for arithmetic right) and `bitcast_convert` (little-endian, a spec fact).
- One real risk to my side: padded or unwritten lanes in a tile hold zeros, and an error on a zero divisor in such a lane is a false alarm; the norm table must say whether a divide is over the declared region only, which the module already is.

Final position: C. One norm table, row by row: errors for integer `divide` and `remainder` at zero and `INT_MIN / -1` and for float-to-integer `convert` out of range; guards for shifts; `bitcast_convert` is little-endian.
Changed: no.
Confidence: medium.
Still divides the panel: the divide-by-zero row (error: me and Opus; guarded value: Fable); float `convert` overflow is also open between error and saturate.

### Q5.5

All three agree on A with the range check on the exact value (D, exact form).
Fable adds that MLIR parses decimal floats through `APFloat` from a double and so double-rounds for `bf16`; I could not verify that this round and keep it as a reason to prefer hex, not as the decisive one.
Final position: A with exact range check.
Changed: no.
Confidence: high.
Still divides the panel: nothing.

### Q5.6

Strongest opposing argument (Opus): keep `set_inplace` as the declared pairing, because decision 17 speaks of "the marked input" and a backend test marks a cross-buffer pair that identical region text cannot express.
Fable's counter is that the mark is derivable and redundant (D10).
Check run: `act-backend/tests/test_extraction.py:500-512` marks `load_d2` (a load into `d2`, so the input is HBM `d0` and the output is `d2`) as in place.
Rebuttal and concession:

- The cross-buffer case is real, so "identical region text" (R5) cannot be the whole definition; I lose nothing by saying so, since round 1 already kept B as the fallback.
- For the oracle the question is narrower: `generate_outputs` slices into the named data model (`IC:93-113`), so a cross-buffer mark changes no value the oracle can observe.
- The oracle observes aliasing only inside one buffer, where inputs load before outputs store, so an equal window changes nothing and a partial overlap is the case needing a rule.
- Therefore the oracle's work is the same under A and B: compare concrete windows after unrolling, accept equal, raise on partial input/output overlap and on overlapping outputs.
- Who owns the pairing (mark versus derivation) is a language and backend question for panels 1 and 2, and the oracle takes whichever pairing it is given.

Final position: C with D for the oracle (concrete-window comparison, error on any other overlap); the mark-versus-derivation choice is not the oracle's to decide.
Changed: yes, in scope only; I no longer argue that the mark can go or must stay.
Confidence: medium on the oracle rule; low on the pairing.
Still divides the panel: whether the mark survives (Opus) or is derived (Fable); no oracle evidence settles it.

### Q5.7

All three agree on B plus C (NumPy references over the instruction record with `Fraction` literals, golden module-text tests per norm row) and no interpreter dependency now.
One refinement from Opus I adopt: the interpreter checks XLA against the spec, not the lowering against the instruction, so it is a later optional job.
A condition from Fable I adopt: revisit if an interpreter wheel pinned to the jax pin's StableHLO version exists.
Final position: B plus C.
Changed: no.
Confidence: medium-high.
Still divides the panel: nothing.

### Q5.8

Strongest opposing argument (Fable, Opus): two copies of the S8.3 table and of the shape rules drift, a vendored copy reintroduces the drift, and a run-time import pinned to the same tag fails loudly on mismatch, as jax does with jaxlib.
Rebuttal and concession:

- Drift is the real cost, and my vendored-copy idea (C) has a mitigation only if a test asserts the vendored file equals the source, which Opus names.
- The decisive fact is packaging: whether a generated oracle must run with only its own directory installed.
- Without that fact the vendored copy and the import are equivalent in drift once the equality test exists, and the import is simpler.
- The generator today copies a self-contained directory, and nobody has said whether that property is required.
- I weaken my preference: the end state is a single source (all three agree), and the oracle package declaring a pinned dependency on the front end is acceptable if the owner confirms a generated oracle may require taidl at run time.
- The split (D) is agreed by Opus and me; Fable takes the whole item as A and does not object to sequencing.

Final position: D, then A as the end state if a run-time dependency is acceptable, otherwise C with an equality test; E now.
Changed: partly; I move from preferring C to treating A and C as equal pending one owner answer.
Confidence: medium.
Still divides the panel: whether a generated oracle may import the front end at run time; this is an owner question with the options A and C.

### Q5.9

All three agree on the order fidelity (O8, O21, norms), then delete the loop algebra, then runtime split and O2/O3; one thin function per instruction; no `jax.lax` lowering; debug handler built once per install.
One addition: with Q5.2 above, the explicit `f32` widening of narrow dots changes module text, so it belongs to the first group, before the byte-identical baseline is taken.
Final position: B, then C, then A; F once per install.
Changed: no.
Confidence: medium; high on F.
Still divides the panel: nothing.

### Q5.10

Strongest opposing argument (Fable, Opus): drop the process-wide x64 setting (B) and do not scope it (C), because a module compiled from text does not depend on x64.
Fable adds that excess precision must be set before the backend starts, through `XLA_FLAGS` or `jax.config`, and not per compile.
Rebuttal and concession:

- On x64 I accept dropping it, with one condition: `D:180-196` passes caller arrays through `jax.device_put` and accepts any declared dtype, and without x64 jax downcasts 64-bit arrays silently.
- So the drop needs a test with an `f64` and an `i64` argument, or a bytes-based feed, before it lands; a silent downcast would be a new violation of decision 16.
- On the excess-precision flag, Fable's start-of-backend route is the safe default, and my compile-options route (`debug_options` on the options built at `D:54`) is an alternative I could not test here.
- The test that decides it is the one every position already requires: a `bf16` add-then-multiply chain against NumPy on both platforms, run with each route.
- A scoped route is preferable only if it passes that test, since a process-wide flag changes every other jax user in the process.
- On the dot, Q5.2 now adds explicit `f32` widening in the module, which removes the oracle's dependence on XLA's `bf16` normalization pass for dots; excess precision still governs elementwise `bf16` chains.

Final position: A (flag off, proved by the per-op test, route chosen by that test), B with the 64-bit feed test, D, E (measure first), F.
Changed: partly; C is dropped in favor of B.
Confidence: high on A, D, F; medium on B and the flag route.
Still divides the panel: nothing on the options; only the route for setting the flag, settled by a one-hour test.
