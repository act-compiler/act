# Panel 5, round 2: Fable (JAX and XLA lens)

Paths: `U` is `act-oracle/taidl_to/generic/utils.py`, `OV` is `taidl_to/oracle_visitor.py`, `T` is `taidl_to/templates/templates.txt`, `IC` is `taidl_to/instruction_codegen.py`, `D` is `taidl_to/generic/decorator.py`.
Evidence verified this round is marked "verified"; the rest is carried from round 1.

### Q5.1

All three take D: unroll every loop and set the compile-time bound before the spike.
Strongest opposing point, Sonnet: no example calls `start_loop` and the largest test trip count is 8, so the attention tests and examples cannot set the bound; the spike needs a synthetic stress kernel.
Verified: `grep start_loop` finds no hit under `act-oracle/examples/` and 63 hits under `act-oracle/tests/`.
Concession: my round-1 bound ("attention example within N times today's time") measures a kernel with no loop, so it cannot fail; Sonnet's shape (a trip count and a wall time on the CI image) is the right fail point, and Opus's per-instruction op-count cap is a useful second gate.
Rebuttal to nothing; the three positions differ only in the numbers, which the owner sets.
Final position: D, with the bound stated as a synthetic kernel of a stated trip count compiling within a stated time, plus an op-count cap per instruction.
Confidence: medium-high.
Still divides the panel: the numbers (Sonnet 1024 trips in 60 s, Opus 2x current time), and whether the fallback on a failed spike is C (Sonnet) or C with an equivalence test (Opus); both are owner calls.

### Q5.2

Strongest opposing point, Sonnet: "exact result rounded once" is not met by `f64` widening either, because an `f64` sum of K terms is not exact, so a single rounding holds only away from near-ties, and the existing test passes by data (`tests/test_mini_trn.py:91-92`), not by construction.
Concession: this is correct; a `bf16` product is exact in `f32` and an `f32` product is exact in `f64`, but the sum is exact only when all partial sums fit 53 bits, which the exponent spread of real data does not guarantee.
So C is a norm the oracle approximates with a small, unquantified failure set, which decision 16 ("defined, checkable") does not accept as the spec's statement.
Opus's split is the consistent form: the spec states the bound (B), and the oracle's widening (C) is its implementation strategy, held to the bound by `taidl.testing` and to the CPU goldens as regression only.
Rebuttal that survives: the bound's unit roundoff must be the declared accumulator type of the word, not the result type, or a tensor core accumulating `bf16` inputs in `f32` (decision 15) is non-conforming under R8's first clause.
Verified on XLA CPU (`cpu_compiler.cc`, main): `FloatNormalization(&bf16_support)` computes `bf16` ops in `f32`, `OperandUpcaster` upcasts dots, and `ChangeOpDataType(F16, F32, ...)` runs `f16` dots in `f32` unless `xla_cpu_strict_dot_conv_math` is set; so R8's first clause is false on the reference platform with or without the excess-precision flag, which answers Sonnet's "changes my mind" measurement in the negative.
Final position: B as the spec norm, written as a K-dependent bound around the exactly rounded result with u of the declared accumulator type; the oracle widens narrow dots and reduces (C) as implementation; A rejected.
Confidence: medium-high; changed from C.
Still divides the panel: whether the oracle should fall back to exact `Fraction` accumulation for narrow types if widening breaks goldens (Opus); I say no, the bound covers it.

### Q5.3

All three take B: every intermediate is a checked i64 on concrete values after unrolling, the overflow in a constraint is the same error, no warning survives.
Opus adds that the spec must fix the evaluation tree (as parsed), since i64 checking is order-sensitive; I agree and it costs one spec line.
Final position: B with the as-parsed evaluation order stated; unchanged.
Confidence: high.
Still divides the panel: only whether S6.2's "does not hold" wording is kept as a message (Sonnet: one error class with the overflow named); I side with Sonnet.

### Q5.4

Strongest opposing point, Sonnet and Opus: atomicity is cheap because one kernel is one module compiled once and executed per call, HBM leaves the oracle only as the return of `execute`, and `state` is a trace-time dict data cannot touch; so A's flag is one extra `i1` output and one host-side test, not a per-instruction commit protocol.
Verified: `compile_module` compiles the whole module once (`D:47-58`) and `wrapper` returns `outputs` from `executable.execute(operands)` after `block_until_ready` (`D:188-196`); `reset_api` makes `state` a trace-time copy (`D:38-43`).
Concession: my round-1 cost objection (doubling device-host traffic, holding every write) was wrong; the flag is read once per kernel with the outputs, and raising before returning them is R55's "writes no region and no state" at kernel granularity.
With the cost gone, the row choice follows the verification lens: integer `divide` and `remainder` at zero and at `INT_MIN / -1`, and float-to-integer `convert` out of range or NaN, are errors, because any value hides a kernel bug and no hardware agrees on one; shifts past the width and `compare` default are guarded values; `bitcast_convert` is little-endian by statement.
XLA's own `select` lowering for divide by zero stays irrelevant under this split, which also removes my round-1 dependence on an undocumented platform value.
Final position: C with errors by default for divide, remainder and convert, guarded values for shifts, little-endian for `bitcast_convert`; changed from guarded-value default.
Confidence: medium-high.
Still divides the panel: nothing of substance; the row list above matches Sonnet and Opus, and the open measurement is the flag chain's compile-time cost on the attention kernel.

### Q5.5

All three take A with the range check on the exact value: parse with `Fraction`, round once to nearest-even, emit hex bits.
Correction to my round-1 evidence: MLIR's `Parser.cpp` (main) parses a decimal float literal with `APFloat::convertFromString` directly in the target semantics, with a comment "Parse with the requested semantics directly", so the parser does not double-round today (verified).
This does not revive B: the pinned jaxlib carries an older MLIR whose behavior would need checking per pin, no document promises it (Sonnet), and hex bits are exact at zero parser dependence, which is the ground-truth property Opus names.
Final position: A with the exact range check; unchanged, with the parser claim withdrawn.
Confidence: high.
Still divides the panel: nothing.

### Q5.6

Strongest opposing point, Opus and Sonnet: decision 17 reads "the output reuses the marked input's storage", so the mark is binding text, and a backend test marks a load's HBM input in place for a `d2` output, a cross-buffer pairing that identical region text cannot express.
Verified: `owner-decisions.md:17` reads "In-place mark | The output reuses the marked input's storage"; `test_extraction.py:507` is `add_load(acc, 'load_d2', 'd2').set_inplace([True])`, a load whose input is HBM and whose output is `d2`.
Concession: A removes text that a binding decision names and that the backend corpus already uses across buffers; the mark stays as the declared pairing.
Rebuttal that survives: in the oracle a cross-buffer mark has no observable effect, since inputs load before any store (`IC:169-181`), so the oracle's only checks are the same-buffer case (windows equal, element count and type match, else raise, O10) and the overlap errors of C; XLA's own aliasing is likewise a declared pairing (`HloInputOutputAliasConfig`), not an inference.
Final position: B plus C; the mark is the declared pairing, checked on concrete windows per call, with an error for any other input/output overlap and for overlapping outputs; changed from A plus C.
Confidence: medium.
Still divides the panel: whether a marked pair in the same buffer with different windows is an error (O10, Opus and I) or a plain non-aliasing store (Sonnet's D); and the output named by a per-input flag, which `taidl/instruction.py:163-172` leaves open and panel 1 must settle.

### Q5.7

All three take B first with C goldens: NumPy references over the instruction record with exact literals, golden module-text tests per norm row, and the StableHLO interpreter only as an optional job after the owner approves the dependency.
Opus's independence argument (the interpreter evaluates the emitted text, so a lowering bug passes both) is the sharpest statement of why B leads, and I adopt it as the first reason.
Final position: B plus C; unchanged.
Confidence: medium-high.
Still divides the panel: nothing.

### Q5.8

Strongest opposing point, Sonnet: the generated oracle is a self-contained copy of `generic/` (verified, `generator.py:60-64` copies the tree into the staging oracle), so a run-time import of the front end (A) adds a pinned dependency to every generated oracle, and a vendored, tag-stamped copy (C) keeps one source without the pin; and the module move and the front-end checks carry different risk, so the item splits (D, with the audit's L sizing).
Concession: the split is right; the move of `generic/shapes.py` is mechanical and lands first, the front-end checks are new code with diagnostics and land as their own item, and E lands now at no cost.
Rebuttal that survives: the generated oracle already imports `jax` at run time (`D:6-8`), so it is not self-contained in the sense that matters, and a generated copy is a copy that the backend, a third consumer, does not share; a version check at import, as jax does with jaxlib, fails loudly where a stale copy fails silently.
Final position: D now, A as the end state with a pinned import and version check, E now; changed from A outright to D then A.
Confidence: medium; medium-low on A over C for the shape module.
Still divides the panel: A versus C for the shape module, which is a packaging value judgment (one install everywhere versus an oracle that runs with only its own package) for the owner.

### Q5.9

All three take B then C then A: fidelity items first, delete the loop algebra before splitting `utils.py`, then the runtime move and O2 with O3 only if measured; one thin function per instruction; reject E; the C++ debug handler built once per install.
Sonnet adds that `xla-debug/CMakeLists.txt` takes no ISA input yet `generator.py:72-84` rebuilds it per generation (verified: `cmake -S xla_debug_dir` with no accelerator argument), which makes F's once-per-install half a fact rather than a preference.
Final position: B, C, A in that order, F once per install, E rejected; unchanged.
Confidence: medium-high.
Still divides the panel: nothing.

### Q5.10

All three take A, D and F; the differences are scoping and the small cleanups.
Strongest opposing point, Sonnet: the flag can be set per compile in the `debug_options` of the compile options the oracle already builds (`D:55`), scoped to the oracle's modules, instead of a process-wide `XLA_FLAGS`.
Verified: jax's `get_compile_options` (`jax/_src/compiler.py`, main) reads `compile_options.executable_build_options.debug_options` and sets fields on it, so the object is writable per compile; and `cpu_compiler.cc` gates `SimplifyFPConversions` on `debug_options.xla_allow_excess_precision()` read from the module's config, so a per-module setting takes effect.
Concession: per-compile scoping is better than my round-1 backend-start flag, and it also removes the ordering constraint I put on O21.
New fact that sharpens A: on CPU both excess-precision gates sit in the oneDNN contraction block of `cpu_compiler.cc`, so on a build or path without oneDNN the `bf16` converts from `FloatNormalization` already stay and per-op rounding holds without the flag; on GPU the gate is unconditional, so A is needed for decision 15's platform regardless, and Sonnet's per-op rounding test on both platforms is the proof that the setting does anything on the pinned CPU wheel.
On B versus C: `jax_default_matmul_precision` does not reach a module compiled from text, where `HIGHEST` is per dot (`U:603-604`), and `jax_enable_x64` serves only `jax.device_put` (`D:188`); feeding inputs without `device_put` makes C moot, so B.
Final position: A scoped per compile, B, D, E measured, F; unchanged in substance, scoping changed.
Confidence: high on A, D, F; medium on E.
Still divides the panel: whether E's `triangular_solve` widening stays by default until measured (Sonnet and Opus: measure first, I agree) and nothing else.
