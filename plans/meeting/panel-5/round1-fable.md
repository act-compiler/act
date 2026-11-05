# Panel 5, round 1: Fable (JAX and XLA lens)

Assumptions held throughout: the oracle lowers to StableHLO text compiled by XLA through the pinned `jax==0.11.2` (`act-oracle/pyproject.toml:18-19`); XLA CPU is the platform the goldens run on (`act-oracle/README.md:163`); XLA's `DebugOptions` defaults apply because the oracle sets no flag (`act-oracle/taidl_to/templates/templates.txt:6` imports jax only).

### Q5.1

Position: option D, unroll every loop and set the compile-time bound now, measured on the attention tests and the examples.

1. XLA already unrolls or specializes most small static-trip-count `while` loops, and a `stablehlo.while` compiles to an HLO `while` whose body is compiled once but whose trip count XLA must prove to optimize; an unrolled straight-line module gives XLA the same information with no loop algebra on the oracle side.
2. The oracle already unrolls on every path that touches state or a varying shape (`templates.txt:101-104,137-140`) and the per-call checks enumerate iterations anyway (`utils.py:68-72`), so the symbolic algebra buys only the loops whose body is pure and shape-stable.
3. Two code paths for one construct is where defined-behavior gaps hide; decision 16 wants every violation checkable per call, which the concrete iteration already provides.
4. The spike needs a fail point or it cannot fail; the bound should be "trace plus XLA compile of the attention example within N times today's time", measured with `jax.jit(...).lower().compile()` timings, since XLA compile time is the known cost of a long straight-line module (XLA's `hlo_pass_pipeline` is linear in instruction count but some passes such as algebraic simplifier are superlinear in practice).
5. Option B keeps 400 lines whose only remaining job is a check that the unrolled path gives for free; option C keeps both paths.

Evidence: `utils.py:68-72,750-760`; `templates.txt:101-104,109-140,222`; `openxla/xla` `debug_options_flags.cc` (`xla_allow_excess_precision` and the while-loop simplification flags live beside each other, both default on); StableHLO spec, `while` section (trip count not part of the op).
Confidence: medium.
Changes my mind: a measured spike where an unrolled attention example compiles more than an order of magnitude slower than the `while` form, or a user ISA with loop counts in the thousands.

### Q5.2

Position: option C, exact result rounded once, with the oracle widening its own narrow dots and reduces; A as a fallback written only in terms of a tolerance the spec owns.

1. XLA CPU does not honor "accumulate in the result type" for `bf16`: `cpu_compiler.cc` runs `FloatNormalization` with `CpuFloatSupport bf16_support(BF16, ...)` so BF16 ops are computed in F32, `OperandUpcaster` upcasts dots, and `ChangeOpDataType(F16, F32, ...)` runs f16 dots in f32 unless `xla_cpu_strict_dot_conv_math` is set.
2. So R8's first clause ("accumulate in the result element type") is not what the reference computes, and the test asserts a single rounding from an `f64` product (`tests/test_mini_trn.py:91-92`), which is the opposite norm.
3. StableHLO leaves `dot_general` accuracy implementation-defined: `precision_config` "controls the tradeoff between speed and accuracy" and "the default parameters are implementation defined" (spec, `dot_general` section); `reduce` order is likewise not fixed. Decision 14 therefore does not pick an accumulation type for us.
4. A norm that says "exact then round once" is one sentence, explains the bit-exact CPU goldens today, and is independent of any XLA flag; the oracle meets it by emitting `convert` to `f32` or `f64` around narrow dots and `reduce`, as it already does for `dot` through `dot_type` and `.wide` (`utils.py:597-610`) and for `triangular_solve` (`utils.py:498-511`).
5. For `f32` and `f64` the oracle cannot be exact on either platform (summation order is XLA's), so the norm needs a second clause there: within a stated ulp bound as a function of K, which is where `ULPS` (`tests/conftest.py:227`) already lives.
6. GPU tensor cores (decision 15) accumulate in f32 at `HIGHEST` for bf16 and f16 inputs, so the widened norm is also what the hardware under test does; option A would make an f32-accumulating tensor core non-conforming in principle.

Evidence: `openxla/xla` `xla/service/cpu/cpu_compiler.cc` (passes named above); StableHLO spec `dot_general`; `utils.py:597-610`; `tests/test_mini_trn.py:91-92`; `tests/conftest.py:227`.
Confidence: high on step 1 and 3, medium on the chosen norm.
Changes my mind: an owner statement that hardware with narrow accumulation is a conformance target, which would force option A or B with a looser bound.

### Q5.3

Position: option B, every intermediate checked in i64 on concrete values after unrolling; S6.2's "constraint does not hold" folded in as a documented consequence for constraints only if the owner prefers it.

1. The backend already checks every intermediate in i64 (`act-backend/act_backend/expression_writer.py:72,94`); a generator contract where the oracle accepts an expression the backend rejects breaks decision 16's "checkable per call" in the one direction the user notices.
2. Python's exact integers make option A trivially cheap in the oracle, but option A is a norm no compiled generator can match without bignum arithmetic in generated C or MLIR `arith` (which wraps or traps, never widens); the spec norm should be the one every generator can implement.
3. After Q5.1 unrolls, every intermediate is concrete, so the check is one `assert -2**63 <= v < 2**63` per node in `evaluate` (`utils.py:60-65`) and the loop-expression warning paths (`utils.py:287,659,706`) collapse into it.
4. State updates and constraints never warn today (`templates.txt:353-363`), which is a hole under option D.

Evidence: `expression_writer.py:72,94`; `utils.py:60-65,287,659,706`; `templates.txt:353-363`; MLIR `arith` dialect doc (`arith.addi` has `overflow` flags and no widening).
Confidence: medium.
Changes my mind: a spec decision that the ISA description language is defined over mathematical integers, with the backend changed to match.

### Q5.4

Position: option C, a per-case split in the norms table, with the default row being a defined value written by the emitter (B) and errors only where the hardware under test faults.

1. StableHLO leaves integer `divide` by zero "an implementation-defined value of the element type" (spec, `divide`), and `convert` overflow "TBD" (spec, `convert`); XLA CPU lowers integer divide by zero to a `select` returning -1 or `INT_MIN` for `INT_MIN / -1`, and GPU does the same in its emitter, so the platform value is stable but undocumented.
2. Decision 16 says every case is an error or one simple norm; a `select` guard in the emitter costs one op per case and makes the module's value spec-defined on both platforms with no run-after flag.
3. Option A's commit-after-check (O23) needs every output and state write held until a reduce_or flag is read back, doubling device-host traffic per call and complicating the straight-line module; it is only justified where an accelerator really traps, and no current ISA in `examples/` does.
4. Option D hands the norm to XLA, whose `select` lowering is not part of any spec and could change under a jax bump (Q5.10 F).

Evidence: StableHLO spec `divide` and `convert` sections; `openxla/xla` `xla/service/elemental_ir_emitter.cc` (integer divide guards); `utils.py:651-806`; O22, O23.
Confidence: medium.
Changes my mind: an ISA in scope whose hardware raises on divide by zero, which would move that row to A.

### Q5.5

Position: option A, parse with `Fraction`, round once to the target type, emit hex bits; range check on the exact value.

1. `float(text)` then `repr(float(value))` rounds twice for `bf16` and `f16` (`oracle_visitor.py:383`, `utils.py:408-409`); double rounding is a known source of one-ulp errors and the spec cannot describe it as a norm.
2. StableHLO constants accept a hexadecimal literal that denotes the bits exactly, which is the same path the oracle already uses for non-finite values (`utils.py:411-420`); emitting hex for every float removes dependence on the MLIR parser's decimal rounding.
3. Option B relies on the MLIR parser rounding decimal text correctly for narrow types; MLIR parses decimal floats through `APFloat` from a double, so it double-rounds too for `bf16`.
4. The range check at `utils.py:404-406` compares the double-rounded value; with `Fraction` it compares the exact value with `largest + half`, which is the correct tie rule.

Evidence: `oracle_visitor.py:378-390`; `utils.py:400-420`; StableHLO spec, constants (hex literal form); LLVM `APFloat::convertFromString` documentation.
Confidence: high.
Changes my mind: evidence that MLIR's parser rounds decimal text directly to the narrow type.

### Q5.6

Position: option C on top of A: alias exactly when region texts are identical, plus a call-time error for any other overlap between an input and an output or between two outputs.

1. Decision 17 makes the mark derivable, so `set_inplace` is redundant text that the oracle does not read (D10).
2. An input/output overlap that is not identical text is today a silent order-dependent result: inputs load before outputs store (`instruction_codegen.py:169-181`), and two overlapping outputs let the later write win (`instruction_codegen.py:667-681`); both are undefined behavior under decision 16.
3. In XLA the same question is `may-alias` on the entry computation's parameters and outputs; XLA refuses or copies on partial overlap rather than guessing, which is the precedent for an error.
4. Option D (normalization to concrete windows) is cheap after Q5.1 but makes aliasing depend on the call's values, so a spec reader cannot see it; keep it as a check, not as the aliasing rule.

Evidence: `instruction_codegen.py:169-181,667-681`; `taidl/instruction.py:163-172`; XLA `HloInputOutputAliasConfig` (must-alias and may-alias).
Confidence: medium.
Changes my mind: an ISA whose instructions legitimately write overlapping output windows with a defined order.

### Q5.7

Position: option B, NumPy references over the instruction record with `Fraction` literals; the StableHLO interpreter is an optional second check, not the reference.

1. The StableHLO interpreter follows the spec at a given stablehlo commit, while the jax pin carries its own stablehlo version; a mismatch tests the interpreter version, not the oracle.
2. The interpreter leaves open exactly the rows Q5.2 and Q5.4 must settle (implementation-defined divide, reduce order), so it cannot serve as the norm's reference; a reference cannot settle a norm.
3. The interpreter needs CHLO legalization and the oracle emits some CHLO words, adding a pipeline step the tests do not exercise today.
4. NumPy already is the reference for elementwise and reduce (`tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`); one module over the instruction record removes the per-test copies and lets `Fraction` literals express the norm exactly.
5. Option A also adds a dependency not on PyPI, which the shared engineering rule says to ask about first.

Evidence: `tests/test_elementwise.py:7`; `tests/test_reduce.py:82-91`; `openxla/stablehlo` `stablehlo/reference` (interpreter) and `stablehlo/tests/interpret`; R6, R29.
Confidence: medium.
Changes my mind: an interpreter wheel on PyPI pinned to the stablehlo version jaxlib ships.

### Q5.8

Position: option A with E: all static checks move to the front end, the oracle imports the front end's module at run time pinned to the same tag, and spec errors are `ValueError`.

1. The checks in question are properties of the ISA text, not of a call (`semantic_generator.py:41-56`, `oracle_visitor.py:251-289,373-389`), so decision 16's "per call" clause does not keep them in the oracle.
2. Two copies of the S8.3 table (D7) and of the shape rules (`generic/shapes.py`) drift; the backend is a third consumer.
3. Importing at run time is what jax itself does with jaxlib: a version check at import pins the pair, which is the model for the oracle and the front end.
4. Option C (vendored generated copy) reintroduces the drift it set out to remove.

Evidence: `semantic_generator.py:41-56`; `oracle_visitor.py:251-289,373-389`; `taidl/docs/conformance.md:59`; jax `jax/_src/lib/__init__.py` version check pattern.
Confidence: medium.
Changes my mind: a measured generation-time cost or a packaging constraint that forbids the oracle depending on the front end.

### Q5.9

Position: option B first (fidelity, O8 and O21 and the norms), then C (delete the loop algebra), then A (runtime to `runtime/`, O2, O3 with O4 folded in); keep one thin function per instruction; no `jax.lax` lowering (reject E).

1. A byte-identical `api.py` gate needs a stable baseline; landing fidelity fixes after the split means re-baselining twice.
2. Deleting the loop algebra before splitting `utils.py` means no deleted code is moved.
3. Lowering through `jax.lax` would reintroduce jax's own dtype promotion and tracing rules between the spec and XLA; StableHLO text is the narrowest contract and what jax itself exports.
4. Per-generation C++ debug-handler build (F) belongs once per install; a build per generation couples generation time to a compiler.

Evidence: `templates.txt:26-293`; `generator.py:72-84`; `api_generator.py:61-86`; jax export docs (StableHLO as the serialization boundary).
Confidence: medium.
Changes my mind: a measured compile time showing O3 must land before unrolling can be adopted.

### Q5.10

Position: options A, D, E, and F together; B as drafted; not C.

1. `xla_allow_excess_precision` defaults true (`debug_options_flags.cc`, "Allow xla to increase the output precision of an instruction"), and on CPU it enables `SimplifyFPConversions` to remove `f32 -> bf16 -> f32` casts inserted by bf16 normalization (`cpu_compiler.cc`), so the softmax chain at `bf16` (`mini_trn.py:101-104`) can skip the per-op rounding the spec implies.
2. Setting it false is a backend-start flag (`XLA_FLAGS` or `jax.config` before first use), which is why O21 must run before the backend initializes.
3. The CPU must be the normative platform (D) because the goldens are bit-exact there (`README.md:163`) and GPU is held to bounds; every GPU deviation goes in the coverage table.
4. The `f16` solve is already widened by XLA CPU through `ChangeOpDataType(F16, F32, ...)` for dots and convolutions, but not for `triangular_solve`; measure before deciding (E).
5. A jax bump changes the stablehlo version and XLA passes; regenerate and diff every golden and remeasure the ulp table (F).
6. Process-wide x64 serves only `jax.device_put`, and a module compiled from text does not depend on it; drop it (B) rather than scope it (C).

Evidence: `openxla/xla` `xla/debug_options_flags.cc`; `xla/service/cpu/cpu_compiler.cc`; `examples/mini_trn/mini_trn.py:98-106`; `utils.py:498-511,603-604`; `pyproject.toml:18-19`; `README.md:163`.
Confidence: high on steps 1 and 2, medium on the rest.
Changes my mind: a measurement showing excess precision false changes no golden, which would make A a documentation line instead of a flag.
