# Panel 5 round 1: Sonnet (oracle maintainer)

Paths: `U` is `act-oracle/taidl_to/generic/utils.py`, `OV` is `taidl_to/oracle_visitor.py`, `T` is `taidl_to/templates/templates.txt`, `IC` is `taidl_to/instruction_codegen.py`, `D` is `taidl_to/generic/decorator.py`.
Assumptions across all answers:

- The oracle's job is to be the reference the backend is compared against, so a divergence from the backend is a defect.
- Decision 16 holds as written: every violation is an error or one simple norm.
- The kernel body never reads input data, so every trace-time value is a concrete Python value once loops unroll.

Facts that shape several answers:

- One kernel is one StableHLO module, compiled once in `fsim-compile` and executed per call in `fsim` (`D:147-200`).
- `state` is a trace-time Python dict of addressing values, reset per kernel (`D:38-43`, `T:14-16`, `T:353-363`).
- Only HBM contents are data-dependent, and they leave the oracle only as the return value of `fsim` (`D:191-196`).

### Q5.1

Position: D. Unroll every loop (A), and fix a numeric fail point before the spike, because no example or test exercises a loop of real size.

1. Unrolling is already the path for any state update or varying shape (`T:101-104`, `T:137-140`), and every check already enumerates the iterations a value reads (`U:68-72`).
2. So unrolling adds Python trace cost only for loops that today stay symbolic, and it deletes `Unroll`, the `deepcopy` retrace loop (`U:~750-761`, the `trace_outermost` shape), `while` emission, and `held` and `fixed`.
3. The loop algebra is the origin of D11, D17, D19, D20; with concrete ints, Q5.3 becomes a ten-line checked evaluator.
4. The "agreed bound" has no baseline: no file under `examples/` calls `start_loop`, and the largest trip count in the tests is 8 (`tests/test_kernel.py:532`, `tests/test_kernel.py:404`).
5. The attention tests and the examples therefore cannot set the bound; the spike needs a synthetic stress kernel.
6. Proposed fail point: a loop of 1024 trips around 4 instructions compiles in 60 s on the CI CPU image, and a loop of 64 trips compiles in 5 s; the owner may move the numbers, but the spike must have both a count and a time.
7. If the spike fails, the fallback is C (unroll only where needed), not B, because C keeps the algebra but B also adds O11 on top of it.
8. An unrolled-size cap is a coverage limit (`NotImplementedError`, per O6), not a spec violation, so it does not conflict with decision 16.

Evidence: `U:68-72`, `T:101-104`, `T:137-140`, `tests/test_kernel.py:532`, `plans/reviews/oracle/architect.md` lines 9-11 and 90, `plans/oracle/review-response.md:70`.
External: XLA compile time grows with HLO instruction count; the StableHLO `while` op exists precisely to keep module size flat (openxla.org/stablehlo/spec, `while`), so the real cost of unrolling is compile time, which only a measurement settles.
Confidence: medium.
Changes my mind: a spike showing superlinear XLA compile time at tens of trips, or a real ACT kernel with a trip count in the tens of thousands.

### Q5.2

Position: B. The spec owns a tolerance bound as a function of K; the oracle is held to it; CPU bit-exact goldens stay as regression tests, not as the norm.

1. Clause one of R8, "accumulate in the result type", is not what the oracle does: XLA upcasts a `bf16` dot and accumulates wider (`U:603-604`, and the critic's citation of `cpu_compiler.cc:617-627`).
2. A norm the reference platform violates cannot be enforced; option A would make the oracle non-conforming on day one.
3. Option C, exact then round once, is not met by widening to `f64` either: an `f64` sum of K terms is not exact, so "rounded once" holds only up to near-ties; the existing test passes by data, not by construction (`tests/test_mini_trn.py:91-92`).
4. Truly exact needs an exact accumulator (`Fraction` or integer scaling), too slow for a 64x64 dot in CI and absent from GPU tensor cores (decision 15).
5. StableHLO leaves this open: "the default parameters are implementation defined" for dot algorithms, and reduce order is implementation-defined (openxla.org/stablehlo/spec, `dot_general`, `reduce`).
6. Option D (fixed order) forbids tensor-core and tiled-hardware accumulation, so the oracle could not model decision 15.
7. A tolerance in the standard inner-product form, |fl(x.y) - x.y| <= c * K * u * sum|x_i y_i| with u the unit roundoff of the accumulator type, is checkable by any implementation (Higham, Accuracy and Stability of Numerical Algorithms, inner-product error bound).
8. Hardware that accumulates wider is covered by the same bound; hardware that accumulates narrower needs a larger c, which the spec states per word.
9. The oracle's own check is `taidl.testing` (Q5.7), with `tests/conftest.py:227` `ULPS` as the pattern.
10. Option E is the right statement for goldens only: the CPU goldens (`README.md:163`) are pinned to one jax version (Q5.10 F).

Evidence: `U:603-604`, `tests/test_mini_trn.py:91-92`, `tests/conftest.py:227`, `README.md:163`, `plans/reviews/oracle/critic.md` Missing list, openxla.org/stablehlo/spec.
Confidence: medium.
Changes my mind: a measurement that XLA CPU accumulates `bf16` dots in `bf16` when excess precision is off (Q5.10), which would make clause one true on the reference platform.

### Q5.3

Position: B. Every intermediate is a checked i64, evaluated on concrete values after unrolling; an overflow in a constraint is the same error.

1. The backend already checks every operator and returns "None past i64" (`act-backend/act_backend/expression_writer.py:72-94`, `CheckedWriter.operation`).
2. Under option A the oracle accepts `a * 2^40 * 2^40 / 2^80`, which the backend rejects; the reference is then looser than the thing it references.
3. Exact intermediates (A) are only simpler if the language never needs to say what hardware does; an address generator in hardware wraps or traps at 64 bits, so checking each step is the closer model.
4. After unrolling (Q5.1) the evaluator is a recursive function over the expression with a range test per node, about ten lines in Python.
5. Without unrolling, B needs symbolic per-step bound proofs over `while` loops, which is the hard route the architect rejects (`plans/reviews/oracle/architect.md:84`).
6. Option C differs from B only in message: a constraint that does not hold already raises `broken_constraint` (`T:~350`), so "does not hold" and "error" are the same outcome; use one error class with the overflow named.
7. Option D keeps a warning, which decision 16 does not allow.
8. `T:353-363` (state updates and constraints never warn) is the gap to close, with the same evaluator.

Evidence: `act-backend/act_backend/expression_writer.py:72-94`, `T:353-363`, `U:287,659,706`, `plans/reviews/oracle/architect.md:84`.
External: Rust `checked_*` returning `None` is the model the backend copies (doc.rust-lang.org/std/primitive.i64.html, `checked_add`); StableHLO integer ops wrap, so the check must be explicit.
Confidence: high on B over A; medium on dropping S6.2's separate wording.
Changes my mind: evidence that a real ISA relies on a wrapped intermediate that returns in range.

### Q5.4

Position: C. A per-row split, errors by default, with the flag computed in the module and read once per kernel.

1. Atomicity is cheap here: one kernel is one module, and HBM leaves only as the return of `execute` (`D:191-196`); state is a trace-time value that data cannot touch (`T:353-363`).
2. So "writes no region and no state" holds if the wrapper raises before returning outputs; O23 is a flag output and one host-side test, not a per-instruction commit protocol.
3. The flag is an extra `i1` module output: each guarded op contributes a `compare`, all are OR-ed with `or`, and one `reduce_or` ends the module; cost is a few lines per affected word.
4. Rows with no shared hardware value are errors: integer `divide` and `remainder` at zero and at `INT_MIN / -1`, float `convert` to integer out of range (StableHLO says the result is TBD, openxla.org/stablehlo/spec, `convert`).
5. Rows with an obvious total definition are guarded values via `select`: shifts past the width, `compare` default; the guard lives in the emitter so the module is the definition.
6. `bitcast_convert` byte order is a spec fact, not a guard: state little-endian and test it once per type pair.
7. StableHLO says integer `divide` at zero or `INT_MIN / -1` yields "an implementation-defined value" (openxla.org/stablehlo/spec, `divide`), so option D would leave decision 16 violated.
8. Option B alone defines a value for divide-by-zero that no hardware agrees on, which hides kernel bugs; option A alone makes shifts errors for no benefit.
9. Option E is acceptable only for the word "fault"; the row list belongs in the norm table, which the oracle implements as given.

Evidence: `D:191-196`, `U:651-806` (checks before tracing), `plans/reviews/oracle/critic.md` Missing list, openxla.org/stablehlo/spec.
Confidence: medium.
Changes my mind: a measured compile-time or runtime cost of the flag chain above a few percent on the attention kernel, which would push more rows to guarded values.

### Q5.5

Position: A with the exact range check of D. Parse the text with `Fraction`, round once to nearest-even into the target type, emit hex bits.

1. Today `float(text)` (`OV:383`) then `repr(float(value))` (`U:408-409`) rounds to `f64` first, so a narrow float rounds twice and an integer above 2^53 at a float type does too.
2. The MLIR documentation states the two literal forms and says nothing on rounding of a decimal to the target type (mlir.llvm.org/docs/Dialects/Builtin/, float attribute), so option B relies on parser behavior no document promises.
3. Hex bits (`U:411-420` already emits them for non-finite values) are exact and independent of the parser, so the path exists.
4. `Fraction(text)` reads decimal and exponent forms exactly; `inf` and `nan` stay on the bits path.
5. The rounder is about 30 lines: scale to the type's exponent range, round half to even on the mantissa, handle subnormals, saturate per `NO_INFINITY`.
6. It is testable exhaustively for the 8-bit types and by midpoints for `f16` and `bf16` against `ml_dtypes`.
7. The range check compares the exact value with the type bound `largest + half ulp` (ties to even keeps a no-infinity type's largest), which `U:404-406` already encodes but evaluates on an `f64`.
8. Option C leaves the generators disagreeing, which contradicts "fidelity before structure" in `plans/oracle/plan/decisions.md`.

Evidence: `OV:383`, `U:404-420`, `plans/oracle/plan/fidelity.md`, mlir.llvm.org/docs/Dialects/Builtin/.
Confidence: high.
Changes my mind: a pinned MLIR release whose source (`Parser::getFloatAttrValue`) rounds a decimal directly into the target semantics, which would make B a one-line equal.

### Q5.6

Position: C with D. Aliasing is equality of concrete windows; any other input/output overlap, and any output/output overlap, is a call-time error; the pairing question stays open.

1. The oracle loads all inputs before any output stores (`IC:169-181`), so with identical windows an in-place mark changes no value; the mark only decides the storage claim of decision 17.
2. A partial overlap between an input and an output is the case where a defined result needs a rule; "error" is the simplest rule and fits decision 16.
3. Overlapping outputs today let the later write win silently (`IC:667-681`), which is a violation decision 16 forbids.
4. Region text equality (R5) breaks on `2 * %i` against `%i * 2`; after unrolling every region is a concrete window, so comparison on windows is exact and needs no normalizer.
5. The kernel-level check already refuses overlapping inputs and constants in HBM (`D:125-139`), so a call-level counterpart has an existing pattern.
6. `set_inplace` takes one flag per input with no output named (`taidl/instruction.py:163-172`); the backend test marks a `d2` load while the output is another data model (`act-backend/tests/test_extraction.py:507`), which identical region text cannot express.
7. So R5's removal of the mark is safe only if no ISA marks an input in place for an output in a different buffer; that is a fact to check in the backend corpus, not an opinion.
8. Until it is checked I keep B's semantics (marked input is the target, count and type must match, O10) as the fallback.

Evidence: `IC:169-181`, `IC:667-681`, `taidl/instruction.py:163-172`, `act-backend/tests/test_extraction.py:507`, `D:125-139`.
Confidence: medium on C and D; low on whether the mark can go.
Changes my mind: a survey showing every in-place mark pairs same-buffer, same-window input and output.

### Q5.7

Position: B first, with C goldens; the StableHLO interpreter is an optional CI job after the owner approves the dependency.

1. The bugs found so far (D1, D3 literal rounding) are lowering bugs; a second interpreter fed the same emitted text agrees with XLA and misses them, while a NumPy reference over the instruction record catches them.
2. The StableHLO interpreter's value is the other class: XLA execution against the StableHLO spec; it is a rare class, and the interpreter's own float checks use one tolerance of 0.0001 for everything and call transcendental precision implementation-defined (github.com/openxla/stablehlo/blob/main/docs/reference.md).
3. The interpreter is built from LLVM/MLIR and CHLO needs legalization first, so it adds a heavy build to the CI image; the engineering rules require asking before a new dependency.
4. The tests already hold NumPy references for the elementwise and reduce words (`tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`), so factoring them into one module over the record is size M, matching O13.
5. Use `Fraction` literals in the reference so it does not copy the double rounding.
6. Golden module-text tests (C) pin the lowering form for each word and one case per norm row; they are cheap and make the byte-identical gate of Q5.9 a test.
7. A reference cannot settle a norm: its comparison is exact only where a norm fixes the result and a bound elsewhere (Q5.2).

Evidence: `tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`, `plans/reviews/oracle/critic.md` O13 row, github.com/openxla/stablehlo/blob/main/docs/reference.md.
Confidence: medium.
Changes my mind: a lowering bug class that NumPy over the record cannot see but the interpreter does, found in a trial run on the 77 words.

### Q5.8

Position: D. Split the item: move the shape module first, then the front-end checks as their own item, and do E now.

1. `generic/shapes.py` is 543 lines of rules copied from the front end (D8), used at call time through `check_shapes` (`T:CHECK_SHAPES`); moving it is mechanical and removes the largest duplicate.
2. The region, S8.3, compare-fit, literal-shape, and range checks (`OV:251-289`, `OV:373-389`, `semantic_generator.py:41-56`) are generation-time and need the front end to own and run them (`taidl/docs/conformance.md:59`).
3. These two jobs have different risk: the move needs a pin and an import path, the checks need new front-end code and diagnostics; one item sized M hides that.
4. Run-time import of the front end (A) makes every generated oracle depend on the same taidl tag; today the oracle directory is a self-contained copy (`taidl_to/generator.py:~56-70`), so a vendored generated copy (C) keeps that property and removes the pin.
5. For the shape module I prefer C: the generator copies the front end's module into `runtime/` at generation, stamped with the tag, so there is still one source.
6. Option E needs no move: spec errors as `ValueError`, coverage gaps as `NotImplementedError`, and it makes tests assert one class.
7. Decision 16 keeps per-call checks in the oracle, so the call-time `disagreement` entry stays regardless.

Evidence: `OV:251-289,373-389`, `taidl/docs/conformance.md:59`, `generic/shapes.py` (543 lines), `plans/reviews/rulings/audit.md` R62.
Confidence: medium-low on C over A; high on the split and on E.
Changes my mind: a requirement that a generated oracle run with only the oracle package installed would favor C; a requirement for one install of taidl everywhere would favor A.

### Q5.9

Position: B and C together: fidelity items, then loop deletion, then the split; one thin function per instruction; the generic emitter only if measured.

1. The byte-identical gate compares against a baseline, so O8 and O21 must land first; otherwise the baseline moves under the structural steps.
2. Deleting the loop algebra before the split avoids moving about 260 of 989 lines of `utils.py` that a later step deletes (`plans/reviews/oracle/architect.md:74`).
3. Moving runtime code out of `API_FILE` (`T:26-293`) is independent of loops and can land with O1, since ruff and tests do not see template text today.
4. Per-instruction functions with a thin call keep tracebacks naming the instruction; this is A's shape and I keep it.
5. O3's generic-form emitter is conditional on O2's measured compile time (critic); until measured, keep the `PRETTY` form (D in part), since a changed module text moves every golden.
6. Lowering through `jax.lax` (E) adds decompositions and loses the one-to-one StableHLO name (`OV:124`).
7. The C++ debug handler build depends on jax, nanobind, and CUDA only; `CMakeLists.txt` takes no ISA input, yet `generator.py:72-84` rebuilds it at every generation.
8. So F's second half: build once per install and per jax version, and copy the artifact; this removes a cmake dependency and minutes from every generation.

Evidence: `T:26-293`, `taidl_to/xla-debug/CMakeLists.txt`, `taidl_to/generator.py:72-84`, `plans/oracle/plan/overhaul.md` ordered steps, `plans/reviews/oracle/critic.md` O3 row.
Confidence: medium; high on F.
Changes my mind: a measurement that the per-generation build is under a few seconds, or a reason the handler must be compiled against the ISA's custom call names.

### Q5.10

Position: A, F, and D together, with B and C as small cleanups; E only if measured.

1. `xla_allow_excess_precision` defaults to true, so a `bf16` chain such as softmax (`examples/mini_trn/mini_trn.py:101-104`) may skip intermediate rounding; the oracle claims per-op rounding, so the flag must be off.
2. The oracle compiles through `get_compile_options` and `compile_and_load` (`D:47-58`); the flag can be set in the compile options' build options instead of a process-wide `XLA_FLAGS`, which scopes it to the oracle's modules (assumption to test: `debug_options.xla_allow_excess_precision` is writable there).
3. A per-op rounding test on both platforms (a `bf16` add-then-multiply chain against NumPy) is the proof; without it the setting is a claim.
4. `jax_enable_x64` and `jax_default_matmul_precision` are set process-wide (`D:22-23`); the per-dot `HIGHEST` attribute is in the module (`U:603-604`), so the matmul flag is redundant (B).
5. x64 serves `jax.device_put` of 64-bit arrays only (`D:188`); a context manager (C) or feeding bytes as `uint8` arrays removes the global with no effect on the module text.
6. D is a statement, not a change: the CPU is the normative platform, GPU is held to bounds (`README.md:163`), and each deviation is a coverage-table row; this is the only honest position given decision 15's tensor cores.
7. E is a one-hour measurement: compile an `f16` `triangular_solve` on CPU and read the optimized HLO for converts; `U:498-511` widens on every platform, which is a choice, not a proven need.
8. F is cheap and protects everything above: `jax==0.11.2` is pinned (`pyproject.toml:18-19`); a bump regenerates goldens, diffs them, and re-measures `ULPS` (`tests/conftest.py:227`).

Evidence: `D:22-23,47-58,188`, `U:498-511,603-604`, `examples/mini_trn/mini_trn.py:101-104`, `pyproject.toml:18-19`, `plans/reviews/oracle/critic.md` Missing list.
External: XLA does not promise bit-identical results across versions or backends for transcendentals; the StableHLO reference interpreter calls their precision implementation-defined (github.com/openxla/stablehlo/blob/main/docs/reference.md).
Confidence: high on A, F and D; medium on scoping through compile options.
Changes my mind: the excess-precision flag proving to have no effect on the pinned CPU and GPU pipelines, which would turn A into a test only.
