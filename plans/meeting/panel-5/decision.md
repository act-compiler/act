# Panel 5 decision: oracle architecture and fidelity

Moderator: Opus.
Paths: `U` is `act-oracle/taidl_to/generic/utils.py`, `OV` is `taidl_to/oracle_visitor.py`, `IC` is `taidl_to/instruction_codegen.py`, `D` is `taidl_to/generic/decorator.py`, `G` is `taidl_to/generator.py`.
Citations marked "checked" were read by the moderator; the rest rest on the debaters' files.

| Question | Decision (one line) | Confidence | Owner? |
| -------- | ------------------- | ---------- | ------ |
| Q5.1 | D: unroll every loop; fail point set before the spike on a synthetic stress kernel; fallback C with an equivalence test | High on D; numbers open | Yes, the numbers only |
| Q5.2 | B is the norm: a spec-owned K-dependent bound with u of the declared accumulator type; the oracle widens narrow float dot and reduce to f32 in the module | Medium-high | No |
| Q5.3 | B: every intermediate is a checked i64 on concrete values; one error class; evaluation order as parsed | High | No |
| Q5.4 | C: one norm table; errors by in-module flag for divide, remainder, float-to-int convert; guards for shifts; little-endian bitcast | Medium-high | No |
| Q5.5 | A: `Fraction` parse, round once to nearest-even, hex bits; range check on the exact value | High | No |
| Q5.6 | B plus C: the mark stays as the declared pairing; equal windows allowed; partial overlap and overlapping outputs are errors | Medium | Yes, the cross-buffer pairing check |
| Q5.7 | B plus C: NumPy references over the instruction record with exact literals, module-text goldens; no interpreter dependency now | High | No |
| Q5.8 | D: split the item; static checks run in the front end at generation; E now; shape-module delivery is the owner's | Medium | Yes, run-time import or generated copy |
| Q5.9 | B, then C, then A; one thin function per instruction; reject E; debug handler built once per install and jax version | High | No |
| Q5.10 | A per compile, proven by a per-op test; B after a 64-bit feed test; D; E measured first; F | High on A, D, F | No |

### Q5.1: loop unrolling and the compile-time bound

Decision: D; unroll every loop, delete the loop algebra, `Unroll`, `while` emission, `held` and `fixed`, and set the fail point before the spike.
1. Unrolling is already the path for state updates and varying shapes (`T:101-104,137-140`), and per-call checks already enumerate iterations (`U:68-72`).
2. The symbolic algebra therefore adds no checking power, only a second path whose equivalence is untested.
3. No file under `act-oracle/examples/` calls `start_loop` (checked: zero hits), so a bound relative to the examples cannot fail.
4. The fail point is a synthetic stress kernel with a stated trip count and compile time on the CI image, plus a per-instruction op-count cap.
5. An unrolled-size cap is a coverage limit (`NotImplementedError`, O6), not a spec violation, so decision 16 holds.
6. If the spike fails, the fallback is C with a test that the `while` and unrolled paths agree, not B.
Evidence: `U:68-72`; `T:101-104,137-140`; grep over `examples/` (checked); round-2 concessions by Opus and Fable.
Dissent: none on the option; Sonnet's round-1 numbers (1024 trips in 60 s, 64 in 5 s) were withdrawn as unmeasured.
Confidence: high on D; the numbers are a value judgment for the owner.

### Q5.2: float `dot` and `reduce` accumulation norm

Decision: B as the spec norm, a bound of the form c K u sum |x_i y_i| with u the unit roundoff of the declared accumulator type and c stated per word, checked in `taidl.testing`.
The oracle widens narrow float dot and reduce operands to `f32` explicitly in the module and converts once to the result type; R8's first clause is dropped; CPU bit-exact goldens are regression tests only.
1. XLA CPU computes `bf16` ops in `f32` and upcasts dots (checked: `cpu_compiler.cc` adds `FloatNormalization(&bf16_support)`, `OperandUpcaster`, and `ChangeOpDataType(F16, F32, ...)` unless `xla_cpu_strict_dot_conv_math`).
2. So A's "accumulate in the result type" is false on the reference platform; a norm the oracle breaks cannot judge a backend.
3. C's "exact then round once" is not met by `f64` widening either: a sum of K products is exact only within 53 bits of exponent spread (Sonnet, conceded by Opus and Fable).
4. The oracle widens no float dot today: `DOT_WIDTHS` holds only `i2` and `ui2` (checked, `U:575`), and a `bf16` dot is emitted at `bf16` with `HIGHEST` (checked, `U:597-606`).
5. Explicit `f32` widening makes the oracle's value independent of an XLA pass and matches what XLA CPU computes now, so the CPU goldens should not move.
6. D forbids tree and tensor-core accumulation, which StableHLO leaves implementation-defined; E ties the norm to a jax pin.
7. The existing test holds the dot within two `bf16` ulps of an `f64` reference (checked, `tests/test_mini_trn.py:51-55,91-92`), not to a single rounding; it is consistent with B.
Evidence: `cpu_compiler.cc` (checked); `U:575,597-606` (checked); `tests/test_mini_trn.py:51-55,91-92` (checked); StableHLO spec `dot_general` and `reduce` as quoted by Opus and Fable.
Dissent: Opus keeps an exact `Fraction` accumulation for narrow types as a fallback if widening breaks goldens; Fable and Sonnet hold the bound covers it; no evidence yet decides.
Confidence: medium-high; the per-word constant c is not yet set.

### Q5.3: integer overflow in addressing and constraints

Decision: B; every intermediate is a checked i64 on concrete values after unrolling; an overflowing constraint raises the same error class with the overflow named; the spec fixes the evaluation tree as parsed.
1. The backend returns "None past i64 or by zero" per operator (checked, `act-backend/act_backend/expression_writer.py:72,94`).
2. Under A the oracle accepts programs the backend rejects, so the reference disagrees with the implementation on validity.
3. C's "does not hold" reaches the same outcome as a raised `broken_constraint`; one class is simpler.
4. D keeps a warning, which decision 16 forbids; state updates and constraints never warn today (`T:353-363`).
5. i64 checking is order-sensitive, so the as-parsed tree is a one-line spec rule (Opus).
Evidence: `expression_writer.py:72-94` (checked); `T:353-363`; `U:287,659,706`.
Dissent: none.
Confidence: high.

### Q5.4: element-level violations and atomicity

Decision: C; one norm table, row by row.
- Error, by in-module `i1` flag and host check before outputs return: integer `divide` and `remainder` at zero and at `INT_MIN / -1`; float-to-integer `convert` out of range or NaN.
- Defined value by `select` guard: shifts past the width (zero, or sign fill for arithmetic right).
- Stated fact: `bitcast_convert` is little-endian, tested once per type pair.
- Front-end rule, not a run-time row: `compare` default.
1. All outputs of a kernel come from one `executable.execute` call and return together (checked, `D:188-196`), so commit-after-check is a host test before return, not a per-instruction protocol.
2. Fable's round-1 cost objection (doubled traffic) was withdrawn on that evidence.
3. A defined value for divide by zero adopts XLA's lowering as the norm and hides the kernel bug; an error is satisfied by every hardware.
4. StableHLO leaves `divide` implementation-defined and `convert` overflow TBD, so D violates decision 16.
Evidence: `D:188-196` (checked); `U:651-806` (all checks before tracing); StableHLO `divide` and `convert` as quoted by Opus.
Dissent: none in round 2; Sonnet flags padded lanes with zero divisors as a false-alarm risk, answered by limiting the check to the declared region.
Confidence: medium-high; the flag chain's compile-time cost on the attention kernel is unmeasured.

### Q5.5: literal rounding and range checks

Decision: A with the exact form of D; parse with `Fraction`, round once to nearest-even, emit hex bits; compare the exact value with largest plus half an ulp.
1. `float(text)` (checked, `OV:383`) then `repr(float(value))` (checked, `U:408`) rounds twice for narrow types and for integers above 2^53.
2. Hex bits are already the path for non-finite values (checked, `U:411-420`), so A extends an existing path.
3. B depends on the pinned MLIR parser; Fable withdrew the double-rounding claim about `APFloat`, but no document promises the behavior per pin.
4. The tie rule at `U:404-406` is right; only its input must become exact.
Evidence: `OV:383`, `U:400-420` (checked).
Dissent: none.
Confidence: high.

### Q5.6: in-place marks and overlapping regions

Decision: B plus C, refined by the owner's input.
- The mark stays as the declared pairing: decision 17 names "the marked input", and window equality does not imply storage reuse.
- An unmarked input whose window equals an output window is allowed; inputs read before any write define its value.
- A partial overlap between an input and an output, and any overlap between two outputs, is a call-time error on concrete windows.
- A marked pair in the same buffer with unequal windows is an error.
1. The owner states that overlap alone does not settle read-after-write: `r0 = r1 + r0` may or may not reuse storage, and `r0 = r1 + r2` allows `r2` equal to or different from `r0`.
2. So the mark carries information that region text and windows do not, and is not redundant; R5's removal (A) is rejected.
3. The oracle loads every input before any store (checked, `IC:165-181`), so an equal window, marked or not, changes no value in the oracle.
4. Outputs store in declaration order (checked, `IC:97-111`), so overlapping outputs today resolve as the later write, an emission accident; the agenda's `IC:667-681` is out of range (the file has 224 lines).
5. The backend marks `load_d2` in place (checked, `act-backend/tests/test_extraction.py:507`): an `hbm` input `u8[n*128]` paired with a `d2` output `bf16[n,64]` (checked, `:154-163`).
6. That pair is equal in bytes but not in element count or type, so O10's check as drafted would reject a case in use (Opus).
7. The output a per-input flag pairs with is unstated (checked, `taidl/instruction.py:163-172`); panel 1 owns that wording.
Evidence: owner input; decision 17; `IC:97-111,165-181`, `test_extraction.py:154-163,507`, `instruction.py:163-172` (checked).
Dissent: Fable held A in round 1 and conceded; Sonnet holds that the mark-versus-derivation choice is not the oracle's, which the owner input now settles.
Confidence: medium; the cross-buffer pairing check goes to the owner.
Owner options for the cross-buffer check:
- Byte size: the oracle checks a marked pair by equal byte size, which keeps `load_d2` valid.
- Count and type (O10 as drafted): stricter, and the backend test changes.

### Q5.7: differential reference

Decision: B plus C; NumPy references in one module over the instruction record with `Fraction` literals; golden module-text tests per word and per norm row; the StableHLO interpreter is not added now.
1. The interpreter evaluates the emitted text, so a lowering bug passes both; the oracle's known bugs (D1, D3) are lowering bugs (Opus, Sonnet).
2. Its float check uses one tolerance of 0.0001 for every type (StableHLO `reference.md`, quoted by Sonnet and Opus).
3. It is a new dependency, which the engineering rule says to ask about first.
4. NumPy references already exist per test (`tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`).
Evidence: as cited by all three; no dissent on the evidence.
Dissent: none; revisit if an interpreter wheel pinned to the jax pin's StableHLO version exists.
Confidence: high.

### Q5.8: static checks and the front end

Decision: D; the item splits into the shape-module move and the front-end checks, each sized on its own; E (spec errors `ValueError`, coverage gaps `NotImplementedError`) lands now.
1. The region, type, compare-fit, literal-shape, and range checks are properties of the ISA text (`semantic_generator.py:41-56`, `OV:251-289,373-389`), so they run in the front end at generation and need no run-time import.
2. Only the call-time shape module reaches the generated oracle.
3. The generator copies `generic/` into each oracle (checked, `G:59-64`), so a generated oracle is self-contained apart from jax.
4. A copy written by the generator from the installed front end cannot drift from it (Opus); a run-time import with a version check fails loudly and serves the backend too (Fable).
5. Both satisfy the single-source end state all three agree on; the choice turns on packaging, a value judgment.
Evidence: `G:59-64` (checked); audit R62 sizing as cited.
Dissent: Fable holds A (run-time import); Opus holds C at generation; Sonnet treats them as equal.
Confidence: medium.
Owner options for the shape module:
- A, run-time import pinned to the same tag: one install everywhere, one copy shared with the backend, a loud version check.
- C, copy at generation stamped with the tag: a generated oracle runs without taidl installed, reusing the existing copy path.

### Q5.9: generated-oracle structure and order

Decision: B, then C, then A.
- First: fidelity items (O8, O21, the norm rows, and Q5.2's `f32` widening), so the byte-identical baseline is stable.
- Second: delete the loop algebra before splitting `utils.py`, so no deleted code is moved.
- Third: runtime code to `runtime/`, then O2; O3 only if O2's compile time is measured to need it; `PRETTY` stays until then.
- One thin function per instruction; lowering through `jax.lax` (E) is rejected.
- The C++ debug handler is built once per install and jax version.
1. The `cmake` call takes no ISA input (checked, `G:72-84`), so a per-generation build has no fidelity value.
2. `jax.lax` would add jax's own promotion and decompositions between the spec and StableHLO and lose the module text the goldens pin.
Evidence: `G:72-84` (checked); `T:26-293`; overhaul.md ordered steps as cited.
Dissent: none.
Confidence: high.

### Q5.10: platform and XLA configuration

Decision: A, B, D, E, F.
- A: `xla_allow_excess_precision` false, set per compile on the oracle's compile options; its effect proven by a `bf16` chain test against NumPy on CPU and GPU.
- B: drop `jax_default_matmul_precision` (`HIGHEST` stays per dot); drop `jax_enable_x64` only after a test with `f64` and `i64` arguments, or a bytes feed, shows no silent downcast.
- D: CPU is the normative platform; GPU is held to stated bounds; each GPU deviation is a coverage-table row.
- E: measure whether XLA widens an `f16` `triangular_solve` before keeping or dropping the widening at `U:498-511`.
- F: a jax bump regenerates and diffs every golden and remeasures `ULPS`.
1. On CPU both excess-precision gates sit inside the oneDNN block (checked, `cpu_compiler.cc`: `if (debug_options.xla_allow_excess_precision())` before `SimplifyFPConversions`), so the effect on the pinned wheel is an empirical question the test answers.
2. Both globals are set process-wide (checked, `D:22-23`); arguments pass through `jax.device_put` (checked, `D:188`), which is why x64 cannot go untested (Sonnet).
3. The per-compile route is supported by Opus and Fable from jax `compiler.py` but was not run; the test decides it, and a process-wide flag is the fallback.
Evidence: `cpu_compiler.cc` (checked); `D:22-23,188` (checked); `README.md:163`; `pyproject.toml:18-19`.
Dissent: none on the options; C (scoped x64) is dropped by all three.
Confidence: high on A, D, F; medium on the route and on B's x64 step.

## Items for the owner

- Q5.1: the stress kernel's trip count, instruction count, compile-time limit, and per-instruction op-count cap.
- Q5.6: the cross-buffer marked-pair check, byte size (keeps `test_extraction.py:507`) or element count and type (O10 as drafted).
- Q5.8: shape-module delivery, run-time import pinned to the tag (A) or a copy at generation stamped with the tag (C).

## Escalations to the owner from the round files

- None: no round file has an "Escalation to the owner" section.
