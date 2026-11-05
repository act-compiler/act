# Panel 5 agenda: oracle architecture and fidelity

Scope: act-oracle (TAIDL-TO) lowering, loops, runtime, reference checks, numerics.
Questions are ordered blockers first, then majors, then the rest.
Options are listed neutrally; the draft answers (R28-R38 and others in plans/minutes/rulings.md) appear as one option each.
Abbreviations: `U` is `act-oracle/taidl_to/generic/utils.py`, `OV` is `taidl_to/oracle_visitor.py`, `T` is `taidl_to/templates/templates.txt`, `IC` is `taidl_to/instruction_codegen.py`.
Plan items O1-O24 are in plans/oracle/plan/; reviews are in plans/reviews/oracle/ (V verification, C critic, A architect) and plans/reviews/rulings/audit.md.

### Q5.1: Does the oracle unroll every loop at trace time, and who fixes the compile-time bound?

- Context: loop variables are symbolic today, with an algebra of about 400 lines (`U:15-260`), an `Unroll` retrace, and `stablehlo.while` emission.
- Context: unrolling is already the path for a loop around a state update or a varying shape (`T:101-104,137-140`), and every call-time check already enumerates iterations (`U:68-72`).
- Context: the draft needs a spike with "an agreed bound" that nobody has set; the semantics track's F4 rejects rather than unrolls.
- Options:
  - A. Unroll every loop (O19 spike, then O20); delete the loop algebra, `Unroll`, `while` emission, `held` and `fixed`; D11, D17, D19, D20 go by construction.
  - B. Keep the symbolic algebra and add O11: reject a loop-expression start whose 64-bit steps overflow, with D19 and D20 folded in.
  - C. Unroll only where the loop body needs it today (state update, varying shape) and keep `while` elsewhere.
  - D. Option A plus a stated compile-time bound set now (for example, on the attention tests and the examples), so the spike has a fail point.
- Evidence: R28, R38, O11, O19, O20, O24; A (steps 2 and 4, O11 row); V C14, D11, D17, D19, D20; audit R28 and R56; decisions.md "Loops"; `templates.txt:82`.
- Binding: decision 16 (every violation defined, checkable per call); decision 11 (3-4 engineers, two quarters).

### Q5.2: What is the norm for float `dot` and `reduce` accumulation, and how does the oracle meet it?

- Context: R8 states two clauses: accumulate in the result type, and land within a per-word tolerance of the exact result.
- Context: XLA CPU does not accumulate `bf16` in `bf16`; the oracle relies on XLA's upcast and `HIGHEST` per dot (`U:603-604`), and the test asserts a single rounding (`tests/test_mini_trn.py:91-92`).
- Context: the CPU goldens are bit-exact today (`README.md:163`); StableHLO leaves order and accumulation implementation-defined.
- Options:
  - A. Accumulate in the result element type, plus a per-word tolerance (R8 as drafted).
  - B. Tolerance alone, with the bound stated in the spec as a function of K and owned by the spec, implemented in `taidl.testing`.
  - C. Exact result rounded once; the oracle widens narrow dots and reduces itself (to `f64` or exact integers) so the oracle stays bit-exact.
  - D. Fix a summation order in the spec and have the oracle enforce it.
  - E. Leave XLA's behavior as the reference and document it on a normative platform (see Q5.10).
- Evidence: R8, R56 objection in review-response.md, O22, O13; C (Missing: norm table, `dot` upcast); audit R8, RA4, RA5; `cpu_compiler.cc:617-627` as cited by C; `tests/conftest.py:227` `ULPS`.
- Binding: decision 16 (a defined norm, easy to explain); decision 14 (StableHLO precedence where the papers are silent); decision 15 (GPU tensor cores in scope).

### Q5.3: How does the oracle treat integer overflow in addressing and constraint expressions?

- Context: expressions are exact today and warn past 64 bits only for attributes, region starts, sizes, and semantics dimensions (`U:287,659,706`); state updates and constraints never warn (`T:353-363`).
- Context: R56 says exact arithmetic with a fault only when the final value leaves signed 64 bits; S6.2 says an overflowing constraint does not hold; H4 says every intermediate is checked in i64.
- Context: the backend checks every intermediate in i64 (`act-backend/act_backend/expression_writer.py:72,94`).
- Options:
  - A. Exact intermediates; a final value outside i64 is an error at the call (R56, R38).
  - B. Every intermediate is i64; leaving i64 at any step is an error, checked on concrete values after unrolling (H4, A checked-i64 row).
  - C. An overflowing constraint evaluates as not holding (S6.2); other overflows are errors.
  - D. Keep the warning for starts and sizes; add no error.
- Evidence: R38, R56, O24, O11; objection R56 in review-response.md; audit R56 and RA8; A S1.3 table (line 34); V "constraint arithmetic overflow" and "initial state outside 64 bits"; `api_generator.py:43-44`.
- Binding: decision 16; decision 11.

### Q5.4: How are element-level violations detected and made atomic?

- Context: every oracle check runs before tracing (`U:651-806`), so a data-dependent violation such as integer divide by zero cannot be reported.
- Context: R55 says a fault "writes no region and no state"; that needs commit-after-check for violations found inside the compiled module (O23), a cost the draft does not size.
- Context: the cases StableHLO leaves open are integer `divide` and `remainder` at zero and at `INT_MIN / -1`, float `convert` overflow, shifts past the width, `compare` default, and `bitcast_convert` byte order.
- Options:
  - A. Element cases are errors: the module computes a flag (`compare`, `reduce_or`), the oracle raises after the run, and outputs commit only when no flag is set (O23).
  - B. Element cases are always defined values: the emitter writes guards (`select`, clamp) so the module fixes each result (A, O22).
  - C. A per-case split: errors for some rows, guarded values for others, chosen row by row in the numerics norm table.
  - D. Leave the value to XLA CPU and record it in the coverage table as the platform's result.
  - E. Leave the word "fault" and atomicity wording to the language track; the oracle implements whichever rows it is given.
- Evidence: R55, R57, O22, O23; C (Missing: in-module detection, norm table); A (S1.3 table lines 39, 44); audit RA6, RA7; V "bitcast_convert byte order", "float convert overflow".
- Binding: decision 16 (no undefined behavior; every case an error or one simple norm).

### Q5.5: How are literals rounded and range-checked?

- Context: `constant_value` reads `float(text)` (`OV:383`) and `mlir_literal` emits `repr(float(value))` (`U:408-409`), so a narrow float rounds twice (D1, D3); an integer above `2^53` at a float type does too.
- Context: S8.4 has no rounding rule; the generators disagree today.
- Context: R57 says floats round once, to nearest-even, out-of-range literals are rejected, and out-of-range backtick expressions fault.
- Options:
  - A. Parse with `Fraction`, round once to the target type, emit hex bits (O8 as planned).
  - B. Emit the spec's decimal text verbatim if the pinned MLIR parser reads it exactly, with hex bits as fallback.
  - C. Keep the `f64` path and document the double rounding in the spec.
  - D. The range check compares the exact value with the type bound, or the rounded value (`U:404-406` today).
- Evidence: R57; O8, D1, D2, D3; V "integer literal above 2^53"; C (literal rounding is a spec gap); decisions.md "Fidelity before structure"; `U:411-420` (non-finite by bits).
- Binding: decision 16; decision 14.

### Q5.6: What does the oracle do with in-place marks and overlapping regions?

- Context: nothing under `taidl_to/` reads `set_inplace` (D10); R5 derives aliasing from identical region text and removes the mark.
- Context: inputs load before any output stores (`IC:169-181`), and two overlapping outputs are unchecked today (the later write wins, `IC:667-681`).
- Context: `set_inplace` takes one flag per input, so the output pairing is open (`taidl/instruction.py:163-172`); a backend test marks a `d0` input in place for a `d2` output (`act-backend/tests/test_extraction.py:507`).
- Options:
  - A. Alias exactly when region texts are identical; remove `set_inplace` (R5).
  - B. Keep `set_inplace` as the declared pairing; the oracle stores into the marked input's region and raises when element count or type differ (O10).
  - C. Option A or B plus a call-time error for any other input/output overlap and for overlapping outputs (RA2, RA3, O24).
  - D. Equal regions spelled differently count as aliasing after normalization to concrete windows.
- Evidence: R5; O10, O24; D10, V C11; A (ruling 17 section, S1.3 lines 42 and 48); audit R5, RA2, RA3, R21; `act-backend/tests/test_extraction.py:507`.
- Binding: decision 17 (the output reuses the marked input's storage); decision 16; decision 10 (breaking changes batched in one major).

### Q5.7: What is the differential reference for the oracle?

- Context: the tests compare against NumPy references, not jax (`tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`); no XLA comparison run exists (V C28).
- Context: the StableHLO reference interpreter evaluates the emitted module text but is a new dependency, not on PyPI, needs CHLO legalization, and may follow a different StableHLO version than the jax pin (R6).
- Context: a reference cannot settle a norm; it makes its own choice.
- Options:
  - A. StableHLO reference interpreter first, NumPy only for words it lacks (R29, size M).
  - B. NumPy references factored into one module over the instruction record, `Fraction` literals, StableHLO interpreter optional (O13, size M).
  - C. Golden module-text tests for lowering plus one golden per norm case, with no second interpreter.
  - D. Run both A and B for different word classes.
- Evidence: R29; O13, O22; A (focus points, O13 row); C (O13 row, version pin); audit R29, R6, R49; V C28.
- Binding: decision 5 (planning only); the shared engineering rule requires asking before any new dependency.

### Q5.8: Which static checks move from the oracle to the front end, and how does the oracle consume them?

- Context: the oracle checks region type and count (`semantic_generator.py:41-56`), S8.3 types (`OV:251-264`), compare-type fit (`OV:276-283`), literal shapes (`OV:285-289`), and constant ranges (`OV:373-389`); the front end skips these (`taidl/docs/conformance.md:59`).
- Context: `TYPES` copies the S8.3 table row by row (D7); `generic/shapes.py` holds a copy of the shape rules.
- Context: R62 sizes the merged item (L3, L12, O6, O7) as M; the audit sizes it L or larger.
- Options:
  - A. Widened O7: all of these move to the front end; the generated oracle imports the front end's module at run time, pinned to the same tag (R31, R39).
  - B. Narrow O7: only the shape module moves; region and literal checks stay in the oracle.
  - C. Front end exports the rules and the oracle vendors a generated copy.
  - D. Split the item into the module move and the front-end checks, with its own size (audit R62).
  - E. Spec errors raised as `ValueError`, `NotImplementedError` reserved for coverage gaps (O6), independent of where the rules live.
- Evidence: R31, R39, R62, R25; O6, O7; A (O7 row, Focus points); V D5-D8, D13; C (O7 row); audit R62.
- Binding: decision 16 (checks per call remain in the oracle); decision 11.

### Q5.9: What is the structure of the generated oracle, and in what order do the steps land?

- Context: `API_FILE` holds about 260 lines of fixed runtime code as template text (`T:26-293`) that ruff and tests do not see.
- Context: the plan keeps one function per instruction so tracebacks name the instruction; O3 adds a generic-form emitter and a lowering table, conditional on measured compile time.
- Context: the gate for structural steps is a byte-identical `api.py`; the order of the fidelity fixes, the loop spike, O1, and O3 differs between the plan, the architect, and the critic.
- Options:
  - A. Runtime code to `runtime/` first (R30), then O2, O3 with O4 folded in; one thin function per instruction calling one runtime `call` (A).
  - B. Fidelity items (O8, O21, norms) before any structure so the byte-identical gate has a stable baseline (C, overhaul.md).
  - C. Delete loop algebra before splitting `utils.py` so no deleted code is moved (A, O1 after O20).
  - D. Keep per-word templates and the `PRETTY` form; no generic emitter.
  - E. Lower through `jax.lax` instead of StableHLO text.
  - F. Per-generation C++ debug-handler build stays, or moves to once per install.
- Evidence: R30, R37; O1, O2, O3, O4, O5, O18; decisions.md "Lowering target", "Templates and generated code"; overhaul.md; A (revised order); C (Ordered steps); `generator.py:72-84`; `api_generator.py:61-86`.
- Binding: decision 10; decision 11; decision 5.

### Q5.10: What is the oracle's platform and XLA configuration contract?

- Context: `xla_allow_excess_precision` defaults to true, no XLA flag is set, and the softmax example chains `exponential`, `reduce_add`, `divide` at `bf16` (`examples/mini_trn/mini_trn.py:101-104`).
- Context: the oracle sets jax globals process-wide (`decorator.py:22-23`); x64 serves only `jax.device_put` (`decorator.py:188`); the `HIGHEST` precision is per dot (`U:603-604`).
- Context: `triangular_solve` at `f16` is solved at `f32` on every platform (`U:498-511`); the GPU failure claim is untested; `jax==0.11.2` is pinned (`pyproject.toml:18-19`).
- Options:
  - A. Set excess precision false before backend start and test per-op rounding on both platforms (O21).
  - B. Drop the matmul-precision flag and avoid `jax.device_put` (R32, O12); keep the `HIGHEST` per-dot attribute.
  - C. Keep the x64 setting but scope it with a context manager.
  - D. State the CPU as the normative platform and list every GPU deviation in the coverage table; GPU held to bounds.
  - E. Measure whether XLA CPU already widens an `f16` solve, then decide `triangular_solve` widening (O9).
  - F. Treat a jax bump as a fidelity event: regenerate and diff every golden, and remeasure the ulp table.
- Evidence: R32, R48; O9, O12, O13, O14, O21; C (Missing: excess precision, normative platform, version pin); V D4, D14; audit RA5.
- Binding: decision 15 (GPU tensor-core instructions in scope); decision 16; decision 5.

## For the authors

- R36: keep the `kernel(...)` entry name; no change planned (glossary term).
- R37: rename `generic/` to `runtime/`; O1 covers it.
- R33 and O14: fold `docs/operations.md` into the coverage table and keep the test-bounds table; `docs/operations.md:103` states "rounds to the nearest value", contradicting `OV:383`.
- R35 and O18: drop the line splitter; `tests/conftest.py:278-282` ruff check and `tests/test_templates.py` layout cases go with it.
- V D12: log line names the final directory while copying into staging (`generator.py:62-65`).
- V D13: the `sh_copy` check at `U:554-555` is unreachable; delete it.
- V D15: only the first operand's type is checked (`OV:251-254`); moot for `dot`.
- V D16: sequential `str.replace` can re-substitute (`template.py:41-42`); fixed by deletion in O3.
- V D17: wording to reduce to "enumerates only the loop variables a value reads (`U:68-72`)".
- V C6: coverage row partial; `accelerator.py:33-44` accepts a data model after an instruction.
- V C14: warning scope narrower than the row states.
- V C16: `ROOT` is a front-end rule; reword the row.
- V S1.2: planning table keeps `supported` rows; the shipped table keeps `partial` and `rejected` only.
- V missing rows: `@a.` outside a start or constraint, computed divisor 0, `constant` operand forms, compare `type` rule, `dot`, `broadcast`, `pad` forms, `add_constraints`, `update=`, `constraints=`.
- A O5: delete the dead branch (`OV:237-241`) now; the double visit is fixed by O2.
- Audit R20, R21, R22, R26, R54, R58: backend or process rulings outside this panel's scope.
- Likely owned by other panels (scope in panels.md): R3, R4, R10, R12, R7 (panel 1); R44 clause ids and the word "fault" (panels 1 and 3); R62 sizing (panels 3 and 8).
