# Panel 5 round 2: Opus (numerics and verification)

Lens: the oracle is the ground truth for the backend, so each of its behaviors must be stated, reproducible, and checkable against something independent of it.
Evidence checked this round:
- No file under `act-oracle/examples/` calls `start_loop` (grep is empty).
- `decorator.py:191-195` returns every output of one `executable.execute` call together.
- `act-backend/tests/test_extraction.py:154-163,507`: `load_d2` marks a `d0` input `u8[n*128]` in place for a `d2` output `bf16[n,64]`.
- `act-oracle/taidl_to/generator.py:58-66` copies `generic/` into each generated oracle, so a generated oracle is self-contained today.
- Owner decision 17 reads "The output reuses the marked input's storage".
- jax `compiler.py` `get_compile_options` passes unknown keys as `compile_options.env_option_overrides`, a per-compile DebugOptions override.

### Q5.1

Strongest opposing argument (Sonnet): the bound cannot be set on the attention tests and the examples, since no example calls `start_loop` and the largest test trip count is 8 (`tests/test_kernel.py:532`).
Concession: verified; my round-1 bound measured on those files would pass trivially and so could not fail.
The spike needs a synthetic stress kernel with a stated trip count and a stated time, as Sonnet proposes (1024 trips around 4 instructions in 60 s; 64 trips in 5 s).
Sonnet's point that an unrolled-size cap is a coverage limit (`NotImplementedError`, O6) and not a spec violation keeps the bound consistent with decision 16.
Final position: D, with the fail point set on a synthetic stress kernel (trip count plus compile time), not on the examples.
Confidence: medium-high.
What still divides the panel: only the numbers, which the owner may set; all three agree on D and on C as the fallback if the spike fails.

### Q5.2

Strongest opposing argument (Sonnet): C is not met by widening either, since an `f64` sum of K terms is not exact; "rounded once" then holds only away from near-ties and large cancellation, so the passing test (`tests/test_mini_trn.py:91-92`) passes by data, not by construction.
Concession: correct; my round-1 step 4 already admitted near-ties, and cancellation makes it worse, since the `f64` error scales with sum |x_i y_i| and not with the result.
A norm stated as "exact then round once" would therefore be one the oracle itself breaks on adversarial data, which is the same defect I held against A.
Truly exact accumulation (`Fraction` or scaled integers) is too slow for CI and is not what any tensor core does (decision 15), so it cannot be the norm either.
Fable's evidence that XLA CPU computes `bf16` in `f32` and upcasts dots (`cpu_compiler.cc`, `FloatNormalization`, `OperandUpcaster`) kills A's first clause on the reference platform.
So the norm is B: a spec-owned bound in the inner-product form, error at most c K u sum |x_i y_i|, with u the unit roundoff of the accumulator type and c stated per word.
C survives as the oracle's implementation: widen narrow dots and reduces (`utils.py:597-610`) so the oracle sits far inside the bound.
Bit-exact CPU goldens (`README.md:163`) stay as regression tests pinned to one jax version (Q5.10 F), never as the norm.
Final position: B as the norm; C's widening as the oracle's means; bit-exactness is a regression property only (changed from C plus B).
Confidence: medium-high.
What still divides the panel: Fable keeps C as the norm for narrow types; the split is whether a norm may claim exactness that holds only off near-ties.

### Q5.3

All three hold B: every intermediate is a checked i64 on concrete values after unrolling, matching `expression_writer.py:72,94`.
Strongest opposing argument (Fable, on S6.2): an overflowing constraint may be documented as "does not hold" if the owner prefers.
Rebuttal: Sonnet shows a failed constraint already raises `broken_constraint`, so "does not hold" and "error" reach the same outcome; one error class naming the overflow is the honest form.
I keep one addition the others did not state: i64 checking is order-sensitive, so the spec must fix the evaluation tree as parsed.
Final position: B, one error class, evaluation order fixed as parsed (unchanged).
Confidence: high.
What still divides the panel: nothing of substance; the S6.2 wording goes to the language track.

### Q5.4

Strongest opposing argument (Fable): default rows to defined values by `select` guard, since XLA's divide-by-zero value is stable, no ISA in `examples/` traps, and commit-after-check "doubles device-host traffic" and complicates the module.
Rebuttal on cost: `decorator.py:191-195` returns all outputs from one `execute`, so the flag is one extra `i1` output and one host test before returning; no output is held or copied, so traffic does not double.
Rebuttal on default: a defined value for integer divide by zero makes the oracle a conformance judge of a value no hardware promises, so a backend that traps or returns another value is "wrong" for no reason.
An error makes the program invalid instead, which every hardware satisfies and which catches the kernel bug the oracle exists to catch.
Concession: Fable is right that XLA's guard value could change under a jax bump, which argues against D as well as for an explicit row in the norm table.
Final position: C; errors by flag for integer `divide` and `remainder` at zero and `INT_MIN / -1` and for out-of-range float-to-integer `convert`; guarded values for shifts; little-endian for `bitcast_convert`; `compare` default is a front-end rule (unchanged).
Confidence: medium-high.
What still divides the panel: the default row; Sonnet and I default to error, Fable to a guarded value.

### Q5.5

All three hold A with the range check on the exact value; no rebuttal is needed.
Final position: A plus exact-value D (unchanged).
Confidence: high.
What still divides the panel: nothing; Fable's claim that MLIR parses decimals through a double is unverified here but does not decide the question.

### Q5.6

Strongest opposing argument (Fable): decision 17 makes the mark derivable, so `set_inplace` is redundant text the oracle never reads (D10), and identical region text plus a call-time overlap error covers every case.
Rebuttal: decision 17 reads "The output reuses the marked input's storage", so the mark is in the binding text; deriving it is a panel 1 or 2 reading, not this panel's.
The backend case at `test_extraction.py:154-163,507` pairs a `d0` input with a `d2` output, so identical region text cannot express it.
New finding: the same case pairs `u8[n*128]` with `bf16[n,64]`, equal in bytes but not in element count or type, so O10's check as drafted ("raises when element count or type differ") would reject a case already in use.
So under B the oracle's pairing check must compare byte size, or the backend test changes; that is a fact for the owner, not an opinion.
Agreement with all: overlapping outputs (`IC:667-681`) and partial input/output overlap become call-time errors on concrete windows after unrolling.
Final position: B plus C, with O10's check on byte size rather than element count and type (refined).
Confidence: medium.
What still divides the panel: whether the mark stays (Sonnet and I, pending a corpus survey) or goes (Fable); and whether D's window comparison is the aliasing rule (Sonnet) or only a check (Fable).

### Q5.7

All three hold B, with C's module-text goldens, and the StableHLO interpreter at most an optional job after owner approval of the dependency.
Final position: B plus C (unchanged).
Confidence: medium-high.
What still divides the panel: nothing.

### Q5.8

Strongest opposing argument (Sonnet): a generated oracle is a self-contained copy today (`generator.py:58-66`), so a run-time import of the front end (A) adds a version pin, while a copy the generator makes from the installed front end at generation time keeps one source.
Concession: verified; drift comes from hand-maintained copies (D7), and a copy written by the generator at each generation cannot drift from the front end it was generated with.
The static checks (`semantic_generator.py:41-56`, `OV:251-289,373-389`) run at generation in the front end, so they need no run-time import at all.
Only the call-time shape module (`generic/shapes.py`) needs to reach the generated oracle, and the copy-at-generation path already exists for `generic/`.
Fable's jax and jaxlib analogy fits a native library that cannot be copied, not a pure Python module.
Final position: D; static checks move to the front end and run at generation; the shape module is copied from the front end at generation, stamped with its tag; E now (changed from A as end state to C-at-generation).
Confidence: medium.
What still divides the panel: Fable holds run-time import (A); the owner may settle it by stating whether a generated oracle must run without taidl installed.

### Q5.9

All three hold fidelity first (B), then loop deletion (C), then the runtime split and O2 and O3 (A), one thin function per instruction, no `jax.lax` (E), and the debug handler built once per install (F).
Sonnet adds two refinements I accept: the `API_FILE` move to `runtime/` is independent of loops, and the `PRETTY` form stays until O2's compile time is measured.
Final position: B, C, A; reject E; F once per install and per jax version (unchanged).
Confidence: medium-high.
What still divides the panel: nothing of substance.

### Q5.10

Strongest opposing argument (Fable): excess precision is a backend-start flag, set through `XLA_FLAGS` before the backend initializes, which is why O21 must run first.
Rebuttal: jax `get_compile_options` passes unknown keys as `env_option_overrides` on `CompileOptions`, a per-compile DebugOptions override, and the oracle compiles through that path (`decorator.py:47-58`).
So Sonnet's scoping is supported: set `xla_allow_excess_precision` false per oracle module, which leaves a user's other jax code untouched, consistent with removing the other process-wide globals (B).
Its effect must still be proven by the per-op rounding test (a `bf16` chain against NumPy on both platforms), since an ignored override would be silent.
On C against B: Fable and Sonnet both show x64 serves only `jax.device_put` (`decorator.py:188`); feeding raw bytes removes the global, so C is moot.
Final position: A set per compile through `env_option_overrides` and proven by test; B; D; E measured; F (refined on the mechanism).
Confidence: high for A, D, F; medium for the override mechanism until the rounding test runs.
What still divides the panel: process-wide flag (Fable) or per-compile override (Sonnet and me); all agree on A, D, E, and F.
