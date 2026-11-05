# Oracle plan: outside critic

Judged against how reference models are built: the StableHLO reference interpreter, the Sail RISC-V model, NumPy golden models, and NKI's CPU simulator.
Rulings 16 (no undefined behavior: every run-time violation is a defined error or a defined norm) and 17 (an in-place mark is true aliasing) are factored in.

## Verdict

The plan keeps the right things: StableHLO text naming each word's reference operation (`taidl_to/oracle_visitor.py:124`), exact integer expressions, atomic generation, NumPy references in the tests, and a byte-identical `api.py` as the gate for every structural step.
Its weakness is that it treats fidelity as a list of three found bugs, while the oracle's claim to be ground truth rests on XLA 0.11.2's choices wherever the spec is silent.
S8.4 delegates every word's meaning to StableHLO (`/home/user/taidl/docs/operation_meaning.md:7-8`), and StableHLO leaves open exactly the points the backend will be judged on: dot accumulation ("default parameters are implementation defined"), reduce order ("The order of reductions is implementation-defined"), integer divide and remainder at a zero divisor ("an implementation-defined value"), `convert` out of range ("TBD"), and accuracy ("StableHLO does not provide guarantees about numerical accuracy").
Reference: https://openxla.org/stablehlo/spec.
Ruling 16 turns each of those into a norm the L track must write and the oracle must enforce, and none has a plan item.
The structural half (O1, O4, O5, O18, O2, O3) is sound and well ordered among itself, but the small fidelity fixes should land before it, since they are independent and the structural gate needs a stable behavioral baseline.
One premise in `decisions.md` is wrong: the tests do compare with NumPy references, not with jax (`tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`); jax appears only for a bf16 bitcast (`tests/conftest.py:274`) and six narrow types (`docs/operations.md:140`).
So O13 is smaller than planned, and its "exact comparison for every non-transcendental word" cannot hold for float `reduce_add` or `dot` until a norm fixes their order.

## Items

| Plan item | Verdict (keep / change / drop) | Reason | Reference |
| --- | --- | --- | --- |
| O1 split `utils.py`, rename `runtime/` | keep | Pure moves, gated by a byte-identical `api.py`; the reserved-name diff is the right extra check (`taidl_to/api_generator.py:61-86`). | `overhaul.md` step 2 |
| O2 one line record | keep | Removes the `holding` toggle (`oracle_visitor.py:266-274`); the StableHLO interpreter likewise evaluates one op record per line. | https://github.com/openxla/stablehlo/blob/main/docs/reference.md |
| O3 emitter calls instead of templates | change | Make it conditional on O2's measured compile time and keep one function per instruction; a reference model is read when it disagrees with hardware, as the plan itself argues. | `decisions.md` templates ruling |
| O4 one word table | keep | Six lookups to one row per word; same shape as the interpreter's per-op dispatch. | `oracle_visitor.py:19-124, 208-221` |
| O5 dead branch, double visit | keep | Behavior-neutral, first. | `oracle_visitor.py:237-241` |
| O6 `ValueError` for spec errors | keep | Ruling 16 wants every spec violation a defined error; `NotImplementedError` then means coverage only. | `oracle_visitor.py:254-264` |
| O7 shape rules to the front end | keep | One checker for every generator is how Sail gets one model for many tools; the cross-repo pin is the real cost, stated. | https://github.com/riscv/sail-riscv |
| O8 round float literals once | change | Correct and first in order; MLIR's parser now reads a decimal literal directly in the target semantics, so emitting the spec's text verbatim is a one-line fix if the pinned jaxlib carries it, and hex bits are the fallback; today's `repr(float(value))` rounds through `f64` before MLIR sees it (`taidl_to/generic/utils.py:408-409`). | https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/Parser.cpp |
| O9 `triangular_solve` at `f16` | change | Decide the norm first: the CPU is the reference platform, so a GPU-only widening is a documented GPU deviation; measure whether XLA CPU already widens through `FloatNormalization`. | https://github.com/openxla/xla/blob/main/xla/service/cpu/cpu_compiler.cc (lines 718-746) |
| O10 in-place marks | change | Ruling 17 settles the question: the output is stored into the marked input's region; drop the "ask L" and the NaN-poison option; verify the declaration check of `instructions.md:56`; size S, Q1. | `/home/user/taidl/docs/instructions.md:50-56` |
| O11 wrapping loop starts | change | Rejection fits ruling 16, and a zero divisor in a loop expression is already refused at every iteration (`utils.py:77-80`); fold D19 and D20 into the item; the affine fast path stays deferred. | `utils.py:241-243`, `bugs.md` D11, D19, D20 |
| O12 scoped jax configuration | keep | Q2 is right; the x64 context manager is experimental; test first whether the text module needs x64 at all beyond array I/O. | `generic/decorator.py:22-23` |
| O13 NumPy reference interpreter | change | Factor the existing test references into a module over the line record (size M, not L); parse literals with `Fraction`, since NumPy's string parse also rounds through `f64`; exact comparison only where a norm fixes the order. | `tests/test_elementwise.py`, `tests/test_reduce.py:82` |
| O14 fold `operations.md` | keep | Keep the test-bounds table; it is the oracle's accuracy contract, as the interpreter's `expect_almost_eq` tolerance is. | https://github.com/openxla/stablehlo/blob/main/docs/reference.md |
| O17 coverage kept true by test | keep | Mirrors the interpreter's one test per op row. | `coverage.md` |
| O18 drop line splitter | keep | Simplicity, no behavior. | `template.py:61-156` |
| Ordered steps 1-9 | change | Move O8, the excess-precision flag, and the dot and reduce norms ahead of O1; they are S items with no structural dependency and they set the baseline the byte-identical gate compares against. | `overhaul.md` |
| Lowering ruling (StableHLO text) | keep | Naming the reference operation is the strongest fidelity argument; `jax.lax` would add its own decompositions. | `decisions.md` |
| Enumeration ruling (every iteration) | keep | Exactness over speed is the reference-model norm; Sail enumerates too. | `utils.py:82-86` |

## Missing

- `xla_allow_excess_precision` defaults to true and lets XLA skip intermediate rounding; the GPU pipeline runs `SimplifyFPConversions` under that flag (gpu_compiler.cc:2189-2190), the oracle sets no XLA flag (`grep XLA_FLAGS` empty), and the softmax example chains `exponential`, `reduce_add` and `divide` at `bf16` (`examples/mini_trn/mini_trn.py:101-104`). Add an S item: set the flag false before the backend starts, and test that a `bf16` chain rounds per op on both platforms. References: https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc, https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_compiler.cc.
- A numerics norm table, owned by L and enforced by the oracle: `dot` and float `reduce_*` accumulation and order, integer `divide` and `remainder` at zero and at `INT_MIN / -1`, `convert` out of range, shifts past the width, and the ulp bound per transcendental word. Ruling 16 requires each row to be an error or a norm.
- `dot` today relies on XLA's upcast (cpu_compiler.cc:617-627) and `HIGHEST` (`utils.py:603-604`); the test asserts "rounds to bf16 once" against an `f64` product (`tests/test_mini_trn.py:91-92`), which an `f32` accumulation over 64 terms does not guarantee. If the norm is "exact, rounded once", the oracle should widen narrow dots and reduces itself (to `f64` or exact integers) rather than inherit XLA's accumulation, as Sail computes with softfloat rather than host arithmetic.
- Run-time violation detection inside the module: every check today runs before tracing (`utils.py:651-806`), so a data-dependent violation such as a zero divisor in `divide` cannot be reported. Ruling 16 needs a flag computed in the module (compare, `reduce_or`) and raised after the run, or a norm that defines the value.
- A normative-platform statement in `docs/coverage.md`: the CPU is the reference, the GPU is held to bounds (`README.md:163`), and every GPU deviation (O9, excess precision) is listed there. NKI's simulator makes the same split, a CPU simulator beside the device, without stating numerics; the oracle should state them. Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.simulate_kernel.html.
- The version pin as a fidelity contract: `jax==0.11.2` is pinned (`pyproject.toml:18-19`), but no item says a jax bump regenerates and diffs every golden and re-measures the ulp table; XLA changes its transcendental approximations between releases.
- Overlap and placement marks (`set_apart`, `set_no_partial_overlap`, `/home/user/taidl/docs/conformance.md:27-28`): the kernel check refuses overlapping inputs and constants in HBM (`generic/decorator.py:125-139`), but ruling 16 also needs the per-instruction marks checked over every loop iteration, and the plan has no item for it.
- Literal rounding is a spec gap, not only an oracle bug: S7.5 gives literal forms (`semantics.md:69`) and S8.4 no rounding rule; L16 must say "nearest, ties to even, from the decimal" before O8 can claim exactness.
