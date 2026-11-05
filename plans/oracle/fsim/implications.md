# fsim-lib: implications for the oracle plan

Under owner decision 23, the oracle moves to fsim-lib later; this plan does not pivot to it.
This page sorts the current O items and panel 5 decisions by their value on the way to fsim, and lists what a later migration plan needs.
Paths start at the fsim-lib repo root; `act-oracle/` and `plans/` name the other trees.

## Classes

| Class | Meaning |
| --- | --- |
| carries | Engine-independent: the work lands in the spec, the front end, the lowering or the tests, and fsim reuses it unchanged. |
| carries, reworked | The rule carries, but its mechanism is rebuilt on fsim; the tests written now still apply. |
| interim | Serves only the jax path; wasted effort if fsim lands within the plan's two quarters. |
| moot | Already moot by another item, independent of fsim. |
| split | Part of the item carries and part is interim; the row says which. |

## Plan items

| Item | Class | Why |
| --- | --- | --- |
| O8 literals rounded once | carries | Literal bits sit in the module text either engine compiles. |
| O9 `f16` `triangular_solve` | carries, reworked | Same XLA 0.11.2, but fsim's CPU compile forces `xla_backend_optimization_level=1` (`fsim/src/compile_options.cc:54-63`); remeasure there. |
| O21 no excess precision | carries, reworked | The per-op test carries; on fsim the flag is one more row in `kCompilerFlags` (`fsim/src/compile_options.cc:54-63`). |
| O22 norm table | carries | Guards are emitted ops; fsim adds no rounding of its own. |
| O23 fixed values for data cases | carries | Guards are emitted ops; no error flag exists, so `@main` returning exactly the state (`fsim/src/pipeline.cc:325-358`) is no constraint. |
| O24 call-time errors | carries | Concrete trace-time checks; the NKI driver does the same (`drivers/nki/design.md:303-316`). |
| O10 overlap constraints and in-place marks | carries | Decisions 17, 24 and 31 and panel 9 are lowering and check rules; fsim's own aliasing is buffer-level (`fsim/passes/passes.cc:101-110`). |
| O11 wrapping starts | moot | Only if O20 fails, as today. |
| O12 drop jax configuration | interim | No jax runs in an fsim process; the globals and `device_put` vanish (`act-oracle/taidl_to/generic/decorator.py:22-23`, `act-oracle/taidl_to/generic/decorator.py:188`). |
| O13 differential reference | carries | Compares results over the record; it can also compare the two engines. |
| O19 unroll spike | carries, reworked | Unrolling matches fsim's model (`drivers/nki/design.md:163`); the compile-time bound changes from one module to modules of at most the soft trigger. |
| O20 unroll every loop | carries | Unrolled values are plain integers, which a driver needs to resolve addresses at trace time. |
| O1 runtime code out of `API_FILE` | split | `runtime/trace.py`, `emit.py`, `checks.py` carry; `runtime/decorator.py` and `kernel_io.py` are interim. |
| O2 one instruction record | carries | The seam a C++ driver or a trace emitter consumes; the most valuable item for migration. |
| O3 emitter calls, generic form | carries | fsim's pass stage parses any MLIR form of `func` and `stablehlo` (`fsim/passes/passes.cc:79-83`). |
| O18 drop line splitter | carries | Small and independent; generated Python stays as the kernel API. |
| O5, O6, O7 | carries | Dead code, error classes and front-end checks do not touch the engine. |
| O14, O17 coverage docs | carries | The coverage table later gains an engine column. |
| O15, O16 naming and log lines | carries | Engine-independent. |

## Panel 5 decisions

| Decision | Class | Why |
| --- | --- | --- |
| Q5.1 unroll, fail point numbers | carries, reworked | The policy carries; the numbers apply to the one-module jax path and to fsim's `taidl` variant only. |
| Q5.2 widen narrow dot and reduce to f32 | carries | An emitted convert; the NKI driver already computes in f32 at `HIGHEST` (`drivers/nki/src/stablehlo/Module.cpp:516-521`). |
| Q5.3 checked i64 addressing | carries | Trace-time arithmetic; a C++ driver needs checked builtins for the same rule. |
| Q5.4 norm table | carries | As O22 and O23. |
| Q5.5 `Fraction` literals | carries | Generation-time work; the emitted hex bits are engine-independent. |
| Q5.6 in-place and overlap | carries | Decision 24 settles the cross-buffer pairing: rejected statically; same-buffer pairs are constraints (panel 9). |
| Q5.7 NumPy reference | carries | Engine-independent. |
| Q5.8 shape module, decision 25 | carries, reworked | The copy lands in a Python oracle; a C++ path needs the shapes checked at generation or ported. |
| Q5.9 order B, C, A | carries | The order holds; its C++ debug handler is interim. |
| Q5.10 A excess precision per compile | carries, reworked | fsim already compiles with per-compile overrides (`fsim/src/compile_options.cc:75-83`). |
| Q5.10 B drop jax globals after x64 test | interim | No jax globals under fsim. |
| Q5.10 D CPU normative, GPU bounded | carries | fsim has CPU and GPU compilers on the same plugin version (`fsim/README.md:26-33`). |
| Q5.10 E `f16` solve measurement | carries, reworked | As O9. |
| Q5.10 F jax bump regenerates goldens | carries, reworked | Becomes a PJRT entry bump in fsim's setup. |

## Effort that would be wasted if fsim lands soon

- O12 and Q5.10 B: jax globals and `device_put` do not exist on fsim.
- The jax-specific part of O1: `runtime/decorator.py` and `runtime/kernel_io.py` move code the fsim path drops.
- The `xla-debug/` handler and its build per install and jax version (`act-oracle/taidl_to/generator.py:56-84`): fsim registers no FFI target.
- Q5.1's compile-time numbers, if measured only on the one-module jax path.
- Keep each of these at its current S size and avoid widening them.

## Items most worth landing first

- O2: the instruction record is the input a migration consumes, whatever form the driver takes.
- O8, O21, O22: they fix the numbers fsim must later match bit for bit, so the engine swap has a stable baseline.
- O24 and O10: trace-time checks on concrete values are exactly how the NKI driver works.
- O20: fsim assumes an unrolled stream and bounds compile by splitting modules.

## What a later migration plan needs

| Need | Detail | Evidence |
| --- | --- | --- |
| State-threading `@main` | Data models become state tensors; inputs and outputs move through `d0`'s host memory. | `fsim/include/fsim/pipeline.h:2`, `act-oracle/taidl_to/templates/templates.txt:39-59` |
| Driver form | A choice between a generic C++ driver over a TAIDL-TO trace and a generated per-ISA C++ driver. | `drivers/nki/design.md:202-230` |
| Kernel front end | A Python recorder from the oracle API to a trace, or a C++ kernel API; fsim has no Python binding. | `drivers/nki/record/record.py` |
| Error channel | An error-flag state tensor and a map from module to instruction for messages. | `fsim/src/runtime.cc:87-101` |
| Compile flags | Excess precision off; decide whether optimization level 1 stays. | `fsim/src/compile_options.cc:54-63` |
| Pass-stage equivalence | Bit-identical goldens between `custom-cpu` and `xla-0.11-cpu`, since the stock optimization rewrites ops. | `fsim/passes/passes.cc:85-93` |
| Engine equivalence | Every golden equal between the jax oracle and fsim's `taidl` variant before any default switch. | `fsim/src/pipeline.cc:448-454` |
| Element types | `ui8` storage for sub-byte and `pred` data models, or an `ElementType` extension upstream. | `fsim/include/fsim/tensor.h:11-26` |
| Debug output | A replacement for the `print_handler` custom call. | `act-oracle/taidl_to/templates/templates.txt:290` |
| Run-time values | Any value read from state for addressing costs a submit and a sync. | `drivers/nki/src/Driver.cpp:104-109` |
| Platform and packaging | x86_64 only, container-built, 1-2 h plugin build from source; owner approval as a new dependency. | `fsim/README.md:142-146` |
| Benchmarks | TAIDL-kernel timings for `taidl` against `overlap`; none are checked in for any driver. | `drivers/nki/design.md:49-52` |

## Open questions for the owner

1. Driver form: generic driver over a TAIDL-TO trace, or a generated per-ISA driver? Default: generic driver, so per-ISA code stays in TAIDL-TO.
2. Kernel API after migration: keep the Python `api.py` as a recorder? Default: yes, so `examples/` and tests keep working.
3. Optimization level: keep fsim's `xla_backend_optimization_level=1` for the oracle? Default: adopt the oracle's jax defaults until engine equivalence passes.
4. Interim items: keep O12 and the jax part of O1 in this plan? Default: keep at S size, defer anything beyond.
