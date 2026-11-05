# The path to an all-C/C++ simulator

Owner decision 23 names fsim-lib as the direction of the oracle's execution, and this note records the follow-up path.
It is a direction, not a plan the current release carries: the oracle stays on its Python and jax path for now.
Paths under `taidl_to/` are the current oracle; paths under `fsim/` are the upload.

## What depends on what today

| dependency | where it lives | what it does |
| ---------- | -------------- | ------------ |
| Python | `taidl_to/*.py`, the generated oracle | generates the oracle; traces each kernel's instruction calls into one StableHLO text |
| jax | `taidl_to/generic/decorator.py:6-9`, `:22-33` | starts the platform, sets `jax_enable_x64` and `highest` matmul precision, holds the device |
| jax.extend / XLA | `decorator.py:47-58` | `backend.compile_and_load(module, [device], options)` compiles the text; `executable.execute(operands)` runs it (`:191-192`) |
| jax FFI + nanobind | `decorator.py:49-52`, `taidl_to/xla-debug/print_handler_main.cc` | the debug print handler, a `custom_call` with `api_version = 4` (`taidl_to/template.py:290`) |
| NumPy | `decorator.py:179-188` | input and output arrays, `device_put` per input |
| pyproject | `pyproject.toml:18-25` | pins `jax==0.11.2`, `jaxlib==0.11.2`, `nanobind`, `cmake` |

In the upload, the C++ library itself depends on these:

| dependency | where | what it does |
| ---------- | ----- | ------------ |
| PJRT C API header | `fsim/src/pjrt.h:9`, `third_party/pjrt/<entry>/include` | every runtime call |
| PJRT plugin `.so` | `fsim/src/runtime.cc:41-46`, `FSIM_PJRT_PLUGIN` | the XLA compiler and device, from a jax release's plugin build |
| XLA | inside the plugin | compiles StableHLO to machine code; its passes decide the actual arithmetic |
| MLIR + StableHLO static libraries | `fsim/passes/CMakeLists.txt`, `setup.sh:233-244` | the pass stage, parse and serialize at StableHLO 1.20.0 (`fsim/passes/passes.cc:28`) |
| jax releases, indirectly | `setup.sh:16-17`, `:83-86`, `:141-142` | the plugin tarballs and CUDA wheels `setup.sh` pins |

No Python runs inside the library: the plugin is a C library and `setup.sh` builds it with `MLIR_ENABLE_BINDINGS_PYTHON=OFF` (`setup.sh:242-244`).

## The minimal steps, in order

1. Emit the kernel's StableHLO from a C++ driver instead of the Python trace.
   The generated oracle today writes the module text in Python (`taidl_to/generic/utils.py`, `taidl_to/template.py`), so the generator gains a C++ target that emits the same text per instruction, as fsim-lib's NKI driver does for its ISA (`fsim-lib/drivers/nki/design.md:7-9`).
   Python then remains only at generation time.
2. Replace `decorator.py`'s `compile_and_load` and `execute` with `fsim::Pipeline::submit`, `end_trace` and `sync` (`fsim/include/fsim/pipeline.h:73-82`).
   The state tensors become the pipeline's `Tensor` list with `tf.aliasing_output` applied by the stage (`fsim/passes/passes.cc:105-109`), which removes jax, jaxlib and NumPy from the run.
3. Move the debug print out of `custom_call`.
   The oracle's `print_handler` needs jax's FFI registration (`decorator.py:49-52`); under fsim a debug read becomes a `sync` followed by a host-side print of the state, which the batch path already treats as a data-dependent read (`fsim/src/pipeline.cc:845-855`).
   This also removes nanobind and the per-oracle C++ build at generation time (`README.md:26`).
4. Keep PJRT as the last external dependency and pin it by header and plugin hash, as `setup.sh` does today (`setup.sh:109-119`, `:141-142`).
   At this point the oracle is C/C++ with one dlopen'd plugin, and jax appears only as the release that names the plugin build.
5. Only if PJRT itself must go: write a StableHLO interpreter or a direct lowering for the oracle's op set.
   The op set is the words of `taidl_to/docs/operations.md` plus `dynamic_slice`, `dynamic_update_slice`, `concatenate`, `while` and the bitcast and reshape views (`utils.py:822-843`, `:901-902`, `:958`).
   This step changes the arithmetic reference from XLA to the new code, so every fidelity bound in `operations.md:139-153` is re-measured.

## What each step removes

| step | removed from the run | still present |
| ---- | -------------------- | ------------- |
| 1 | Python tracing per kernel | Python at generation; jax at run |
| 2 | jax, jaxlib, NumPy, Python at run | PJRT plugin, XLA, MLIR static libs |
| 3 | nanobind, jax FFI, per-oracle C++ build | PJRT plugin, XLA, MLIR static libs |
| 4 | nothing more; pins the remaining plugin | PJRT plugin, XLA, MLIR static libs |
| 5 | PJRT, XLA, the plugin, MLIR if the text is bypassed | a C++ interpreter owned by the project |

## Gaps the batch path opens

- `fsim-batch` refuses `dynamic_slice`, `concatenate`, `while` and `custom_call` (`fsim/passes/batch.cc:306-324`), each of which the oracle emits today, so step 1's emitter should prefer `slice` with constant starts, which the unrolled loops make possible (`utils.py:805-808`, `README.md:40`).
- A data-dependent read (`debug`) forces one member (`fsim/src/pipeline.cc:845-855`), so batching only pays for kernels without prints.
- The `taidl` variant runs one module at the end of the trace (`fsim/README.md:56-59`), which matches the oracle's one-module-per-kernel shape today (`decorator.py:200-211`).
