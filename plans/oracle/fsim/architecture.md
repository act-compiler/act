# fsim-lib architecture

What fsim-lib is, how data flows through it, and how fast and how exact it is, from its public sources.
Paths start at the fsim-lib repo root; `act-oracle/` and `plans/` name the other trees.

## What it is

- A "scalable functional simulator for AI accelerators", shared by ISA drivers that hand it their tensor instructions (`README.md:3-4`).
- It "builds StableHLO lazily and compiles and runs it through XLA's PJRT C API" (`README.md:5`).
- The library is a static C++ library over a fixed set of state tensors (`fsim/README.md:3`).
- It holds no ISA semantics: a driver emits every instruction as StableHLO text, and XLA computes it.
- One driver exists today, `fsim-nki` for AWS NKI on trn2 (`drivers/nki/README.md:3-5`).
- The NKI driver follows TAIDL's model of an ISA and cites act-oracle as its pattern (`drivers/nki/design.md:3`, `drivers/nki/design.md:8-9`).
- "All C/C++" means no Python in the run loop: the C++ host dlopens a PJRT plugin, the one jax 0.11.2 ships (`fsim/src/runtime.cc:23`, `fsim/README.md:26-33`).
- XLA stays the numerics engine; act-oracle pins the same jax 0.11.2 (`act-oracle/pyproject.toml:18-19`).

## Layers

| Layer | Path | Role |
| --- | --- | --- |
| Public API | `fsim/include/fsim/{runtime,tensor,pipeline}.h` | `Runtime`, `Tensor`, `Pipeline`, `Knobs`, `Stats` |
| Runtime | `fsim/src/runtime.cc` | plugin load, version check, one client, one device |
| Pipeline | `fsim/src/pipeline.cc` | state upload, module queue, compile workers, executor, sync |
| Compile options | `fsim/src/compile_options.cc` | hand-encoded `CompileOptionsProto` with per-compiler XLA overrides |
| Pass stage | `fsim/passes/` | MLIR passes over each module before PJRT compile, in a shared object |
| Driver | `drivers/nki/src/` | trace parser, accelerator state, StableHLO emitter, per-instruction semantics |
| Recorder | `drivers/nki/record/*.py` | runs a kernel under NKI's frontend and writes the text trace |

- No PJRT type leaves `fsim/src/` (`fsim/README.md:9`).
- The pass stage exports only a C entry, so its static MLIR never meets a driver's LLVM (`fsim/README.md:104`, `fsim/passes/passes.h:15`).

## Data flow

1. A driver builds `Tensor`s, each an element type, dims and caller-owned host memory (`fsim/include/fsim/tensor.h:29-33`).
2. `Pipeline(runtime, state)` uploads one device buffer per state tensor (`fsim/src/pipeline.cc:153-169`).
3. The driver appends ops to a module and asks `soft_trigger(ops)` before each instruction (`drivers/nki/src/Driver.cpp:96-97`).
4. `submit(text)` queues the module; `@main` takes every state tensor in order and returns them all in order (`fsim/include/fsim/pipeline.h:1-4`).
5. Compile runs the pass stage under a `custom-` compiler, then `PJRT_Client_Compile` on format `mlir` (`fsim/src/pipeline.cc:298-322`).
6. The compiled results must match the state in count, type and dims, or compile throws (`fsim/src/pipeline.cc:325-358`).
7. Execute passes the state buffers and the results become the new state (`fsim/src/pipeline.cc:366-395`).
8. `sync()` waits for every module, then copies each state buffer back to its host memory (`fsim/src/pipeline.cc:471-495`).

- State stays on the device between modules; only `sync()` copies it to the host.
- A driver syncs mid-run only to read a run-time value, as NKI's `+s(...)` offsets do (`drivers/nki/src/Driver.cpp:104-109`).

## Build-time choices

| Choice | Values | Effect | Source |
| --- | --- | --- | --- |
| `FSIM_COMPILER` | `xla-0.11-cpu`, `xla-0.4-cpu`, `custom-cpu`, three `-gpu` | PJRT entry, device, pass stage on or off | `fsim/README.md:26-35` |
| `FSIM_VARIANT` | `eager`, `taidl`, `trigger`, `overlap` | when modules flush, where they compile | `fsim/README.md:55-60` |
| `--trigger N` | 10000 | soft trigger op count | `fsim/src/pipeline.cc:432-441` |
| `--overlap T` | 4 | compile threads | `fsim/src/pipeline.cc:170-175` |

- `taidl` runs one module at the end and throws on a second, the act-oracle shape (`fsim/src/pipeline.cc:448-454`).
- `overlap` compiles on worker threads in any order; one executor runs modules strictly in submission order (`fsim/src/pipeline.cc:199-278`).
- CPU builds create a client over one device; every build runs on the first addressable device (`fsim/src/runtime.cc:52-61`, `fsim/src/runtime.cc:70-73`).
- Setup runs on x86_64 only; a CPU plugin built from source takes 1-2 h (`fsim/README.md:142-146`).

## Pass stage

- Parse into `func` and `stablehlo`, then CSE, StableHLO target-independent optimization, canonicalize, CSE (`fsim/passes/passes.cc:85-93`).
- Library passes in order (`fsim/passes/fsim_passes.cc:8-19`):
  - `fsim-byte-grids`: sinks byte-to-element bitcasts below slices and updates (`fsim/passes/byte_grids.cc:1-5`)
  - `fsim-forward-slices`: forwards a slice past updates it misses, or onto the update holding its window (`fsim/passes/forward_slices.cc:1-4`)
  - `fsim-dead-updates`: drops updates that later updates overwrite (`fsim/passes/dead_updates.cc:1-3`)
- Every `@main` argument gets `tf.aliasing_output` to its own index, so state updates in place (`fsim/passes/passes.cc:101-110`).
- The module serializes as a portable artifact at StableHLO 1.20.0 (`fsim/passes/passes.cc:28`, `fsim/passes/passes.cc:113-118`).
- A failed phase fails the compile with its diagnostics (`fsim/README.md:121`).
- The `xla-` compilers skip the stage, so they get neither the rewrites nor the aliasing attribute (`fsim/src/pipeline.cc:299-301`).

## Performance and scalability

Stated:

- "Scalable" and "under review at a major systems conference" (`README.md:3`, `README.md:7`).
- The nki.simulate wall-time table "is the baseline the driver is compared against" (`docs/nki-simulate.md:122-145`).
- That baseline ranges from 0.00 s to 108.06 s (`mlp_cte seq8192`) on a 16-core EPYC (`docs/nki-simulate.md:119-141`).

Evident from code:

- Lazy modules bounded by the soft trigger keep each XLA compile small, whatever the trace length.
- Compile overlaps tracing and execution on threads (`fsim/src/pipeline.cc:170-175`).
- In-place aliasing avoids copying whole arenas per module; the arenas are large (`drivers/nki/design.md:239-244`).
- The custom passes shrink chains of slices and updates before XLA sees them.
- Stats split wall into simulation, compile overhead and tracing overhead (`README.md:116-127`).

Not in the repo:

- No fsim timing table is checked in; the only figure is one `rmsnorm_tkg` stats line, 0.073 s wall over 11 instructions (`drivers/nki/design.md:49-52`).
- A per-case wall limit is 60 s by default, and CI raises it to 300 s (`README.md:106`, `.github/workflows/test.yml:31-33`).
- So "much faster" is the owner's statement; the speedup over the JAX oracle on TAIDL kernels is unmeasured here.

## Numerics

- The library computes nothing itself: every value is XLA's result for the driver's StableHLO.
- CPU compilers force `xla_backend_optimization_level=1` and region-analysis copy insertion (`fsim/src/compile_options.cc:54-63`).
- GPU compilers set autotune level 0 and Triton GEMM off (`fsim/src/compile_options.cc:54-63`).
- No compiler sets `xla_allow_excess_precision`, so XLA's default holds (`fsim/src/compile_options.cc:54-63`).
- The overrides travel per compile in `env_option_overrides` (`fsim/src/compile_options.cc:75-83`).
- Rounding and accumulation are the driver's choices in the emitted module.
- The NKI driver computes in f32 over converts of the operands (`drivers/nki/design.md:261`).
- Its `dot_general` carries `HIGHEST` precision with an f32 result (`drivers/nki/src/stablehlo/Module.cpp:516-521`).
- f32 to integer converts round to nearest even, saturate, and map NaN to the maximum (`drivers/nki/src/stablehlo/Module.cpp:534-545`).
- Accumulation order inside a `dot_general` or `reduce` is XLA's, not stated by fsim.
- Transcendentals are XLA's f32 ops (`drivers/nki/design.md:332`).
- The stock target-independent optimization may fold or rewrite ops before XLA, a pass the JAX oracle does not run (`fsim/passes/passes.cc:88-89`).

## Errors and faults

- Every PJRT error throws `std::runtime_error` with the PJRT message and the call's name (`fsim/src/runtime.cc:87-101`).
- A failed submit or execute becomes the pipeline's sticky error, rethrown by the next submit or sync (`fsim/src/pipeline.cc:443-447`, `fsim/src/pipeline.cc:461-467`).
- A module whose results differ from the state is refused at compile (`fsim/src/pipeline.cc:337-356`).
- A bad knob refuses the command line and exits 1 (`fsim/src/pipeline.cc:411-421`).
- Data-dependent faults are the driver's job; the library has no error-flag channel.
- The NKI driver checks memory limits per allocation and per `nc_matmul` at trace time, naming the record's line (`drivers/nki/design.md:303-316`).
- It refuses an unknown kind, a bad record or a refused view loudly (`drivers/nki/design.md:47`).
