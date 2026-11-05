# The latest fsim upload over the public fsim-lib

What the uploaded `fsim/` folder adds to the public fsim-lib clone's `fsim/`.
Paths are relative to the upload's `fsim/`, and `public:` marks the clone.

## What is new, file by file

| file | state | what it adds |
| ---- | ----- | ------------ |
| `passes/order_reads.cc` | new | the `fsim-order-reads` pass |
| `passes/batch.cc` | new | the `fsim-batch` pass |
| `src/batch.cc`, `include/fsim/batch.h` | new | the `fsim::Batch` loop over a driver's inputs |
| `src/pipeline.cc` | changed | members, host sets, replay, restart, device-side broadcast and agreement check |
| `src/runtime.cc` | changed | `bytes_limit()`, `host_bytes()`, GPU `memory_fraction` 0.95 |
| `passes/passes.cc`, `passes.h`, `stage.h` | changed | a `batch` phase after `library`; `fsim_passes_run` takes a `batch` count |
| `passes/tool.cc` | changed | `--batch B` on `fsim-passes` |
| `include/fsim/pipeline.h` | changed | `Knobs::batch`, `Pipeline::chunk`, `replay`, a `sets` constructor argument |
| `src/tensor.cc`, `include/fsim/tensor.h` | changed | `element_size` becomes public |

- `byte_grids.cc`, `forward_slices.cc`, `dead_updates.cc`, `window.*`, `compile_options.*` and `setup.sh` are byte-identical to the public clone.
- The pass order gains one step: `fsim-order-reads` runs last, after the second `fsim-byte-grids`, canonicalize and CSE (`passes/fsim_passes.cc:19-20`).
- The pass stage then aliases every `@main` argument to its result with `tf.aliasing_output` (`passes/passes.cc:105-109`) and, above one member, runs `fsim-batch` (`passes/passes.cc:113-119`).

## The passes, new and inherited

| pass | what it does to the program | why it is faster | what it assumes |
| ---- | -------------------------- | ---------------- | --------------- |
| `fsim-byte-grids` (`passes/byte_grids.cc`) | sinks a bitcast between a `[..., K]` byte grid and K-byte elements below slices, reshapes and updates; a 2-D byte tensor's `[P, M, K]` update runs in `[P, M*K]` | the state buffer keeps one byte shape, so updates alias it without a relayout | a trailing byte dim of width K (`byte_grids.cc:24-43`); the update's byte-dim start is a literal zero (`byte_grids.cc:152-153`) |
| `fsim-forward-slices` (`passes/forward_slices.cc`) | each slice walks its update chain: past a disjoint update, onto an equal or inside tile as a slice or reshape of the tile, or onto the chain's head | a read of freshly written bytes never materializes the whole buffer | constant start indices (`window.cc:147-150`); windows under 4096 runs (`window.cc:13`, `:194`); at most 2048 hops (`forward_slices.cc:22`) |
| `fsim-dead-updates` (`passes/dead_updates.cc`) | from each returned value, up its chain of single-use updates, drops an update whose bytes the later updates cover | fewer updates to execute and fewer buffers alive | the chain is single-use (`dead_updates.cc:16-28`); byte coverage is exact (`window.cc:240-249`) |
| `fsim-order-reads` (`passes/order_reads.cc`) | before an update, threads the buffer and every read of it through `stablehlo.optimization_barrier`, hoisting a read that comes after the update, with what it needs, before it | XLA updates the buffer in place with no defensive copy, and a read stays fused into its consumer (`order_reads.cc:1-4`) | reads are in the update's block, not a return and not a second update (`order_reads.cc:31-34`); a hoisted op has no region and no memory effect (`order_reads.cc:58`); only the entry block's updates (`order_reads.cc:152-154`) |
| `fsim-batch` (`passes/batch.cc`) | gives every `@main` argument, result and ranked value a leading dim B, vmap-style; constants and elementwise ops over unbatched operands stay unbatched until a batched op uses them | B inputs run as one executable, amortizing compile and launch | one one-block `@main` of ranked tensors (`batch.cc:352-380`); an op has a `kRules` entry or the Elementwise trait, else refused by name (`batch.cc:306-342`, `:393-396`); a 0-d index, init or pad value is unbatched (`batch.cc:165-173`) |

- `fsim-batch` covers 17 named ops plus every elementwise StableHLO op (`passes/batch.cc:306-324`).
- Missing from `kRules`: `dynamic_slice`, `concatenate`, `gather`, `scatter`, `while`, `custom_call`, `convert` outside the trait, `dot` without `_general`, `sort`, `triangular_solve`, `cholesky`.
- The pass fails the module's compile for any of those, so a module the oracle emits today, which uses `dynamic_slice` (`taidl_to/generic/utils.py:822`), `concatenate` (`utils.py:958`), `while` (`taidl_to/template.py:222`) and `custom_call` (`template.py:290`), does not batch as is.

## The runtime and PJRT use

- The runtime `dlopen`s the baked-in PJRT plugin and resolves `GetPjrtApi` (`src/runtime.cc:41-46`).
- Every device step is a PJRT C API call: `PJRT_Client_BufferFromHostBuffer` (`src/pipeline.cc:406-415`), `PJRT_Client_Compile` (`:580-586`), `PJRT_LoadedExecutable_Execute` (`:641-650`), `PJRT_Buffer_ToHostBuffer` (`:670-675`).
- New: `bytes_limit()` asks `PJRT_Device_MemoryStats`, treats `UNIMPLEMENTED` as no limit, and on the CPU falls back to host RAM capped by the cgroup's `memory.max` (`src/runtime.cc:17-27`, `:154-188`).
- New: a `custom-gpu` client reserves 95% of free GPU memory through the `memory_fraction` option (`src/runtime.cc:81-92`).
- The compile options are a hand-rolled protobuf of `CompileOptionsProto` with the per-compiler XLA overrides (`src/compile_options.cc:1-5`, `:47-62`).
- The library emits StableHLO text for its own helper modules: broadcast, agreement and member-0 slice (`src/pipeline.cc:146-243`), so even the batching plumbing is XLA-compiled.

## Batching

- `--batch B` is a run-time knob; a non-`custom-` build refuses it above 1 (`src/pipeline.cc:744-749`).
- `B*`, the chunk, is `min(B, 16)` on the CPU and B on the GPU, lowered to the largest power of two whose `2 x B* x state bytes` fits the device limit (`src/pipeline.cc:257-272`).
- Above one member each state tensor lives on the device as `[B*, dims]` (`src/pipeline.cc:369-377`), and `check_outputs` checks the module returns those device dims (`:591-630`).
- With `sets = B*` each member uploads from and syncs to its own host set; with `sets = 1` the device broadcasts one set (`broadcast_module`, `:146-160`) and `sync` runs the agreement module, which fails on the first member whose raw bits differ from member 0 (`:164-220`, `:679-699`).
- A `sync` before `end_trace` makes the run data-dependent: `drop()` slices member 0 out on the device, frees the kept executables and continues at one member (`src/pipeline.cc:701-727`, `:845-855`).
- At B above B* the pipeline keeps every executed module and `replay()` re-uploads the host sets and reruns them in order on the caller's thread (`src/pipeline.cc:505-511`, `:867-895`).
- `fsim::Batch` draws B iid sets from a fixed-seed `std::mt19937_64`, restores members, traces the first chunk, replays the rest, and stores each member's outputs (`src/batch.cc:25-37`, `:41-67`); a driver implements `Batch::Driver` (`include/fsim/batch.h:22-38`).
- The summary line gains `inputs` and `chunk` (`src/pipeline.cc:939-947`).
- The agreement check compares members bitwise (`src/pipeline.cc:184-197`), so a replicated batch doubles as a determinism check of XLA at B* members.
