# fsim-lib and act-oracle: interface and fit

How a generated oracle would sit on fsim-lib, what fsim already replaces, and how owner decisions 16, 17 and 22-24 land on it.
Paths start at the fsim-lib repo root; `act-oracle/` and `plans/` name the other trees.

## The interface a consumer sees

- A consumer adds `fsim/` with `add_subdirectory` and links `fsim` (`fsim/README.md:16`).
- `Runtime()` loads the build-time plugin and creates the client; it throws on failure (`fsim/include/fsim/runtime.h:12-13`).
- `Pipeline(runtime, state)` takes the state as `std::vector<Tensor>` and uploads it (`fsim/include/fsim/pipeline.h:57-58`).
- `soft_trigger(ops)`, `submit(text)`, `sync()` and `stats()` drive a run (`fsim/include/fsim/pipeline.h:64-71`).
- `begin_run`, `end_trace` and `summary` print the `fsim stats:` line (`fsim/include/fsim/pipeline.h:73-78`).
- `Knobs::parse` consumes `--trigger` and `--overlap` from argv (`fsim/include/fsim/pipeline.h:49-52`).
- There is no Python binding; the API is C++ only.

The module contract:

- `@main` takes every state tensor in order and returns them all in order (`fsim/include/fsim/pipeline.h:2`).
- The results must match the state in count, element type and dims (`fsim/src/pipeline.cc:325-358`).
- The text is StableHLO in generic or pretty form that MLIR parses into `func` and `stablehlo` (`fsim/passes/passes.cc:79-83`).
- State element types are the fourteen of `ElementType` (`fsim/include/fsim/tensor.h:11-26`).

## What fsim takes from TAIDL or TAIDL-TO

- Nothing directly: fsim reads no TAIDL file and no generated `api.py`.
- A driver supplies the state declaration and the module text.
- TAIDL's data models map to state tensors, as the NKI driver maps `hbm`, `sbuf`, `psum` and `reduce_regs` (`drivers/nki/design.md:234-244`).
- `hbm` plays TAIDL's `d0` as a flat byte buffer (`drivers/nki/design.md:236`).
- Each instruction is one `semantics` call that reads its views, computes, and writes its destination view (`drivers/nki/design.md:249`).
- Reads and writes lower to slices, reshapes, bitcasts and `dynamic_update_slice` over byte grids (`drivers/nki/design.md:253-265`).

## How a generated oracle would use it

Today the oracle traces a whole kernel in Python into one module and runs it through jax:

- `generate_module` emits `@main` over the inputs and constants, returning the outputs (`act-oracle/taidl_to/templates/templates.txt:39-59`).
- `compile_module` calls `backend.compile_and_load` (`act-oracle/taidl_to/generic/decorator.py:47-61`).
- Arguments pass through `jax.device_put`, then `executable.execute` (`act-oracle/taidl_to/generic/decorator.py:188-191`).

On fsim the same kernel would run as follows:

1. Declare one state tensor per data model, `d0` as `ui8[hbm]` and each on-chip buffer as bytes or its declared type.
2. Write each input and constant into `d0`'s host memory at its `addr`, as `check_declaration` already lays them (`act-oracle/taidl_to/generic/decorator.py:64-139`).
3. Emit per-instruction StableHLO into a module whose `@main` threads the state, not the inputs and outputs.
4. Submit at the soft trigger, or once at the end under the `taidl` variant (`fsim/src/pipeline.cc:448-454`).
5. Sync, then read each output from `d0`'s host memory at its `addr`.

- The `taidl` variant reproduces today's one-module run, so it is the A/B baseline against `overlap`.
- Step 3 changes the shape of `@main` from inputs-to-outputs to state-to-state; the lowering of each word does not change.

## What fsim already replaces in act-oracle

| act-oracle today | fsim counterpart | Source |
| --- | --- | --- |
| `set_simulation_backend`, jax platform choice | `FSIM_COMPILER` at build time | `act-oracle/taidl_to/generic/decorator.py:18-35`, `fsim/README.md:26-35` |
| `jax_enable_x64`, `jax_default_matmul_precision` globals | none; no jax in the process | `act-oracle/taidl_to/generic/decorator.py:22-23` |
| `get_compile_options`, `compile_and_load` | `compile_options()`, `PJRT_Client_Compile` | `act-oracle/taidl_to/generic/decorator.py:54-58`, `fsim/src/pipeline.cc:298-322` |
| `jax.device_put`, `execute`, `block_until_ready` | state upload, `execute`, `sync` | `act-oracle/taidl_to/generic/decorator.py:188-192`, `fsim/src/pipeline.cc:153-169` |
| compile-time measurement | `Stats`, `fsim stats:` line | `act-oracle/taidl_to/generic/decorator.py:57-59`, `fsim/include/fsim/pipeline.h:22-42` |
| one module per kernel | `taidl` variant, or many modules under `trigger` and `overlap` | `fsim/README.md:55-60` |

## What fsim does not replace

| act-oracle part | Why it stays | Source |
| --- | --- | --- |
| TAIDL-to-StableHLO lowering | fsim holds no ISA semantics | `act-oracle/taidl_to/oracle_visitor.py`, `act-oracle/taidl_to/instruction_codegen.py` |
| Python kernel API and generated `api.py` | fsim has no Python entry; NKI uses a separate Python recorder | `drivers/nki/record/record.py` |
| Declaration and call-time checks | a driver's job; NKI does its own in `Limits.cpp` | `act-oracle/taidl_to/generic/decorator.py:64-139`, `drivers/nki/design.md:303-316` |
| Shape module (decision 25) | fsim checks only result shapes against state | `fsim/src/pipeline.cc:337-356` |
| Debug print handler | fsim registers no FFI target, so a `print_handler` custom call has no target | `act-oracle/taidl_to/templates/templates.txt:290`, `act-oracle/taidl_to/generic/decorator.py:49-52` |
| Numerics norms and fixed data values | emitted in the module by the generator | `plans/oracle/plan/lowering.md`, `plans/oracle/plan/fidelity.md` |
| Sub-byte and `pred` element types | fsim's `ElementType` lacks `u2`, `u4`, `s2`, `s4`, `pred` | `act-oracle/taidl_to/generic/utils.py:851-871`, `fsim/include/fsim/tensor.h:11-26` |

- Sub-byte types matter only for a state tensor; inside a module any StableHLO type is legal.
- A data model of a sub-byte type would be stored as `ui8` and bitcast in the module, as NKI does for every dtype (`drivers/nki/design.md:298`).

## Decision 16: no undefined behavior, faults detected and reported

| Fault class | fsim mechanism | What the oracle must add |
| --- | --- | --- |
| Malformed module, failed pass, failed compile | throw with PJRT or pass diagnostics (`fsim/src/runtime.cc:87-101`, `fsim/README.md:121`) | map the module back to the instructions it holds |
| Result not matching state | refused at compile (`fsim/src/pipeline.cc:337-356`) | nothing |
| Addressing or region violation | none in the library | trace-time checks on concrete values, as O24 plans and NKI does (`drivers/nki/design.md:303-316`) |
| Data-dependent element fault | none in the library | nothing: tensor data never faults, and the norm table's fixed values (O23) cover the cases |
| Value StableHLO leaves open | none; XLA's value | the norm table's guards in the emitted module |

- Library errors name a PJRT call, never an instruction, so the driver owns the instruction-level message.
- `@main` must return exactly the state, and the module needs no error flag, since tensor data never faults.
- A sticky error stops the pipeline, so no later module runs after a fault (`fsim/src/pipeline.cc:461-467`).
- No race is modelled: execution is one program order (`drivers/nki/design.md:379`).

## Decisions 17, 22-24: in-place marks

- fsim's in-place is buffer-level: state argument i aliases result i so XLA reuses the device buffer (`fsim/passes/passes.cc:101-110`).
- That aliasing is a performance property of the whole state and does not express an instruction's mark.
- Under `xla-` compilers there is no aliasing attribute; the old buffers are destroyed after execute instead (`fsim/src/pipeline.cc:391-394`).
- In a byte-arena state, decision 17 is a write of the output into the marked input's region of the same buffer.
- The pre-state rule holds in the module: a semantics function that reads every input before its write gives one defined value for `r0 = r1 + r0` (panel 9).
- The NKI driver reads its views, computes, then writes, the same order act-oracle uses (`drivers/nki/design.md:249`).
- Decision 24 is a static check in the generator or front end, since a cross-buffer pair names two state tensors; fsim needs no change.
- The overlap constraints stay call-time checks on concrete windows, which a driver computes before emitting any load (`plans/meeting/panel-9/decision.md`).
- In fsim two allocations at the same address alias by construction (`drivers/nki/design.md:294`), so overlap is real storage, not an emission accident.
