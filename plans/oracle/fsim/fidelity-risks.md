# Fidelity risks of the fsim passes against the owner decisions

Where each fsim pass, or the runtime, can change a result the oracle must hold, and what test catches it.
Decisions are quoted from `plans/minutes/owner-decisions.md` by number.
Paths under `fsim/` are the upload; paths under `taidl_to/` are the current oracle.

## The decisions in play

| no. | decision, in short | what fsim must not do |
| --- | ------------------ | --------------------- |
| 16 | no undefined behavior; a fault is detected and reported, and writes no region and no state | clamp, wrap or drop a faulting access silently |
| 17 | the output reuses the marked input's storage | copy where the mark says reuse, or alias where it does not |
| 22 | overlap alone does not settle read-after-write; `set_inplace` stays | reorder a read past the write it must precede, or vice versa |
| 24 | an in-place mark across different buffers is rejected | alias two state tensors |

## Risks, pass by pass

| source | risk | decision | where in the code |
| ------ | ---- | -------- | ----------------- |
| `Window::of(DynamicUpdateSliceOp)` | an out-of-range constant start is clamped so the update fits, as the op defines it; a store past the buffer's end lands at the end instead of faulting | 16 | `fsim/passes/window.cc:154-155` |
| `fsim-forward-slices` | a slice whose window relates `Inside` an update reads the update's tile; if the tile was reshaped through `rebox`, a wrong box reads the wrong bytes | 22 | `fsim/passes/forward_slices.cc:43-65`, `:80-99` |
| `fsim-forward-slices` | a slice past a `Disjoint` update reads the update's operand, which is correct only if no other write to the same bytes sits between them in the chain | 22 | `fsim/passes/forward_slices.cc:160-164` |
| `fsim-dead-updates` | an update covered by later updates is erased; a read of the state between the two updates, outside the single-use chain, would make it live, and the pass trusts `hasOneUse` for that | 22 | `fsim/passes/dead_updates.cc:16-28`, `:46-48` |
| `fsim-dead-updates` | a faulting update erased as dead never reports, so the fault vanishes | 16 | `fsim/passes/dead_updates.cc:46-48` |
| `fsim-order-reads` | a read after the update is hoisted before it with its producers; the hoist is refused only when it needs the update, a region or a side effect, so a read that must see the update through another view is the trap | 22 | `fsim/passes/order_reads.cc:44-70`, `:100-116` |
| `fsim-order-reads` | the barrier chain pins XLA to in-place updates, which is the aliasing decision 17 wants, but the result is now layout-sensitive to XLA's copy insertion flag | 17 | `fsim/passes/order_reads.cc:122-133`, `fsim/src/compile_options.cc:56-59` |
| `tf.aliasing_output` | every `@main` argument aliases its own result index, one to one, so a cross-buffer alias cannot arise from the stage; the oracle's emitter must keep one result per state tensor | 24 | `fsim/passes/passes.cc:105-109`, `fsim/src/pipeline.cc:591-630` |
| `fsim-byte-grids` | a `[P, M, K]` update rewrites to `[P, M*K]` with the column index scaled by K; a non-constant column index becomes a `multiply`, which overflows silently in its index type | 16 | `fsim/passes/byte_grids.cc:159-171` |
| `fsim-batch` | a batched reduce or dot runs under XLA's batched kernels, whose accumulation order can differ from the one-member kernel's | float rounding | `fsim/passes/batch.cc:213-237` |
| `fsim-batch` | a `select` with a scalar predicate stays unbatched, so a per-member predicate that the emitter wrote as a scalar applies member 0's choice to all | 22 | `fsim/passes/batch.cc:91-93` |
| agreement check | compares members bitwise and fails the sync, which is the right default; but it runs only with `sets = 1`, so a `sets = B*` run has no determinism check | float rounding | `fsim/src/pipeline.cc:184-197`, `:679-699` |
| XLA options | `xla_backend_optimization_level=1` and the GPU options are per compiler, so CPU and GPU results may differ in rounding, as the oracle already documents for `sqrt` and `triangular_solve` | float rounding | `fsim/src/compile_options.cc:55-62`, `taidl_to/README.md:87-88`, `:151` |
| integer semantics | the passes never change integer ops, so wrap stays XLA's; `power` reads the low six bits of the exponent and loop steps wrap at i64 | integer overflow | `taidl_to/docs/operations.md:105`, `taidl_to/README.md:141` |

## Read-before-write within an instruction

- The oracle emits a store as `dynamic_update_slice` of the buffer value after its loads are `dynamic_slice`s of the same value (`taidl_to/generic/utils.py:813-843`), so SSA order already encodes read-before-write.
- `fsim-order-reads` preserves that order and adds a barrier so XLA does too (`fsim/passes/order_reads.cc:86-143`).
- The risk is a read emitted after the store in SSA, which the pass hoists above it (`order_reads.cc:100-116`); decision 22 says that case is the instruction's own choice, so the emitter must not rely on the pass to pick.
- A second update of the same buffer in one block makes the pass leave the buffer as is (`order_reads.cc:31-34`), so XLA's own analysis decides, which is the public behavior today.

## Faults and decision 16

- Today the oracle refuses an out-of-extent window at trace time, per unrolled iteration (`taidl_to/generic/utils.py:698-712`), so no faulting access reaches StableHLO.
- Under fsim with dynamic starts, `dynamic_update_slice` clamps by definition (`fsim/passes/window.cc:154`), which is a defined behavior but not a reported fault.
- To hold decision 16 the driver must check bounds before `submit`, as the oracle does now, or emit an explicit compare and a `sync` that fails the run; the library has no fault channel other than a failed `sync` or an `fsim:` throw.
- A fault that writes no state is met by the pipeline's failure path: a failed module keeps the pipeline's error and `sync` rethrows it before copying to host (`fsim/src/pipeline.cc:834-866`).

## Tests that would catch each risk

| risk | test |
| ---- | ---- |
| clamped store | a kernel whose last iteration writes one element past the region; expect a refusal, not a clamped write |
| forward-slices rebox | a load of a sub-tile of a just-stored tile through a reshaped byte grid, compared element by element with jax |
| disjoint hop | store A, store B over part of A's bytes, load A's window; compare with jax |
| dead update erased | store A, load A, store B covering A; the load must see A |
| dead faulting update | store past the region then overwrite it fully; expect the fault |
| order-reads hoist | `r0 = r1 + r0` style instruction with the read after the write in SSA; compare both orders with the TAIDL semantics |
| cross-buffer alias | a module returning state tensor 1 in result slot 0; expect `check_outputs` to fail |
| batch accumulation | the same kernel at `--batch 1` and `--batch 16` with `sets = 1`; the agreement check must pass bitwise, and f32 sums compare to jax within the `operations.md` bounds |
| scalar select | a per-member predicate kernel at `--batch 2` with two differing sets; outputs must differ |
| CPU vs GPU rounding | the existing `operations.md:139-153` bounds rerun through fsim on both compilers |
| index multiply overflow | a byte-grid update with a column index near `2^63 / K`; expect a refusal or a documented wrap |
