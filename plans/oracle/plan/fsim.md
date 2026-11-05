# act-oracle on fsim-lib: native-path steps 1 to 4

Owner decision 23 moves the oracle's execution to fsim-lib; O29 plans that move as O34 to O36, on U15's pin set.
The items follow steps 1 to 4 of `../fsim/native-path.md`; step 5, a StableHLO interpreter, stays out under owner decision 28.
The oracle's per-instruction StableHLO comes from O3's generic emitter alone, so the oracle never waits on the backend dialect (B50).

| ID | Item | Depends on | Size (S/M/L) | Quarter |
| --- | --- | --- | --- | --- |
| O34 | fsim step 1: C++ emitter per instruction from O3's generic emitter | O29, O3, U15 | L | later |
| O35 | fsim step 2: `Pipeline::submit`, `end_trace`, `sync` replace `compile_and_load` and `execute` | O34 | M | later |
| O36 | fsim steps 3 and 4: debug print out of `custom_call`, PJRT pinned by header and plugin hash; every golden and norm row holds | O35, U15 | M | later |

## O34 C++ emitter per instruction

- Breaking: no.
- Changes: `taidl_to/` gains a C++ target; preserves the Python oracle until O36 passes.
- What: the generator emits, per instruction, C++ that writes the same StableHLO as O3's generic emitter, so the kernel's module is built without the Python trace.
- What: the emitter prefers `slice` with constant starts, since the batch path refuses `dynamic_slice`, `concatenate`, and `while`.
- Benefits: Python remains only at generation time.
- Caveats: MLIR and StableHLO static libraries come from U15, at one pinned StableHLO version.
- Fail points: a word whose C++ form differs from O3's text; noticed by a module-text comparison per word against the Python oracle.
- Order and dependencies: after O29, O3, and U15.
- Verification: per instruction of every example ISA specification, the C++ and Python modules are equal in generic form.

## O35 Run through `fsim::Pipeline`

- Breaking: no.
- Changes: the run path of `taidl_to/generic/decorator.py`; preserves the generated oracle's call interface.
- What: `Pipeline::submit`, `end_trace`, and `sync` replace `compile_and_load` and `execute`; state tensors become the pipeline's `Tensor` list with `tf.aliasing_output`.
- Benefits: jax, jaxlib, NumPy, and Python leave the run.
- Caveats: a data-dependent read (`debug`) forces one batch member, so batching pays only for kernels without prints.
- Fail points: an aliased state tensor not written back; noticed by the state-update tests.
- Order and dependencies: after O34.
- Verification: every oracle test passes on the pipeline run; no jax import at run time.

## O36 Debug print and PJRT pin

- Breaking: no.
- Changes: `taidl_to/xla-debug/`, the oracle's build and pin files.
- What: a debug read becomes a `sync` followed by a host-side print, replacing the `custom_call` print handler.
- What: PJRT is the last external dependency, pinned by header and plugin hash from a released jax plugin build.
- Benefits: nanobind, the jax FFI, and the per-oracle C++ build leave; the oracle is C/C++ with one dlopen'd plugin.
- Caveats: XLA inside the plugin still decides the arithmetic, so the numerics norms stay as measured.
- Fail points: a plugin hash mismatch at load; it stops with an error naming both hashes.
- Order and dependencies: after O35 and U15.
- Verification: every golden and norm row of this plan holds on fsim; owner decisions 16, 17, 22, and 24 hold in each fsim pass.
