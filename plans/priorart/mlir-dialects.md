# MLIR accelerator-facing dialects and IREE microkernels

## What it is

- ArmSME models ZA tiles as SSA values of type `vector<[4]x[4]xf32>`: `arm_sme.get_tile` (undefined contents), `arm_sme.zero`, `arm_sme.outerproduct` with `acc` and operand masks, `fmopa_2way` and `smopa_4way` widening ops, and an `arm_sme.intr.*` layer that takes a `tile_id`; verifiers pin operand widths and target features such as `+sme2`.
- NVGPU bridges vector and memref to PTX: `nvgpu.mma.sync` with `mmaShape = [16, 8, 16]` over per-thread fragments, `nvgpu.ldmatrix`, TMA descriptors, `warpgroup.mma` with `!nvgpu.warpgroup.descriptor` and `!nvgpu.warpgroup.accumulator` types, async tokens `!nvgpu.device.async.token` and `!nvgpu.mbarrier.group` as typed state.
- X86Vector and AMX wrap intrinsics in two layers, a typed op and an `intr` op, with the same pattern as ArmSME.
- IREE microkernels lower `linalg.mmt4d` to `iree_codegen.ukernel.generic "iree_uk_mmt4d"`, a memory-to-memory call with a params struct, compiled from freestanding C to bitcode and always inlined; layout (pack and unpack) stays in codegen.
- Sources: https://mlir.llvm.org/docs/Dialects/ArmSME/, https://mlir.llvm.org/docs/Dialects/NVGPU/, https://mlir.llvm.org/docs/Dialects/X86Vector/, https://iree.dev/community/blog/2024-01-22-exploring-cpu-microkernels-on-a-matmul-example/.

## TAIDL today, for comparison

- A TAIDL data model is a named array with access dimensions and a base element shape (taidl/docs/instructions.md S5.2); ArmSME's tile and NVGPU's fragment are data models with a fixed base element shape.
- TAIDL has no asynchronous or token-typed state; NVGPU's `mbarrier` and async tokens and TMA descriptors are state the owner's new scope (mma, wgmma, TMA) needs.
- The backend models no state at all (act-backend/docs/coverage.md, state row), while the oracle unrolls updates in order (act-oracle/docs/coverage.md, S5.4 row).

## Borrow

- Two layers with one source: the typed op and the intrinsic op are generated from one definition; TAIDL's instruction with its PII-graph node is the same split and should stay generated from one `add_instruction`.
- Undefined-contents constructor: `arm_sme.get_tile` names "contents undefined" as a value; TAIDL states the defined value of an unwritten region.
- Typed hardware state: descriptors, barriers and accumulators as declared state kinds with their own rules, the way TAIDL has `d0` as a special data model; a GPU case study needs a state kind for a descriptor and for an async group.
- Target-feature predicates: `+sme2` gating is an instruction constraint over a configuration state; TAIDL constraints over `@s.` can express it if the state is declared read-only.
- Microkernel boundary: IREE keeps layout in codegen and the inner tile in a kernel; ACT's allocator and PII graph play codegen while an instruction plays the kernel; keep pack and unpack as IR rewrites, not as instruction semantics.

## Avoid

- SIMT thread-level fragments: `nvgpu.mma.sync` operands are per-thread slices, which is the SIMT model the owner excludes; model `mma` and `wgmma` at warpgroup granularity with the fragment as a unit shape.
- Verifier rules written per op in C++: MLIR's constraints live in code; TAIDL's S8 tables should stay the single source and generate the checks.

## Relevance

- High given the GPU tensor-core scope; medium otherwise.
