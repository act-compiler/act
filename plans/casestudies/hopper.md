# NVIDIA Hopper tensor cores (mma, wgmma, TMA)

Sources:
- PTX ISA, `mma.sync`, `ldmatrix`, `wgmma.mma_async`, `cp.async.bulk.tensor`, `mbarrier`: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

This sketch is the GPU tensor-core description; it feeds the GPU requirements.
It models one warpgroup: shared memory as bytes, and the 128 threads' registers as one data model indexed by thread.

```python
from taidl import Accelerator
h = Accelerator('hopper_wg')
# shared memory: 228 KiB as 128-byte lines
h.add_data_model('smem', [1824], [128], 'u8')
# register file of one warpgroup: thread x 32-bit register
h.add_data_model('rf', [128, 255], [], 'u32')
h.add_initial_state('tma_bar', 0)
h.add_initial_state('wg_pending', 0)

# tma_load_2d: global tile -> smem, 128B swizzle, row pitch in bytes
i = h.add_instruction('tma_load_2d', ['rows', 'pitch'], ['g', 's'],
                      update=['@s.tma_bar = @s.tma_bar + 1'])
i.set_inputs([['d0', ['@a.g'], ['@c.rows * @c.pitch']]])
i.set_outputs([['smem', ['@a.s'], ['@c.rows']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.rows * @c.pitch`] parameter(0);
    %a = u8[`@c.rows`,`@c.pitch`] reshape(%In0);
    %t2 = u8[`@c.rows`,128] slice(%a), slice={[0:`@c.rows`], [0:128]};
    %t = u8[`@c.rows`,8,16] reshape(%t2);
    // 128B swizzle: chunk ^= row % 8, built from iota and xor
    %r = s32[`@c.rows`,8] iota(), iota_dimension=0;
    %c = s32[`@c.rows`,8] iota(), iota_dimension=1;
    %eight = s32[`@c.rows`,8] constant(8);
    %rm = s32[`@c.rows`,8] remainder(%r, %eight);
    %sc = s32[`@c.rows`,8] xor(%c, %rm);
    %r3 = s32[`@c.rows`,8,1] reshape(%r);
    %sc3 = s32[`@c.rows`,8,1] reshape(%sc);
    %idx = s32[`@c.rows`,8,2] concatenate(%r3, %sc3), dimensions={2};
    %g = u8[`@c.rows`,8,16] gather(%t, %idx), offset_dims={2}, collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=2, slice_sizes={1,1,16};
    ROOT %Out0 = u8[`@c.rows`,128] reshape(%g);
}
""")

# mbarrier_wait: constraint as wait; faults when false
i = h.add_instruction('mbar_wait', [], [], constraints=['@s.tma_bar >= 1'],
                      update=['@s.tma_bar = @s.tma_bar - 1'])

# wgmma fence, commit, wait: async group as integer state
i = h.add_instruction('wgmma_fence', [], [])
i = h.add_instruction('wgmma_commit_group', [], [], update=['@s.wg_pending = @s.wg_pending + 1'])
i = h.add_instruction('wgmma_wait_group', ['n'], [], constraints=['@s.wg_pending <= @c.n'],
                      update=['@s.wg_pending = 0'])

# wgmma_m64n64k16: D(f32 regs) = scale-d * D + (+-A) * (+-B), A and B in smem bf16
i = h.add_instruction('wgmma_64x64x16', ['scale_d', 'imm_scale_a', 'imm_scale_b', 'imm_trans_a', 'imm_trans_b'],
                      ['a', 'b', 'd'],
                      constraints=['@c.scale_d <= 1', 'abs(@c.imm_scale_a) == 1', 'abs(@c.imm_scale_b) == 1',
                                   '@c.imm_trans_a == 0 and @c.imm_trans_b == 0'])
i.set_inputs([['smem', ['@a.a'], ['16']], ['smem', ['@a.b'], ['16']],
              ['rf', ['0', '@a.d'], ['128', '32']]])
i.set_outputs([['rf', ['0', '@a.d'], ['128', '32']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[64,16,2] parameter(0);
    %A = bf16[64,16] bitcast_convert(%In0);
    %In1 = u8[16,64,2] parameter(1);
    %B = bf16[16,64] bitcast_convert(%In1);
    %In2 = u32[128,32] parameter(2);
    // fragment: thread t, reg r -> row, col of D (affine layout)
    %df = f32[128,32] bitcast_convert(%In2);
    %d6 = f32[4,8,4,8,2,2] reshape(%df);
    %dt = f32[4,2,8,8,4,2] transpose(%d6), dimensions={0,5,1,3,2,4};
    %D = f32[64,64] reshape(%dt);
    %sa = f32[] constant(`@c.imm_scale_a`);
    %sb = f32[] constant(`@c.imm_scale_b`);
    %sd = f32[] constant(`@c.scale_d`);
    %sab = f32[] multiply(%sa, %sb);
    %p0 = f32[64,64] dot(%A, %B), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    %sabb = f32[64,64] broadcast(%sab), dimensions={};
    %p = f32[64,64] multiply(%p0, %sabb);
    %sdb = f32[64,64] broadcast(%sd), dimensions={};
    %Ds = f32[64,64] multiply(%D, %sdb);
    %s = f32[64,64] add(%Ds, %p);
    %s6 = f32[4,2,8,8,4,2] reshape(%s);
    %st = f32[4,8,4,8,2,2] transpose(%s6), dimensions={0,2,4,3,5,1};
    %sf = f32[128,32] reshape(%st);
    ROOT %Out0 = u32[128,32] bitcast_convert(%sf);
}
""")

# st_shared: registers -> smem (one 32-bit word per thread)
i = h.add_instruction('st_shared_b32', [], ['r', 's'])
i.set_inputs([['rf', ['0', '@a.r'], ['128', '1']]])
i.set_outputs([['smem', ['@a.s'], ['4']]])
i.add_semantics("""
ENTRY {
    %In0 = u32[128] parameter(0);
    %b = u8[128,4] bitcast_convert(%In0);
    ROOT %Out0 = u8[4,128] reshape(%b);
}
""")
```

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `tma_load_2d` | expressible with workaround | The tensor map (base, pitch, box, swizzle) is a 128-byte descriptor in memory; it becomes computational attributes; a start can be read from a parameter region only through `gather` or `dynamic_slice` (semantics.md:126). The swizzle is a non-affine index map built with `iota`, `remainder`, `xor`, and `gather` (operations.md:104), because a non-scalar `constant` holds one literal (operation_meaning.md:13). Strides beyond 2-D and out-of-bounds zero fill need more such lines. |
| `mbar_wait` | expressible with workaround | The barrier becomes a counter and the constraint is the wait; this is sound only for a sequential program order, and a program that would legitimately wait faults (C13). Transaction-byte counts and phase bits are lost. |
| `wgmma_64x64x16` | expressible with workaround | The register fragment layout is a fixed affine permutation, `reshape` plus `transpose`; C11 checks it against the PTX fragment figure. The five immediates `scale-d`, `imm-scale-a`, `imm-scale-b`, `imm-trans-a`, `imm-trans-b` are attributes; the transposed operand reads a differently shaped region, so the sketch pins both trans flags to 0 and the transposed arm is a second body (a C4 case). Each `m64nNk16` shape is its own instruction. The smem matrix descriptor (a 64-bit register value) becomes addressing attributes. The fp8 path accumulates with reduced precision, which no construct states (C7). |
| `wgmma.fence`, `commit_group`, `wait_group` | expressible with workaround | The async group is the integer state `wg_pending`; `wait_group` is a constraint that faults when false, sound only for a sequential program order (C13). |
| `st_shared_b32` | expressible as is | `bitcast_convert` + `reshape`. |
| `mma.sync` per-thread scalar view (not sketched) | expressible with workaround | Same as `wgmma`: threads are a dimension of `rf`; per-thread predication (`@p`) needs a `pred` data model and `select`. |
| `ldmatrix`, `stmatrix` (not sketched) | expressible as is | Fixed transposes between smem rows and fragments. |
| `tf32` operands (not sketched) | expressible with workaround | No `tf32` type (semantics.md:51); `reduce_precision` with `exponent_bits=8, mantissa_bits=10` (operations.md:94). |

Doc gaps:
- The docs give no example of `gather`; its seven attributes (operations.md:104) had to be matched against the StableHLO spec at https://openxla.org/stablehlo/spec.
- No doc says whether a data model may model a register file indexed by thread; `dimensions` of rank 2 with an empty unit shape (declarations.md:26) was inferred.
- Scope: the SIMT thread model is outside the owner's scope; each tensor-core instruction is modeled whole, with threads only as a dimension of `rf`, and the docs should say this is the intended view.
