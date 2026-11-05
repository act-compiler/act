# Google TPU (MXU, VPU, VMEM)

Sources:
- TPUv1 ISA and MXU: https://arxiv.org/abs/1704.04760
- TPU VM architecture (MXU 128x128, VPU, VMEM): https://cloud.google.com/tpu/docs/system-architecture-tpu-vm
- VMEM tiling and layouts: https://docs.jax.dev/en/latest/pallas/tpu/details.html

The sketch models one TensorCore: VMEM as untyped bytes, the MXU weight latch as a data model, and the vector unit's 8x128 vregs.

```python
from taidl import Accelerator
tpu = Accelerator('tpu')
# VMEM: 8x128 lanes per tile row, untyped bytes
tpu.add_data_model('vmem', [4096], [8, 512], 'u8')
# MXU weight latch: one 128x128 bf16 tile
tpu.add_data_model('mxu_w', [1], [128, 128], 'bf16')
# MXU result buffer: f32 accumulators
tpu.add_data_model('mrb', [64], [8, 128], 'f32')
tpu.add_initial_state('act_fn', 0)

# dma_in: HBM bytes -> VMEM tile rows
i = tpu.add_instruction('dma_in', ['n'], ['src', 'dst'])
i.set_inputs([['d0', ['@a.src'], ['@c.n * 4096']]])
i.set_outputs([['vmem', ['@a.dst'], ['@c.n']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 4096`] parameter(0);
    ROOT %Out0 = u8[`@c.n`,8,512] reshape(%In0);
}
""")

# push_w: latch a bf16 weight tile into the MXU
i = tpu.add_instruction('push_w', [], ['src'])
i.set_inputs([['vmem', ['@a.src'], ['16']]])
i.set_outputs([['mxu_w', ['0'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[16,8,512] parameter(0);
    %a = u8[128,128,2] reshape(%In0);
    ROOT %Out0 = bf16[1,128,128] bitcast_convert(%a);
}
""")

# matmul_acc: x[8m,128] * W -> mrb += result
i = tpu.add_instruction('matmul_acc', ['m'], ['src', 'acc'])
i.set_inputs([['vmem', ['@a.src'], ['@c.m * 2']], ['mxu_w', ['0'], ['1']],
              ['mrb', ['@a.acc'], ['@c.m']]])
i.set_outputs([['mrb', ['@a.acc'], ['@c.m']]])
i.set_inplace([False, False, True])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.m * 2`,8,512] parameter(0);
    %x8 = u8[`@c.m * 8`,128,2] reshape(%In0);
    %x = bf16[`@c.m * 8`,128] bitcast_convert(%x8);
    %In1 = bf16[128,128] parameter(1);
    %In2 = f32[`@c.m * 8`,128] parameter(2);
    %p = f32[`@c.m * 8`,128] dot(%x, %In1), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = f32[`@c.m * 8`,128] add(%In2, %p);
}
""")

# set_act: configuration op, selects VPU activation
i = tpu.add_instruction('set_act', ['f'], [], update=['@s.act_fn = @c.f'],
                        constraints=['@c.f >= 0 and @c.f <= 1'])

# vpu_act: mrb f32 -> bf16 VMEM, relu or identity by state
i = tpu.add_instruction('vpu_act', ['m'], ['src', 'dst'])
i.set_inputs([['mrb', ['@a.src'], ['@c.m']]])
i.set_outputs([['vmem', ['@a.dst'], ['@c.m']]])
i.add_semantics("""
ENTRY {
    %In0 = f32[`@c.m`,8,128] parameter(0);
    %z = f32[`@c.m`,8,128] constant(0);
    %r = f32[`@c.m`,8,128] maximum(%In0, %z);
    %f = s64[] constant(`@s.act_fn`);
    %one = s64[] constant(1);
    %is = pred[] compare(%f, %one), direction=EQ;
    %isb = pred[`@c.m`,8,128] broadcast(%is), dimensions={};
    %y = f32[`@c.m`,8,128] select(%isb, %r, %In0);
    %h = bf16[`@c.m`,8,128] convert(%y);
    %zb = bf16[] constant(0);
    %pad = bf16[`@c.m`,8,256] pad(%h, %zb), edge_padding_low={0,0,0}, edge_padding_high={0,0,128}, interior_padding={0,0,0};
    %b = u8[`@c.m`,8,256,2] bitcast_convert(%pad);
    ROOT %Out0 = u8[`@c.m`,8,512] reshape(%b);
}
""")

# vreduce: cross-sublane max over a VMEM f32 tile
i = tpu.add_instruction('vreduce_max', [], ['src', 'dst'])
i.set_inputs([['mrb', ['@a.src'], ['1']]])
i.set_outputs([['mrb', ['@a.dst'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = f32[8,128] parameter(0);
    %m = f32[128] reduce_max(%In0), dimensions={0};
    ROOT %Out0 = f32[8,128] broadcast(%m), dimensions={1};
}
""")
```

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `dma_in` | expressible as is | Contiguous byte copy; strided HBM tiles need the C3 workaround (read the covering range, `slice` with a backtick stride). |
| `push_w` | expressible with workaround | MXU weight latch modeled as a hidden data model `mxu_w`; VMEM is `u8` because a data model has one element type (semantics.md:47), so every typed read is `reshape` + `bitcast_convert`. |
| `matmul_acc` | expressible as is | Accumulation is read-modify-write of the same region (instructions.md:26, instructions.md:50); `dot` with bf16 operands and f32 result relies on mixed-type `dot`, which operations.md:121 allows only implicitly. |
| `set_act` | expressible as is | State-only instruction (instructions.md:29) with a constraint (instructions.md:33). |
| `vpu_act` | expressible with workaround | Mode-dependent semantics computes every mode and picks one with `select` on `constant(`@s.act_fn`)` (semantics.md:69); there is no conditional semantics (expressions.md:34). |
| `vreduce_max` | expressible as is | `reduce_max` plus `broadcast` (operations.md:112, operations.md:97). |

Doc gaps:
- No doc says how to model a hidden datapath latch (the MXU weight registers); the glossary bans "register" (glossary.md:21) but gives no pattern.
- Whether `constant` with a backtick expression and a float or `s64` result type converts the integer is not stated (semantics.md:69).
- Mixed-precision `dot` (bf16 in, f32 out) is allowed only by reference to StableHLO (operations.md:121, operation_meaning.md:9); the accumulation precision of the MXU is not expressible.
- No example uses state, constraints, or updates (mini_trn.py:1-102 declares none); the order rules had to be taken from instructions.md:42-46 alone.
