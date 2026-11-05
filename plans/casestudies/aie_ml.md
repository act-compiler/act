# AMD XDNA / AIE-ML

Sources:
- AIE-ML architecture manual AM020 (VLIW core, 64 KB data memory, DMA buffer descriptors, locks, `MAC` and shift-round-saturate): https://docs.amd.com/r/en-US/am020-versal-aie-ml

One AIE-ML compute tile: a VLIW core with 512-bit vector registers and 1024-bit or 2048-bit accumulators, 64 KB local data memory, and a DMA driven by buffer descriptors with up to 3-D strides.

```python
from taidl import Accelerator
aie = Accelerator('aie_ml_tile')
aie.add_data_model('dm', [1024], [64], 'u8')      # 64 KB data memory, 64 B lines
aie.add_data_model('vr', [24], [64], 'u8')        # 512-bit vector registers
aie.add_data_model('acc', [8], [32], 's32')       # 1024-bit accumulators, 32 lanes
aie.add_initial_state('srs_shift', 0)
aie.add_initial_state('lock0', 0)

# dma_bd: 3-D strided stream from d0 into dm (bytes)
i = aie.add_instruction('dma_bd_2d', ['n0', 'n1', 's1'], ['src', 'dst'],
                        constraints=['@c.n0 <= @c.s1', '(@c.n0 * @c.n1) % 64 == 0'],
                        update=['@s.lock0 = 1'])
i.set_inputs([['d0', ['@a.src'], ['@c.n1 * @c.s1']]])
i.set_outputs([['dm', ['@a.dst'], ['@c.n0 * @c.n1 / 64']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n1 * @c.s1`] parameter(0);
    %a = u8[`@c.n1`,`@c.s1`] reshape(%In0);
    %b = u8[`@c.n1`,`@c.n0`] slice(%a), slice={[0:`@c.n1`], [0:`@c.n0`]};
    ROOT %Out0 = u8[`@c.n0 * @c.n1 / 64`,64] reshape(%b);
}
""")

# lock_acquire: synchronization with the DMA
i = aie.add_instruction('lock_acq0', [], [], constraints=['@s.lock0 == 1'],
                        update=['@s.lock0 = 0'])

# vld: dm line -> vector register
i = aie.add_instruction('vld', [], ['src', 'vd'])
i.set_inputs([['dm', ['@a.src'], ['1']]])
i.set_outputs([['vr', ['@a.vd'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[64] parameter(0);
    ROOT %Out0 = u8[64] copy(%In0);
}
""")

# vmac 4x8x4 int8: acc[4x4 as 16 lanes x2] += A(4x8) * B(8x4)
i = aie.add_instruction('vmac_4x8x4_s8', [], ['va', 'vb', 'ac'])
i.set_inputs([['vr', ['@a.va'], ['1']], ['vr', ['@a.vb'], ['1']],
              ['acc', ['@a.ac'], ['1']]])
i.set_outputs([['acc', ['@a.ac'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[64] parameter(0);
    %In1 = u8[64] parameter(1);
    %a8 = s8[64] bitcast_convert(%In0);
    %b8 = s8[64] bitcast_convert(%In1);
    %A = s8[8,8] reshape(%a8);
    %B = s8[8,8] reshape(%b8);
    %A32 = s32[8,8] convert(%A);
    %B32 = s32[8,8] convert(%B);
    %p = s32[8,8] dot(%A32, %B32), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    %pf = s32[64] reshape(%p);
    %ps = s32[32] slice(%pf), slice={[0:32]};
    %In2 = s32[32] parameter(2);
    ROOT %Out0 = s32[32] add(%In2, %ps);
}
""")

# set_srs: configuration op, accumulator shift for SRS
i = aie.add_instruction('set_srs', ['sh'], [], update=['@s.srs_shift = @c.sh'],
                        constraints=['@c.sh >= 0 and @c.sh < 32'])

# srs: acc s32 -> s8 with shift, round half up, saturate
i = aie.add_instruction('srs_s8', [], ['ac', 'vd'])
i.set_inputs([['acc', ['@a.ac'], ['1']]])
i.set_outputs([['vr', ['@a.vd'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = s32[32] parameter(0);
    %sh = s32[32] constant(`@s.srs_shift`);
    %one = s32[32] constant(1);
    %shm1 = s32[32] subtract(%sh, %one);
    %half = s32[32] shift_left(%one, %shm1);
    %r = s32[32] add(%In0, %half);
    %q = s32[32] shift_right_arithmetic(%r, %sh);
    %lo = s32[32] constant(-128);
    %hi = s32[32] constant(127);
    %c = s32[32] clamp(%lo, %q, %hi);
    %b = s8[32] convert(%c);
    %u = u8[32] bitcast_convert(%b);
    %z = u8[] constant(0);
    ROOT %Out0 = u8[64] pad(%u, %z), edge_padding_low={0}, edge_padding_high={32}, interior_padding={0};
}
""")
```

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `dma_bd_2d` | expressible with workaround | Buffer descriptors give up to 3-D strides with per-dimension wrap; a region is a contiguous box (instructions.md:22), so each stride pattern is a covering read plus `slice`. A 3-D pattern with a non-unit innermost step needs `slice` strides, which take backtick values (IDLV2.g4:20). Writes with strides need read-modify-write. |
| `lock_acq0` | expressible with workaround | Locks become state plus a constraint; the blocking wait between core and DMA is concurrency, out of v3. The constraint is sound only for a sequential program order, and a program that would legitimately wait faults (C13). |
| `vld` | expressible as is | `copy`. |
| `vmac_4x8x4_s8` | expressible with workaround | The register-lane layout of the matrix operands is a `reshape`; the sketch uses an 8x8 shape for brevity; the real mode set (4x8x4, 8x8x4, 4x16x8 and more) means one instruction per mode or a `select` across modes, since semantics cannot branch on an attribute (expressions.md:34). |
| `set_srs` | expressible as is | State-only configuration (instructions.md:29). |
| `srs_s8` | expressible as is | Shift, round, and saturate compose from `shift_right_arithmetic`, `add`, `clamp`, `convert` (operations.md:85, operations.md:90); a shift of 0 makes `%half` a shift by -1, an edge case left to the StableHLO reference (operation_meaning.md:8); constraining `@c.sh >= 1` avoids it. |
| VLIW bundle (not sketched) | not expressible | A bundle issues a load, a store, a MAC and a scalar op in one slot word; TAIDL has one instruction per issue and no bundle construct. Modeling each slot as an instruction is a sound functional view. |
| neighbor-tile memory access (not sketched) | not expressible | A core reads the data memory of adjacent tiles; TAIDL describes one accelerator with no array of instances or shared data models across instances. |

Doc gaps:
- The docs do not say whether a constant attribute expression can feed `shift_left` amounts, or how StableHLO's out-of-range shift rule applies; operation_meaning.md:8 defers wholly to StableHLO.
- No pattern for an accumulator register file wider than any element type (2048-bit accumulators with 64-bit lanes would need `s64`, which exists, but lane grouping is undocumented).
- No guidance on describing one tile of a tile array.
