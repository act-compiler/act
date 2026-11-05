# Tenstorrent Tensix

Sources:
- Tensix ISA (unpacker, matrix unit FPU, SFPU, packer, Dst register, `UNPACR`, `MVMUL`, `ELWADD`, `PACR`, `SETDMAREG`): https://github.com/tenstorrent/tt-isa-documentation (not reachable from this session, cited from the public repo)
- Kernel API view (unpack, math, pack threads; tiles of 32x32): https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis.html

A Tensix core moves 32x32 tiles from L1 circular buffers through source registers `SrcA` and `SrcB` into the matrix unit, which writes the destination register `Dst`; the packer converts `Dst` tiles back to L1.
L1 tiles are often block floating point (`bfp8_b`: 16 elements share one 8-bit exponent).

```python
from taidl import Accelerator
t = Accelerator('tensix')
t.add_data_model('l1', [1536], [1024], 'u8')      # 1.5 MB as 1 KB lines
t.add_data_model('srca', [1], [32, 32], 'f32')    # source registers, held at f32 here
t.add_data_model('srcb', [1], [32, 32], 'f32')
t.add_data_model('dst', [8], [32, 32], 'f32')     # Dst: 8 tiles in fp32 mode
t.add_initial_state('dst_acc', 0)

# unpack_a bf16: two L1 lines -> SrcA tile
i = t.add_instruction('unpack_a_bf16', [], ['src'])
i.set_inputs([['l1', ['@a.src'], ['2']]])
i.set_outputs([['srca', ['0'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[2048] parameter(0);
    %a = u8[32,32,2] reshape(%In0);
    %b = bf16[32,32] bitcast_convert(%a);
    ROOT %Out0 = f32[32,32] convert(%b);
}
""")

# unpack_b bfp8_b: 16-element blocks share one exponent (simplified layout)
i = t.add_instruction('unpack_b_bfp8', [], ['src'])
i.set_inputs([['l1', ['@a.src'], ['2']]])
i.set_outputs([['srcb', ['0'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[2048] parameter(0);
    %e8 = u8[64] slice(%In0), slice={[0:64]};
    %m8 = u8[1024] slice(%In0), slice={[64:1088]};
    %e = f32[64] convert(%e8);
    %bias = f32[64] constant(127);
    %eb = f32[64] subtract(%e, %bias);
    %two = f32[64] constant(2);
    %scale = f32[64] power(%two, %eb);
    %ms = s8[1024] bitcast_convert(%m8);
    %mf = f32[1024] convert(%ms);
    %mb = f32[64,16] reshape(%mf);
    %sb = f32[64,16] broadcast(%scale), dimensions={0};
    %k = f32[64,16] constant(0.015625);
    %v = f32[64,16] multiply(%mb, %sb);
    %w = f32[64,16] multiply(%v, %k);
    ROOT %Out0 = f32[32,32] reshape(%w);
}
""")

# mvmul: Dst[d] (+)= SrcB * SrcA, accumulate by state
i = t.add_instruction('mvmul', [], ['d'])
i.set_inputs([['srca', ['0'], ['1']], ['srcb', ['0'], ['1']], ['dst', ['@a.d'], ['1']]])
i.set_outputs([['dst', ['@a.d'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = f32[32,32] parameter(0);
    %In1 = f32[32,32] parameter(1);
    %In2 = f32[32,32] parameter(2);
    %a = bf16[32,32] convert(%In0);
    %b = bf16[32,32] convert(%In1);
    %p = f32[32,32] dot(%b, %a), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    %k = s64[] constant(`@s.dst_acc`);
    %one = s64[] constant(1);
    %isacc = pred[] compare(%k, %one), direction=EQ;
    %ib = pred[32,32] broadcast(%isacc), dimensions={};
    %z = f32[32,32] constant(0);
    %base = f32[32,32] select(%ib, %In2, %z);
    ROOT %Out0 = f32[32,32] add(%base, %p);
}
""")

# set_dst_acc: configuration op
i = t.add_instruction('set_dst_acc', ['on'], [], update=['@s.dst_acc = @c.on'],
                      constraints=['@c.on == 0 or @c.on == 1'])

# sfpu_exp: elementwise on a Dst tile
i = t.add_instruction('sfpu_exp', [], ['d'])
i.set_inputs([['dst', ['@a.d'], ['1']]])
i.set_outputs([['dst', ['@a.d'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = f32[32,32] parameter(0);
    ROOT %Out0 = f32[32,32] exponential(%In0);
}
""")

# pack bf16: Dst tile -> two L1 lines
i = t.add_instruction('pack_bf16', [], ['d', 'dst_l1'])
i.set_inputs([['dst', ['@a.d'], ['1']]])
i.set_outputs([['l1', ['@a.dst_l1'], ['2']]])
i.add_semantics("""
ENTRY {
    %In0 = f32[32,32] parameter(0);
    %h = bf16[32,32] convert(%In0);
    %b = u8[32,32,2] bitcast_convert(%h);
    ROOT %Out0 = u8[2048] reshape(%b);
}
""")
```

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `unpack_a_bf16` | expressible as is | `bitcast_convert` + `convert`; real `SrcA` holds 19-bit `tf32`-like values, which `reduce_precision` could model (operations.md:94). |
| `unpack_b_bfp8` | expressible with workaround | No block floating point type (semantics.md:55-56); the shared exponent is decoded with integer and float ops. The exact `bfp8_b` mantissa encoding (sign-magnitude, implicit bit) and the face-ordered tile layout are simplified here; matching them needs the hardware docs and many more lines. Pack to `bfp8_b` (exponent max per block, mantissa rounding) needs `reduce_max` per block plus shifts; feasible but long. |
| `mvmul` | expressible with workaround | Accumulate mode as state plus `select` (no conditional semantics, expressions.md:34). `Dst` in fp16 mode (16 tiles of 16-bit) versus fp32 mode (8 tiles) changes the data model shape at run time; not expressible except as two ISA descriptions. |
| `set_dst_acc` | expressible as is | State-only configuration (instructions.md:29). |
| `sfpu_exp` | expressible with workaround | SFPU `exp` is an approximation chosen by kernel code; the reference `exponential` (operation_meaning.md:7) differs in low bits, and no accuracy attribute exists (operations.md:35). |
| `pack_bf16` | expressible as is | `convert` with round-to-nearest-even per StableHLO; packers also offer truncation, which needs `bitcast_convert` and a mask. |
| unpack/math/pack thread handshake (not sketched) | expressible with workaround | Semaphores as state counters; three concurrent instruction streams per core are outside TAIDL (overview.md:10). |
| SFPU programs (not sketched) | not expressible | The SFPU runs a user-written per-lane program with its own registers and conditional lanes; each such program is one instruction at best. |

Doc gaps:
- No guidance on a data model whose shape depends on a mode (fp16 vs fp32 `Dst`); declarations.md:23-28 fixes shapes at declaration.
- The rounding mode of `convert` from f32 to bf16 is given only by reference to StableHLO (operation_meaning.md:8); authors need it stated for packers.
- No doc covers encoding a block-scaled format with present types, a common need.
