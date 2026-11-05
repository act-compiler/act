# AWS Trainium2 NeuronCore-v3

Sources:
- Engines, SBUF partitions, PSUM: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/arch/trainium2_arch.html
- NeuronCore-v3 overview: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v3.html
- Instruction set as NKI exposes it (`nc_matmul`, `activation`, `tensor_tensor`, `tensor_tensor_scan`, `iota`, `dma_copy`): https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.isa.html

The sketch models SBUF as 128 partitions of bytes and PSUM as 128 partitions of f32 banks.

```python
from taidl import Accelerator
trn = Accelerator('trn2')
# SBUF: 128 partitions x free bytes, untyped
trn.add_data_model('sbuf', [128, 196608], [], 'u8')
# PSUM: 128 partitions x 8 banks x 512 f32
trn.add_data_model('psum', [128, 8], [512], 'f32')

# dma_copy: HBM rows -> SBUF partitions, row stride in bytes
i = trn.add_instruction('dma_copy', ['p', 'n', 'stride'], ['src', 'pdst', 'fdst'],
                        constraints=['@c.n <= @c.stride'])
i.set_inputs([['d0', ['@a.src'], ['@c.p * @c.stride']]])
i.set_outputs([['sbuf', ['@a.pdst', '@a.fdst'], ['@c.p', '@c.n']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.p * @c.stride`] parameter(0);
    %a = u8[`@c.p`,`@c.stride`] reshape(%In0);
    ROOT %Out0 = u8[`@c.p`,`@c.n`] slice(%a), slice={[0:`@c.p`], [0:`@c.n`]};
}
""")

# nc_matmul: PSUM[m,n] += stationary[k,m]^T moving[k,n], bf16
i = trn.add_instruction('nc_matmul', ['k', 'm', 'n', 'acc'],
                        ['ws', 'mv', 'bank'],
                        constraints=['@c.k <= 128 and @c.m <= 128 and @c.n <= 512',
                                     '@c.acc == 0 or @c.acc == 1'])
i.set_inputs([['sbuf', ['0', '@a.ws'], ['@c.k', '@c.m * 2']],
              ['sbuf', ['0', '@a.mv'], ['@c.k', '@c.n * 2']],
              ['psum', ['0', '@a.bank'], ['@c.m', '1']]])
i.set_outputs([['psum', ['0', '@a.bank'], ['@c.m', '1']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.k`,`@c.m`,2] parameter(0);
    %w = bf16[`@c.k`,`@c.m`] bitcast_convert(%In0);
    %In1 = u8[`@c.k`,`@c.n`,2] parameter(1);
    %x = bf16[`@c.k`,`@c.n`] bitcast_convert(%In1);
    %In2 = f32[`@c.m`,512] parameter(2);
    %old = f32[`@c.m`,`@c.n`] slice(%In2), slice={[0:`@c.m`], [0:`@c.n`]};
    %p = f32[`@c.m`,`@c.n`] dot(%w, %x), lhs_contracting_dims={0}, rhs_contracting_dims={0};
    %accs = s64[] constant(`@c.acc`);
    %one = s64[] constant(1);
    %isacc = pred[] compare(%accs, %one), direction=EQ;
    %z = f32[`@c.m`,`@c.n`] constant(0);
    %ab = pred[`@c.m`,`@c.n`] broadcast(%isacc), dimensions={};
    %base = f32[`@c.m`,`@c.n`] select(%ab, %old, %z);
    %s = f32[`@c.m`,`@c.n`] add(%base, %p);
    %zero = s64[] constant(0);
    ROOT %Out0 = f32[`@c.m`,512] dynamic_update_slice(%In2, %s, %zero, %zero);
}
""")

# activation (ScalarE): out = exp(scale * in + bias), PSUM -> SBUF bf16
i = trn.add_instruction('act_exp', ['p', 'n', 'scale_bits'], ['src', 'bank', 'pdst', 'fdst'])
i.set_inputs([['psum', ['@a.src', '@a.bank'], ['@c.p', '1']]])
i.set_outputs([['sbuf', ['@a.pdst', '@a.fdst'], ['@c.p', '@c.n * 2']]])
i.add_semantics("""
ENTRY {
    %In0 = f32[`@c.p`,512] parameter(0);
    %x = f32[`@c.p`,`@c.n`] slice(%In0), slice={[0:`@c.p`], [0:`@c.n`]};
    %sb = u32[] constant(`@c.scale_bits`);
    %sc = f32[] bitcast_convert(%sb);
    %scb = f32[`@c.p`,`@c.n`] broadcast(%sc), dimensions={};
    %y = f32[`@c.p`,`@c.n`] multiply(%x, %scb);
    %e = f32[`@c.p`,`@c.n`] exponential(%y);
    %h = bf16[`@c.p`,`@c.n`] convert(%e);
    ROOT %Out0 = u8[`@c.p`,`@c.n`,2] bitcast_convert(%h);
}
""")

# tensor_tensor (VectorE): elementwise add of two SBUF bf16 tiles
i = trn.add_instruction('tt_add', ['p', 'n'], ['a', 'b', 'pdst', 'fdst'])
i.set_inputs([['sbuf', ['0', '@a.a'], ['@c.p', '@c.n * 2']],
              ['sbuf', ['0', '@a.b'], ['@c.p', '@c.n * 2']]])
i.set_outputs([['sbuf', ['@a.pdst', '@a.fdst'], ['@c.p', '@c.n * 2']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.p`,`@c.n`,2] parameter(0);
    %In1 = u8[`@c.p`,`@c.n`,2] parameter(1);
    %a = bf16[`@c.p`,`@c.n`] bitcast_convert(%In0);
    %b = bf16[`@c.p`,`@c.n`] bitcast_convert(%In1);
    %c = bf16[`@c.p`,`@c.n`] add(%a, %b);
    ROOT %Out0 = u8[`@c.p`,`@c.n`,2] bitcast_convert(%c);
}
""")

# iota (GpSimdE): per-partition index pattern
i = trn.add_instruction('iota_part', ['p', 'n'], ['pdst', 'fdst'])
i.set_outputs([['sbuf', ['@a.pdst', '@a.fdst'], ['@c.p', '@c.n * 4']]])
i.add_semantics("""
ENTRY {
    %i = s32[`@c.p`,`@c.n`] iota(), iota_dimension=0;
    ROOT %Out0 = u8[`@c.p`,`@c.n`,4] bitcast_convert(%i);
}
""")
```

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `dma_copy` | expressible with workaround | A region is a contiguous box (instructions.md:22) and `d0` is flat (declarations.md:29); the row stride is read as a covering range, then `slice` with backtick limits (IDLV2.g4:20, IDLV2.g4:25). Strided SBUF writes would need read-modify-write of the gap bytes. Indirect DMA is expressible with a workaround: `gather` or `dynamic_slice` takes its indices from a parameter region (semantics.md:126), and an attribute bounds the covering source range, as the Hopper swizzle does with `gather`. NKI access patterns with 3-4 strided dimensions stay a workaround. |
| `nc_matmul` | expressible with workaround | The accumulate flag becomes `select` on `constant(`@c.acc`)` (semantics.md:69); the PSUM bank is read whole and rewritten with `dynamic_update_slice` because region sizes cannot cover a partial unit shape (instructions.md:23). |
| `act_exp` | expressible with workaround | A float immediate is not an attribute type (instructions.md:13); it is passed as `u32` bits and `bitcast_convert`ed. ScalarE evaluates `exp` by table approximation; the result accuracy cannot be stated (operations.md:35 rejects `result_accuracy`). |
| `tt_add` | expressible as is | Elementwise ops over typed views; the `u8` storage costs a `bitcast_convert` per operand (semantics.md:47). |
| `iota_part` | expressible as is | `iota` (operations.md:41). |
| `tensor_tensor_scan` (not sketched) | not expressible | A prefix scan has a body; `scan` is outside the language (operation_meaning.md:22). Only a fixed-length unrolled scan is possible. |
| stochastic rounding mode (not sketched) | not expressible | No random state (operation_meaning.md:24). |
| engine semaphores (not sketched) | expressible with workaround | A no-op state-only instruction (instructions.md:29); concurrency across the four engine queues is outside TAIDL (overview.md:10). |

Doc gaps:
- No pattern for a data model whose elements change type per instruction (SBUF holds bf16, fp8, f32 and int tiles); semantics.md:47 forbids a typed view, so `u8` plus `bitcast_convert` was inferred, not documented.
- Whether slice limits may be backtick expressions is stated only for values in general (semantics.md:79); the grammar was read to confirm (IDLV2.g4:20).
- The byte order of `bitcast_convert` is implementation-defined (overview.md:42), so every typed SBUF view depends on a generator choice.
- Nothing documents how a partition-major memory with a size not divisible by the unit shape should be split into `dimensions` and `unit_dim`.
