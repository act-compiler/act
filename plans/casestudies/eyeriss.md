# Eyeriss-style row-stationary array

Sources:
- Row-stationary dataflow and PE array: https://eyeriss.mit.edu/
- Eyeriss JSSC 2017 (12x14 PEs, 108 KB global buffer, per-PE scratchpads, RLC compression, scan-chain configuration): https://ieeexplore.ieee.org/document/7738524

Eyeriss has no public instruction set; its "instructions" are the commands of the host interface and the configuration scan chain.
The sketch uses a plausible command set: load the global buffer, configure the mapping, and run a 1-D row convolution per PE set.

```python
from taidl import Accelerator
e = Accelerator('eyeriss')
e.add_data_model('glb', [54], [1024], 's16')   # 108 KB global buffer
e.add_data_model('psum', [64], [1024], 's32')  # partial sum storage (simplified)
e.add_initial_state('R', 3)                    # filter rows
e.add_initial_state('S', 3)                    # filter width
e.add_initial_state('relu', 0)

# load_glb: DRAM bytes -> global buffer lines
i = e.add_instruction('load_glb', ['n'], ['src', 'dst'])
i.set_inputs([['d0', ['@a.src'], ['@c.n * 2048']]])
i.set_outputs([['glb', ['@a.dst'], ['@c.n']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 2048`] parameter(0);
    %a = u8[`@c.n`,1024,2] reshape(%In0);
    ROOT %Out0 = s16[`@c.n`,1024] bitcast_convert(%a);
}
""")

# configure: scan-chain mapping as attributes
i = e.add_instruction('configure', ['r', 's', 'relu'], [],
                      update=['@s.R = @c.r', '@s.S = @c.s', '@s.relu = @c.relu'],
                      constraints=['@c.r == 3 and @c.s == 3'])

# conv_rs_3x3: psum += 2-D conv of one ifmap plane (W wide) by one 3x3 filter
i = e.add_instruction('conv_rs_3x3', ['w'], ['ifm', 'flt', 'ps'],
                      constraints=['@s.R == 3 and @s.S == 3', '@c.w <= 1024'])
i.set_inputs([['glb', ['@a.ifm'], ['3']], ['glb', ['@a.flt'], ['1']], ['psum', ['@a.ps'], ['1']]])
i.set_outputs([['psum', ['@a.ps'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = s16[3,1024] parameter(0);
    %In1 = s16[1024] parameter(1);
    %In2 = s32[1024] parameter(2);
    %x = s32[3,1024] convert(%In0);
    %f9 = s16[9] slice(%In1), slice={[0:9]};
    %f = s32[9] convert(%f9);
    %fk = s32[3,3] reshape(%f);
    // unroll S = 3 column taps; R = 3 rows contract in dot
    %x0 = s32[3,1022] slice(%x), slice={[0:3], [0:1022]};
    %x1 = s32[3,1022] slice(%x), slice={[0:3], [1:1023]};
    %x2 = s32[3,1022] slice(%x), slice={[0:3], [2:1024]};
    %g0 = s32[3,1] slice(%fk), slice={[0:3], [0:1]};
    %f0 = s32[3] reshape(%g0);
    %g1 = s32[3,1] slice(%fk), slice={[0:3], [1:2]};
    %f1 = s32[3] reshape(%g1);
    %g2 = s32[3,1] slice(%fk), slice={[0:3], [2:3]};
    %f2 = s32[3] reshape(%g2);
    %y0 = s32[1022] dot(%x0, %f0), lhs_contracting_dims={0}, rhs_contracting_dims={0};
    %y1 = s32[1022] dot(%x1, %f1), lhs_contracting_dims={0}, rhs_contracting_dims={0};
    %y2 = s32[1022] dot(%x2, %f2), lhs_contracting_dims={0}, rhs_contracting_dims={0};
    %y01 = s32[1022] add(%y0, %y1);
    %y = s32[1022] add(%y01, %y2);
    %z = s32[] constant(0);
    %yp = s32[1024] pad(%y, %z), edge_padding_low={0}, edge_padding_high={2}, interior_padding={0};
    ROOT %Out0 = s32[1024] add(%In2, %yp);
}
""")

# store_ofm: psum -> DRAM with optional relu, saturate to s16
i = e.add_instruction('store_ofm', [], ['ps', 'dst'])
i.set_inputs([['psum', ['@a.ps'], ['1']]])
i.set_outputs([['d0', ['@a.dst'], ['2048']]])
i.add_semantics("""
ENTRY {
    %In0 = s32[1024] parameter(0);
    %z = s32[1024] constant(0);
    %r = s32[1024] maximum(%In0, %z);
    %k = s64[] constant(`@s.relu`);
    %one = s64[] constant(1);
    %c = pred[] compare(%k, %one), direction=EQ;
    %cb = pred[1024] broadcast(%c), dimensions={};
    %v = s32[1024] select(%cb, %r, %In0);
    %lo = s32[1024] constant(-32768);
    %hi = s32[1024] constant(32767);
    %s = s32[1024] clamp(%lo, %v, %hi);
    %h = s16[1024] convert(%s);
    %b = u8[1024,2] bitcast_convert(%h);
    ROOT %Out0 = u8[2048] reshape(%b);
}
""")
```

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `load_glb` | expressible as is | Contiguous copy and `bitcast_convert`. |
| `configure` | expressible with workaround | The scan chain is a bitstream; its fields become attributes and state. The mapping (which PE holds which row, PE-set folding) has no functional effect, so dropping it is sound for a test oracle but loses what a compiler backend must choose. |
| `conv_rs_3x3` | expressible with workaround | `convolution` and `reduce_window` are outside the language (operation_meaning.md:22-23); the conv is unrolled into `slice` + `dot` per tap, valid only for a fixed `S`. A filter width from configuration needs one instruction per width or a loop construct. Stride, dilation, and channel accumulation multiply the unrolling. |
| `store_ofm` | expressible as is | `select` across modes, `clamp`, `convert`. |
| RLC compressed DRAM traffic (not sketched) | not expressible | Run-length decoding has data-dependent output positions; it needs `scatter` or a loop, outside the language (operation_meaning.md:22). |
| zero-gating of PE work (not sketched) | expressible as is | Power only; no functional effect. |

Doc gaps:
- The docs say why `convolution` is out (operation_meaning.md:23) but give no recommended workaround pattern, which CNN accelerators need first.
- No doc says whether a 1-D data model with a large unit shape or a 2-D one is preferred for a buffer read in partial lines; the choice changed every start expression here.
