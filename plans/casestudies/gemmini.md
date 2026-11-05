# Gemmini

Sources:
- ISA (`config_ex`, `config_ld`, `config_st`, `mvin`, `mvout`, `preload`, `compute_preloaded`, `compute_accumulated`, `loop_ws`): https://github.com/ucb-bar/gemmini (README, "ISA" section; not reachable from this session, cited from the project's public README)
- Architecture: https://arxiv.org/abs/1911.09925

The default configuration: a 16x16 int8 systolic array, a 256 KB scratchpad of 16-byte rows, and a 64 KB accumulator of 16 x int32 rows.

```python
from taidl import Accelerator
g = Accelerator('gemmini')
g.add_data_model('sp', [16384], [16], 's8')     # scratchpad rows
g.add_data_model('accm', [1024], [16], 's32')   # accumulator rows
g.add_data_model('pe_w', [1], [16, 16], 's8')   # preloaded weights in the array
g.add_initial_state('dataflow', 1)              # 1 = weight-stationary
g.add_initial_state('act', 0)                   # 0 = none, 1 = relu
g.add_initial_state('ld_stride', 16)
g.add_initial_state('acc_scale_bits', 1065353216)  # f32 1.0
g.add_initial_state('st_relu', 0)

# config_ex: dataflow and activation
i = g.add_instruction('config_ex', ['df', 'act'], [],
                      update=['@s.dataflow = @c.df', '@s.act = @c.act'],
                      constraints=['@c.df == 1', '@c.act == 0 or @c.act == 1'])

# config_ld: DRAM row stride for mvin
i = g.add_instruction('config_ld', ['stride'], [], update=['@s.ld_stride = @c.stride'])

# mvin: rows of 16 int8 from DRAM, strided
i = g.add_instruction('mvin', ['rows'], ['dram', 'spaddr'],
                      constraints=['@c.rows <= 16', '@s.ld_stride >= 16'])
i.set_inputs([['d0', ['@a.dram'], ['@c.rows * @s.ld_stride']]])
i.set_outputs([['sp', ['@a.spaddr'], ['@c.rows']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.rows * @s.ld_stride`] parameter(0);
    %a = u8[`@c.rows`,`@s.ld_stride`] reshape(%In0);
    %b = u8[`@c.rows`,16] slice(%a), slice={[0:`@c.rows`], [0:16]};
    ROOT %Out0 = s8[`@c.rows`,16] bitcast_convert(%b);
}
""")

# preload: latch B into the array (WS dataflow)
i = g.add_instruction('preload', [], ['b'], constraints=['@s.dataflow == 1'])
i.set_inputs([['sp', ['@a.b'], ['16']]])
i.set_outputs([['pe_w', ['0'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = s8[16,16] parameter(0);
    ROOT %Out0 = s8[1,16,16] reshape(%In0);
}
""")

# compute_accumulated: accm[c] += A * W (WS)
i = g.add_instruction('compute_acc', [], ['a', 'c'], constraints=['@s.dataflow == 1'])
i.set_inputs([['sp', ['@a.a'], ['16']], ['pe_w', ['0'], ['1']], ['accm', ['@a.c'], ['16']]])
i.set_outputs([['accm', ['@a.c'], ['16']]])
i.set_inplace([False, False, True])  # tied: output reuses the accumulator
i.add_semantics("""
ENTRY {
    %In0 = s8[16,16] parameter(0);
    %In1 = s8[16,16] parameter(1);
    %In2 = s32[16,16] parameter(2);
    %a = s32[16,16] convert(%In0);
    %w = s32[16,16] convert(%In1);
    %p = s32[16,16] dot(%a, %w), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = s32[16,16] add(%In2, %p);
}
""")

# mvout from accumulator: scale, optional relu, saturate to int8
i = g.add_instruction('mvout_acc', ['rows'], ['c', 'dram'])
i.set_inputs([['accm', ['@a.c'], ['@c.rows']]])
i.set_outputs([['d0', ['@a.dram'], ['@c.rows * 16']]])
i.add_semantics("""
ENTRY {
    %In0 = s32[`@c.rows`,16] parameter(0);
    %f = f32[`@c.rows`,16] convert(%In0);
    %sb = u32[] constant(`@s.acc_scale_bits`);
    %sc = f32[] bitcast_convert(%sb);
    %scb = f32[`@c.rows`,16] broadcast(%sc), dimensions={};
    %y = f32[`@c.rows`,16] multiply(%f, %scb);
    %yr = f32[`@c.rows`,16] round_nearest_even(%y);
    %z = f32[`@c.rows`,16] constant(0);
    %rl = f32[`@c.rows`,16] maximum(%yr, %z);
    %ak = s64[] constant(`@s.act`);
    %one = s64[] constant(1);
    %isr = pred[] compare(%ak, %one), direction=EQ;
    %isrb = pred[`@c.rows`,16] broadcast(%isr), dimensions={};
    %v = f32[`@c.rows`,16] select(%isrb, %rl, %yr);
    %lo = f32[`@c.rows`,16] constant(-128);
    %hi = f32[`@c.rows`,16] constant(127);
    %cl = f32[`@c.rows`,16] clamp(%lo, %v, %hi);
    %q = s8[`@c.rows`,16] convert(%cl);
    %u = u8[`@c.rows`,16] bitcast_convert(%q);
    ROOT %Out0 = u8[`@c.rows * 16`] reshape(%u);
}
""")
```

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `config_ex`, `config_ld` | expressible as is | Configuration as state updates with constraints (instructions.md:41, instructions.md:33). Gemmini packs fields into two 64-bit `rs1`/`rs2` registers; unpacking bit fields from one attribute needs `/` and `%` by powers of two (expressions.md:24), which works but no bit operator exists in expressions. |
| `mvin` | expressible with workaround | Stride from state, covering read plus `slice` (IDLV2.g4:20). Gemmini encodes the scratchpad vs accumulator target in the address's top bits; that needs one instruction per target, since a region's data model is fixed by name (instructions.md:19). |
| `preload` | expressible with workaround | The latched weights in the PEs are a hidden data model `pe_w`. The OS dataflow (preload of C, not B) is a second instruction set guarded by `@s.dataflow == 0`. |
| `compute_acc` | expressible as is | Read-modify-write accumulate with in-place mark (instructions.md:50). |
| `mvout_acc` | expressible with workaround | The f32 scale is held as `u32` bits in state (state is an integer, declarations.md:35); the relu option is a `select` across modes. Exact Gemmini scaling uses a configurable rounding (`acc_scale` with round-half-even in the default `ScaleArgs`); matching it bit for bit needs the hardware source. |
| `loop_ws` (not sketched) | expressible with workaround | A macro instruction tiling a whole matmul; one `dot` over the full `I x K x J` problem with strides from state, if bounds are attributes. Its behavior on padding and partial tiles needs reading the RTL. |
| `fence` (not sketched) | expressible as is | A state-only no-op; the queues it orders are timing (overview.md:10). |

Doc gaps:
- No doc example of a float configuration value; the `u32` bits route rests on semantics.md:69 and the unstated conversion rule of `constant` with a backtick expression.
- No doc says whether a state variable may appear in a region size (the table at expressions.md:15 says "every place", but no example uses one).
- Nothing tells an author how to choose between a hidden data model and state for latched tensors.
