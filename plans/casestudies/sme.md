# Arm SME

Sources:
- A64 instruction descriptions (`SMSTART`, `SMSTOP`, `FMOPA`, `LD1W` (ZA tile slice), `ST1W`, `ZERO`, `MOVA`): https://developer.arm.com/documentation/ddi0602/latest
- SME supplement to the architecture reference manual: https://developer.arm.com/documentation/ddi0616/latest

SME adds the ZA array (SVL x SVL bytes), streaming SVE mode (`PSTATE.SM`), ZA enable (`PSTATE.ZA`), and outer-product accumulate into ZA tiles.
SVL is implementation-chosen; the sketch fixes SVL = 512 bits (64 bytes), so ZA is 64 x 64 bytes and an f32 tile is 16 x 16.

```python
from taidl import Accelerator
sme = Accelerator('sme512')
sme.add_data_model('za', [64], [64], 'u8')     # ZA rows
sme.add_data_model('z', [32], [64], 'u8')      # Z0..Z31
sme.add_data_model('p', [16], [64], 'pred')    # P0..P15, one flag per byte
sme.add_initial_state('sm', 0)
sme.add_initial_state('za_on', 0)

# smstart: configuration op, streaming mode and ZA on
i = sme.add_instruction('smstart', [], [], update=['@s.sm = 1', '@s.za_on = 1'])
i.set_outputs([['za', ['0'], ['64']]])
i.add_semantics("""
ENTRY {
    ROOT %Out0 = u8[64,64] constant(0);
}
""")

# ld1w za<t>h slice: one row of an f32 tile from memory, predicated
i = sme.add_instruction('ld1w_zah', ['tile', 'slice'], ['base', 'pg'],
                        constraints=['@s.sm == 1 and @s.za_on == 1',
                                     '@c.tile < 4 and @c.slice < 16'])
i.set_inputs([['d0', ['@a.base'], ['64']], ['p', ['@a.pg'], ['1']],
              ['za', ['@c.tile + 4 * @c.slice'], ['1']]])
i.set_outputs([['za', ['@c.tile + 4 * @c.slice'], ['1']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[64] parameter(0);
    %In1 = pred[64] parameter(1);
    %z = u8[64] constant(0);
    ROOT %Out0 = u8[64] select(%In1, %In0, %z);
}
""")

# fmopa za<t>.s: tile += (pn & zn) outer (pm & zm), f32
i = sme.add_instruction('fmopa_s', ['tile'], ['pn', 'pm', 'zn', 'zm'],
                        constraints=['@s.sm == 1 and @s.za_on == 1', '@c.tile < 4'])
i.set_inputs([['p', ['@a.pn'], ['1']], ['p', ['@a.pm'], ['1']],
              ['z', ['@a.zn'], ['1']], ['z', ['@a.zm'], ['1']],
              ['za', ['0'], ['64']]])
i.set_outputs([['za', ['0'], ['64']]])
i.add_semantics("""
ENTRY {
    %In0 = pred[16,4] parameter(0);
    %In1 = pred[16,4] parameter(1);
    %pa = pred[16] slice(%In0), slice={[0:16], [0:1]};
    %pb = pred[16] slice(%In1), slice={[0:16], [0:1]};
    %In2 = u8[16,4] parameter(2);
    %In3 = u8[16,4] parameter(3);
    %a = f32[16] bitcast_convert(%In2);
    %b = f32[16] bitcast_convert(%In3);
    %o = f32[16,16] dot(%a, %b), lhs_contracting_dims={}, rhs_contracting_dims={};
    %In4 = u8[16,4,64] parameter(4);
    %za = u8[16,4,16,4] reshape(%In4);
    %r0 = s64[] constant(0);
    %c0 = s64[] constant(0);
    %t = s64[] constant(`@c.tile`);
    %tile4 = u8[16,1,16,4] dynamic_slice(%za, %r0, %t, %c0, %c0), dynamic_slice_sizes={16,1,16,4};
    %tile = u8[16,16,4] reshape(%tile4);
    %acc = f32[16,16] bitcast_convert(%tile);
    %s = f32[16,16] add(%acc, %o);
    %pa2 = pred[16,16] broadcast(%pa), dimensions={0};
    %pb2 = pred[16,16] broadcast(%pb), dimensions={1};
    %m = pred[16,16] and(%pa2, %pb2);
    %n = f32[16,16] select(%m, %s, %acc);
    %nb = u8[16,1,16,4] bitcast_convert(%n);
    %w = u8[16,4,16,4] dynamic_update_slice(%za, %nb, %r0, %t, %c0, %c0);
    ROOT %Out0 = u8[64,64] reshape(%w);
}
""")

# smstop: leave streaming mode, ZA contents become undefined
i = sme.add_instruction('smstop', [], [], update=['@s.sm = 0', '@s.za_on = 0'])
```

The f32 tile `ZAt.S` is every fourth ZA row from row `t` (https://developer.arm.com/documentation/ddi0616/latest).

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `smstart` | expressible as is | Mode bits are state (declarations.md:34); zeroing ZA is a `constant` output. |
| `ld1w_zah` | expressible with workaround | Predication is a `pred` data model plus `select` (operations.md:89). The tile slice row is computed from computational attributes in a start, which expressions.md:13 allows. Real SME scales by vector length at run time; SVL is fixed per ISA description because data model sizes are Python integers (accelerator.py:37-41). |
| `fmopa_s` | expressible with workaround | The outer product is `dot` with empty contracting lists (operations.md:34). The interleaved tile-in-ZA layout needs the whole ZA as input and `dynamic_slice`/`dynamic_update_slice` with a `constant(`@c.tile`)` index, because a region box cannot select every fourth row (instructions.md:22). `pred` element per byte is a modeling choice; SVE predicates are one bit per byte. |
| `smstop` | expressible with workaround | State-only (instructions.md:29); ZA contents after `smstop` have no TAIDL value; v3 has no undefined behavior, so it must pick a defined value (such as zero) or make a later read a defined error; the sketch rules such reads out by constraints. |
| SVL-agnostic code (not sketched) | not expressible | A data model size or shape that depends on a run-time vector length; shapes may depend on state (semantics.md:62) but data model sizes may not. |

Doc gaps:
- No example of a computational attribute used in a region start (allowed by the table at expressions.md:13 but unshown).
- `dynamic_slice` operand order and index types are given only by reference to StableHLO (operations.md:102, operations.md:123); no TAIDL example.
- Nothing says how to write a bit-packed predicate register; `pred` storage width in a data model is unspecified.
