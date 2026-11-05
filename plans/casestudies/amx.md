# Intel AMX

Sources:
- `LDTILECFG`, `TILELOADD`, `TILESTORED`, `TILEZERO`, `TDPBF16PS`, `TDPBSSD` in the SDM, Vol. 2: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
- Intrinsics and pseudocode: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

AMX has eight tile registers `tmm0`..`tmm7` of up to 16 rows x 64 bytes, shaped at run time by the tile configuration (palette, rows and bytes per row of each tile).

```python
from taidl import Accelerator
amx = Accelerator('amx')
# eight tiles, 16 rows x 64 bytes each, untyped
amx.add_data_model('tmm', [8, 16], [64], 'u8')
amx.add_initial_state('palette', 0)
for t in range(8):
    amx.add_initial_state(f'rows{t}', 0)
    amx.add_initial_state(f'colsb{t}', 0)

# ldtilecfg: configuration op; fields given as attributes
i = amx.add_instruction('ldtilecfg_t0', ['pal', 'r', 'cb'], [],
                        update=['@s.palette = @c.pal', '@s.rows0 = @c.r', '@s.colsb0 = @c.cb'],
                        constraints=['@c.pal == 1 and @c.r <= 16 and @c.cb <= 64'])

# tileloadd tmm0: rows x colsb bytes, stride from base
i = amx.add_instruction('tileloadd_t0', ['stride'], ['base'],
                        constraints=['@s.palette == 1', '@c.stride >= @s.colsb0'])
i.set_inputs([['d0', ['@a.base'], ['@s.rows0 * @c.stride']]])
i.set_outputs([['tmm', ['0', '0'], ['1', '16']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@s.rows0 * @c.stride`] parameter(0);
    %a = u8[`@s.rows0`,`@c.stride`] reshape(%In0);
    %b = u8[`@s.rows0`,`@s.colsb0`] slice(%a), slice={[0:`@s.rows0`], [0:`@s.colsb0`]};
    %z = u8[] constant(0);
    ROOT %Out0 = u8[16,64] pad(%b, %z), edge_padding_low={0,0}, edge_padding_high={`16 - @s.rows0`,`64 - @s.colsb0`}, interior_padding={0,0};
}
""")

# tilezero tmm2
i = amx.add_instruction('tilezero_t2', [], [], constraints=['@s.palette == 1'])
i.set_outputs([['tmm', ['2', '0'], ['1', '16']]])
i.add_semantics("""
ENTRY {
    ROOT %Out0 = u8[16,64] constant(0);
}
""")

# tdpbf16ps tmm2 += tmm0 * tmm1 (bf16 pairs -> f32), shapes fixed at 16x16x32
i = amx.add_instruction('tdpbf16ps_2_0_1', [], [],
                        constraints=['@s.palette == 1',
                                     '@s.rows0 == 16 and @s.colsb0 == 64 and @s.rows1 == 16 and @s.colsb1 == 64'])
i.set_inputs([['tmm', ['0', '0'], ['1', '16']], ['tmm', ['1', '0'], ['1', '16']],
              ['tmm', ['2', '0'], ['1', '16']]])
i.set_outputs([['tmm', ['2', '0'], ['1', '16']]])
i.set_inplace([False, False, True])
i.add_semantics("""
ENTRY {
    %In0 = u8[16,32,2] parameter(0);
    %A = bf16[16,32] bitcast_convert(%In0);
    %In1 = u8[16,16,2,2] parameter(1);
    %B4 = bf16[16,16,2] bitcast_convert(%In1);
    %Bt = bf16[16,2,16] transpose(%B4), dimensions={0,2,1};
    %B = bf16[32,16] reshape(%Bt);
    %In2 = u8[16,16,4] parameter(2);
    %C = f32[16,16] bitcast_convert(%In2);
    %p = f32[16,16] dot(%A, %B), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    %s = f32[16,16] add(%C, %p);
    ROOT %Out0 = u8[16,16,4] bitcast_convert(%s);
}
""")

# tilestored tmm2: write rows x colsb bytes with a stride
i = amx.add_instruction('tilestored_t2', ['stride'], ['base'],
                        constraints=['@s.palette == 1'])
i.set_inputs([['tmm', ['2', '0'], ['1', '16']],
               ['d0', ['@a.base'], ['@s.rows2 * @c.stride']]])
i.set_outputs([['d0', ['@a.base'], ['@s.rows2 * @c.stride']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[16,64] parameter(0);
    %t = u8[`@s.rows2`,`@s.colsb2`] slice(%In0), slice={[0:`@s.rows2`], [0:`@s.colsb2`]};
    %In1 = u8[`@s.rows2 * @c.stride`] parameter(1);
    %m = u8[`@s.rows2`,`@c.stride`] reshape(%In1);
    %zero = s64[] constant(0);
    %u = u8[`@s.rows2`,`@c.stride`] dynamic_update_slice(%m, %t, %zero, %zero);
    ROOT %Out0 = u8[`@s.rows2 * @c.stride`] reshape(%u);
}
""")
```

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `ldtilecfg` | expressible with workaround | The real instruction reads a 64-byte configuration from memory; a state update cannot read a region (instructions.md:41, expressions.md:8), so the fields become attributes. Per-tile `rows`/`colsb` are 16 scalar state variables, one instruction per tile, because state cannot be indexed by an attribute (expressions.md:15). |
| `tileloadd` | expressible with workaround | The tile index is baked into the instruction name (64 variants across tiles and operands), since a region start that selects a tile could be `@a.t`, but the state it reads (`@s.rows<t>`) cannot. Shapes depend on state through backtick expressions (semantics.md:62). Zeroing the unused rows matches the SDM. |
| `tilezero` | expressible as is | `constant` over the whole tile. |
| `tdpbf16ps` | expressible with workaround | One variant per (dst, a, b) triple: 8x8x8 = 512 names, or fewer if constraints pin the triple. The VNNI pair layout of B is a `transpose`; full generality over configured `M, N, K` needs shapes from state, which works, but the constraint-pinned 16x16x32 form is simpler. The SDM's intermediate rounding (FP32 FMA with DAZ/FTZ, round-to-nearest-even per pair) is not expressible precisely. |
| `tilestored` | expressible with workaround | A strided store rewrites the gap bytes with their own old value: `d0` is both input and output and `dynamic_update_slice` merges (operations.md:103). A generator sees false dependences on the gap. |
| `tdpbssd` (not sketched) | expressible as is | `dot` with `s8` operands and `s32` result, wrapping as StableHLO gives (operation_meaning.md:8). |

Doc gaps:
- Whether an input region and an output region of `d0` may coincide (the read-modify-write store) is implied by instructions.md:26 but no example shows it.
- Whether a backtick expression may appear inside `edge_padding_high` lists is implied by semantics.md:79-80; the grammar was read to confirm (IDLV2.g4:19, IDLV2.g4:25).
- No guidance on register files whose element type changes per instruction (tiles hold `s8`, `bf16`, or `f32`); `u8` plus `bitcast_convert` is the only route (semantics.md:47).
