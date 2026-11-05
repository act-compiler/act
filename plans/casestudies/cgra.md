# Generic CGRA with a configuration bitstream

Sources:
- CGRA survey (PE arrays, context memory, configuration): https://dl.acm.org/doi/10.1145/3357375 (not reachable from this session)
- Open-source CGRA with per-tile configuration memory: https://github.com/tancheng/VectorCGRA (not reachable from this session)

A CGRA is an array of PEs, each with an ALU, a small register file, and a context memory; a host writes a configuration bitstream, then launches it to stream data through the array.

```python
from taidl import Accelerator
c = Accelerator('cgra4x4')
c.add_data_model('spm', [4, 1024], [], 's32')    # 4 scratchpad banks
c.add_data_model('cfg', [16, 8], [], 'u64')      # 16 PEs x 8 contexts
c.add_initial_state('cfg_id', 0)

# dma_in: DRAM -> scratchpad bank
i = c.add_instruction('dma_in', ['n'], ['src', 'bank', 'dst'])
i.set_inputs([['d0', ['@a.src'], ['@c.n * 4']]])
i.set_outputs([['spm', ['@a.bank', '@a.dst'], ['1', '@c.n']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 4`] parameter(0);
    %a = u8[`@c.n`,4] reshape(%In0);
    ROOT %Out0 = s32[`@c.n`] bitcast_convert(%a);
}
""")

# load_cfg: write the bitstream; its meaning is tagged by an id
i = c.add_instruction('load_cfg', ['id'], ['src'], update=['@s.cfg_id = @c.id'],
                      constraints=['@c.id == 1 or @c.id == 2'])
i.set_inputs([['d0', ['@a.src'], ['1024']]])
i.set_outputs([['cfg', ['0', '0'], ['16', '8']]])
i.add_semantics("""
ENTRY {
    %In0 = u8[1024] parameter(0);
    %a = u8[128,8] reshape(%In0);
    ROOT %Out0 = u64[128] bitcast_convert(%a);
}
""")

# launch: run the loaded configuration over n elements
# id 1 = vector multiply-add a*b+c, id 2 = elementwise max
i = c.add_instruction('launch', ['n'], ['x'],
                      constraints=['@s.cfg_id == 1 or @s.cfg_id == 2'])
i.set_inputs([['spm', ['0', '@a.x'], ['1', '@c.n']], ['spm', ['1', '@a.x'], ['1', '@c.n']],
              ['spm', ['2', '@a.x'], ['1', '@c.n']]])
i.set_outputs([['spm', ['3', '@a.x'], ['1', '@c.n']]])
i.add_semantics("""
ENTRY {
    %In0 = s32[`@c.n`] parameter(0);
    %In1 = s32[`@c.n`] parameter(1);
    %In2 = s32[`@c.n`] parameter(2);
    %m = s32[`@c.n`] multiply(%In0, %In1);
    %fma = s32[`@c.n`] add(%m, %In2);
    %mx = s32[`@c.n`] maximum(%In0, %In1);
    %k = s64[] constant(`@s.cfg_id`);
    %one = s64[] constant(1);
    %is1 = pred[] compare(%k, %one), direction=EQ;
    %b = pred[`@c.n`] broadcast(%is1), dimensions={};
    ROOT %Out0 = s32[`@c.n`] select(%b, %fma, %mx);
}
""")

# dma_out: scratchpad bank -> DRAM
i = c.add_instruction('dma_out', ['n'], ['bank', 'src', 'dst'])
i.set_inputs([['spm', ['@a.bank', '@a.src'], ['1', '@c.n']]])
i.set_outputs([['d0', ['@a.dst'], ['@c.n * 4']]])
i.add_semantics("""
ENTRY {
    %In0 = s32[`@c.n`] parameter(0);
    %a = u8[`@c.n`,4] bitcast_convert(%In0);
    ROOT %Out0 = u8[`@c.n * 4`] reshape(%a);
}
""")
```

| Instruction | Verdict | Missing construct or workaround |
| --- | --- | --- |
| `dma_in`, `dma_out` | expressible as is | Two addressed dimensions select bank and offset (declarations.md:24). |
| `load_cfg` | expressible with workaround | The bitstream is stored as data, but its meaning cannot be derived from its bits: semantics cannot interpret a region's contents as a program (operation_meaning.md:22 excludes `while`, `call`, `map`). A tag `@c.id` stands for which known configuration was loaded. |
| `launch` | expressible with workaround | Only a closed, enumerated set of configurations works: each one's dataflow graph is written into the semantics and chosen by `select` on `@s.cfg_id`. Every configuration is computed on every launch, and a new kernel mapping needs a new ISA specification. Configurations with loops carried across PEs (recurrences, reductions with feedback) need `scan` or `while` and are not expressible. |
| general bitstream-defined semantics (not sketched) | not expressible | An instruction whose semantics is a function of a data region's contents; this is an interpreter, beyond a dataflow semantics. |

Doc gaps:
- The docs do not say whether a state variable may stand for "which configuration is loaded"; this works by S5.4 (instructions.md:42-46) but is a modeling idiom without an example.
- No doc tells an author what to do for a programmable unit (CGRA, SFPU, scalar core); a short "outside TAIDL's scope" note would save trial and error.
