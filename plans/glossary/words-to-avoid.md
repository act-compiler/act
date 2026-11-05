# Words to avoid

Words to replace when they name a TAIDL concept in docs, error messages, or user-visible names.

| Word | Replacement |
| --- | --- |
| data model (for one storage unit) | tensor buffer, or buffer |
| state variable | control register |
| register (unqualified) | control register, or tensor buffer for a hardware register file |
| off-chip memory, DRAM, main memory, host memory | global memory |
| HBM (for `d0` in general) | global memory |
| addressed dimensions, access dims, outer dims | access dimensions |
| unit shape, unit dim, inner dims | base element shape |
| dtype, data type (outside NumPy, JAX, or PyTorch names) | element type |
| region | data slice |
| slice (for a data slice) | data slice |
| tile, window (for a data slice or its value) | data slice, input value, or output value |
| region value | input value or output value |
| calling attribute, parameter (for an instruction attribute) | computational attribute or addressing attribute |
| immediate | computational attribute |
| state update | control register assignment |
| guard, assertion, precondition (for a validity constraint) | validity constraint |
| line, statement (in an instruction semantics) | operation |
| operation word, opcode | tensor operator |
| reference operation | reference operator |
| TAIDL specification, ISA specification, TAIDL definition, spec (for one file) | ISA description |
| target (for an accelerator) | accelerator |
| machine code | assembly code |
| tensor kernel, input kernel | tiled kernel |
| oracle, generated oracle (in prose) | test oracle |
| simulator (unqualified) | test oracle, or functional simulator on first use |
| backend (unqualified) | compiler backend, or generator |
| refuse, refuses, refused | reject, rejects, rejected |
| seamless support | the concrete claim, such as "plugs into XLA" |
| leverages | uses |
| just (as in "just from TAIDL specifications") | "from an ISA description alone", or delete |
| honored, respected | applies, holds, or the concrete check |
