# Owner decisions

Each row is a conflict between the owner's papers, the current glossary, and the tools.
The suggested default is what the other files recommend.

| # | Conflict | Evidence | Suggested default |
| --- | --- | --- | --- |
| 1 | One named storage unit: "data model" (glossary, `add_data_model`) or "tensor buffer" | Both papers: storage units "collectively termed data model", each one a "tensor buffer"; NKI says buffer (SBUF, PSUM) | "tensor buffer" (short "buffer") per unit, "data model" for the set; keep the `add_data_model` API name. |
| 2 | Persistent integer state: "state variable" (glossary) or "control register" | [TAIDL] uses "control registers" 31 times as part of the data model; the glossary bans "register" | "control register"; keep the `@s.` prefix and `add_initial_state`. |
| 3 | Dimensions a region indexes: "addressed dimensions" (glossary) or "access dimensions" | [ACT] Sec. 3.2 defines "access dimensions (S0)"; the glossary bans "access dims" | "access dimensions". |
| 4 | Compiled output: "kernel" (glossary bans "assembly") or "assembly code" | [ACT] defines assembly code as "a stream of concrete instructions and a constant tensor"; `act-oracle` keeps kernels under `assembly/`; `act-backend/README.md` says "machine code" | "assembly code" for the output, "tiled kernel" for the XLA-HLO input, "kernel" for an oracle function. |
| 5 | Data an instruction reads or writes: "region" (glossary) or "data slice" | [ACT] defines "data slice d[Is:Ie]"; `slice` is also an operation word | "region"; mention "data slice" once as the paper's name. |
| 6 | Grammar rule `instruction` in `taidl/taidl/antlr4/IDLV2.g4:8` names a semantics line | The glossary bans "instruction (in a semantics)" | Rename the rule `line` when the grammar next changes. |
| 7 | Python parameter `computation_attr` vs the term "computational attribute" | [ACT] says "computational"; the grammar token is `COMPUTATIONAL` | Keep the API name; docs say "computational attribute". |
| 8 | `broadcast` means StableHLO `broadcast_in_dim`, not XLA `Broadcast` | [XLA] has both `Broadcast` and `BroadcastInDim` | Keep `broadcast`; the spec names its reference operation. |
