# Contested terms

Each row names a concept where the sources disagree and the evidence on each side.
The term in force is in `glossary/terms.md`, set by owner decisions 14, 19, 20, 21, and 32.

| # | Concept | Evidence |
| --- | --- | --- |
| 1 | One named storage unit | Both papers: storage units "collectively termed data model", each one a "tensor buffer"; NKI says buffer (SBUF, PSUM). |
| 2 | Persistent integer state | [TAIDL] uses "control registers" 31 times as part of the data model; the shipped docstrings and messages say "state variable". |
| 3 | Dimensions a region indexes | [ACT] Sec. 3.2 defines "access dimensions (S0)". |
| 4 | Compiled output | [ACT] defines assembly code as "a stream of concrete instructions and a constant tensor"; `act-oracle` keeps kernels under `assembly/`; `act-backend/README.md` says "machine code". |
| 5 | Data an instruction reads or writes | [ACT] defines "data slice d[Is:Ie]"; `slice` is also a tensor operator; "tile" means an on-chip value in NKI. |
| 6 | Grammar rule `instruction` | `taidl/taidl/antlr4/IDLV2.g4` uses it for one line of a semantics, while the class `Instruction` names an ISA instruction; StableHLO and MLIR call the line an operation. |
| 7 | Parameter `computation_attr` | [ACT] says "computational"; the grammar token is `COMPUTATIONAL`. |
| 8 | `broadcast` | [XLA] has both `Broadcast` and `BroadcastInDim`; `Broadcast` adds leading dimensions only. |
| 9 | Off-chip memory behind `d0` | Both papers write "global memory" with HBM as an alias; the `hbm` kernel argument and the backend buffer name spell it `hbm`. |
| 10 | The user's file | [ACT] writes "ISA description" 54 times; [TAIDL] writes "TAIDL definition"; the site writes "ISA specifications". |
