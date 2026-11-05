# What TAIDL already expresses well

- Named on-chip memories map directly onto data models with access dimensions and a base element shape: VMEM, SBUF partitions, PSUM banks, AMX tiles, ZA rows, Gemmini scratchpad and accumulator, Tensix `Dst` (declarations.md:23-28, S4.3).
- `dot` as StableHLO `dot_general` covers every main compute op met: MXU and TensorE matmul, `wgmma`, AMX `TDP*`, SME outer products with empty contracting lists, AIE-ML `MAC` modes, Gemmini and Tensix tiles (operations.md:34, operations.md:106, S8.2).
- Accumulation is a read-modify-write of one region (instructions.md:26, S5.2); the in-place mark ties the output to the accumulator input (S5.5).
- Configuration ops fit as state-only instructions with constraints and ordered updates: AMX `ldtilecfg` as one instruction, SME streaming mode, Gemmini `config_ex`, AIE-ML shift, Eyeriss mapping (instructions.md:29, instructions.md:42-46, S5.4).
- Shapes and region sizes may depend on attributes and state through backtick expressions, so run-time configured tile shapes work without per-shape instructions (semantics.md:62, expressions.md:13-15, S6.1).
- Layout changes on load and store compose from `reshape`, `transpose`, `bitcast_convert`, `pad`, and `slice`, including fixed register fragment layouts (mini_trn.py:29-39, S8.1).
- Fixed-point epilogues (shift, round, saturate, scale) compose from existing ops without new constructs (operations.md:85, operations.md:90).
- The S1.3 table lists every case v3 turns into a rejection, a fault, or a defined value (overview.md:31-46), so the no-undefined-behavior rule has a complete list to replace.
