# C3 Instruction families and indexed state

- What: Declare one instruction over an enumerated attribute (tile index, shape mode) and allow state arrays read as `@s.rows[@c.t]`.
- Benefits: Collapses 512 AMX `tdpbf16ps` variants and the per-shape `wgmma` and AIE-ML MAC modes to one declaration each (gap ranks 7 and 10).
- Caveats: A family must expand to plain instructions for today's generators; the expansion size must stay bounded.
- Fail points: Expansion blows up compile time of ACT-Backend; noticed by a timing test on the AMX description.
- Order and dependencies: L, then S, then B and O.
- Verification: The expanded AMX family equals the hand-written variants instruction by instruction under the oracle.
