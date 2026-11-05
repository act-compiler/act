# C8 Data-dependent addressing

- What: Let a start or a state update read a rank-0 region value, for indirect DMA, descriptors in memory, and `ldtilecfg`.
- Benefits: Covers embedding gathers and descriptor-driven instructions (gap rank 8).
- Caveats: Breaks the static analysis that ACT-Backend relies on; may need a separate instruction kind.
- Fail points: The backend cannot select such instructions and silently skips them; noticed by an unsupported-feature row required by S1.2.
- Order and dependencies: L, then O, then B.
- Verification: Trainium2 indirect DMA and AMX `ldtilecfg` from memory run under the oracle.
