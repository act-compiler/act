# C3 Instruction families and indexed state

- What: No syntax; an instruction family is a Python loop in the ISA specification, as the AMX sketch does for tile state.
- Evidence: 512 AMX `tdpbf16ps` variants and the per-shape `wgmma` and AIE-ML MAC modes (gap rank 8); indexed state has one site, AMX `tileloadd` and `tdpbf16ps` (gap rank 10).
- Caveats: The expansion size reaches the backend; that cost is a B-track item, not a language gap.
- Fail points: Expansion blows up the compile time of ACT-Backend; noticed by a timing test on the AMX specification.
- Order and dependencies: C10 carries the loop idiom; re-admission needs a second ran site for indexed state.
- Verification: The AMX specification generates its variants by loop and the front end accepts them.
