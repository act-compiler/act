# C6 Block-scaled and added element types

- What: The extension is L17 (`f4e2m1fn`, `f8e8m0fnu`); this item holds the case-study evidence and the verification.
- Evidence: Blackwell and NeuronCore-v4 are the two public MX sites; Tensix `bfp8` has one site and stays an idiom; `tf32` is not a storage type, since `reduce_precision` models it.
- Caveats: `scaled_dot` stays an idiom, exact for power-of-two scales; the NKI scale group of 8 partitions by 4 elements differs from a 32-element block.
- Fail points: Rounding of the shared exponent differs from hardware; noticed by bit-exact tests against vendor reference encoders.
- Order and dependencies: L17 first.
- Verification: Tensix `unpack_b_bfp8` and an MX `dot` idiom run under the oracle with the two added types.
