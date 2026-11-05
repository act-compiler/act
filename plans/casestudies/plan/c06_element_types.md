# C6 Block-scaled and added element types

- What: Add `tf32` and the remaining 8-bit floats, and a block-scaled form (MX formats, `bfp8`) as a storage type plus scale, or as `scaled_dot`.
- Benefits: Tensix tiles, Hopper `tf32`, and current MX hardware become direct (gap rank 4).
- Caveats: StableHLO has no block-scaled tensor type; the spec must define element order and scale placement itself.
- Fail points: Rounding of the shared exponent differs from hardware; noticed by bit-exact tests against vendor reference encoders.
- Order and dependencies: P, then L and F, then O, then B.
- Verification: Tensix `unpack_b_bfp8` written with the new type equals the integer-decoding sketch.
