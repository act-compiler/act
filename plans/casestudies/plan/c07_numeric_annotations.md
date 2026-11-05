# C7 Numeric annotations

- What: The rounding and saturation attributes are L21 and stay an idiom; this item holds the evidence for `dot` accumulation precision, which no construct states.
- Evidence: Hopper fp8, AMX, TPU, Trainium2, and Tensix accumulate at a stated precision (gap rank 3).
- Caveats: Tolerance checks weaken bit-exact oracle tests; each annotation needs a reference meaning; mixed-type `dot` is given only by reference to StableHLO (operations.md:121).
- Fail points: Tolerances too loose hide real bugs; noticed by mutation tests that perturb results by one ulp beyond the bound.
- Order and dependencies: C10 documents the mixed-type `dot` reference first; a construct waits for a second ran site.
- Verification: Trainium2 `act_exp` and Tensix `sfpu_exp` pass oracle tests against recorded hardware outputs within a stated bound.
