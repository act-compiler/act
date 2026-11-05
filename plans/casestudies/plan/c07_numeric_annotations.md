# C7 Numeric annotations

- What: Operation attributes for `dot` accumulation type and precision, an accuracy bound for transcendentals, and a rounding mode for `convert`.
- Benefits: States the hardware numerics seen in 5 case studies; lets the oracle check within tolerance instead of failing on bits (gap rank 6).
- Caveats: Tolerance checks weaken bit-exact oracle tests; each annotation needs a reference meaning.
- Fail points: Tolerances too loose hide real bugs; noticed by mutation tests that perturb results by one ulp beyond the bound.
- Order and dependencies: L, then O, then D.
- Verification: Trainium2 `act_exp` and Tensix `sfpu_exp` pass oracle tests against recorded hardware outputs within the stated bound.
