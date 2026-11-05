# C9 Float attributes and bit operators

- What: Float-typed computational attributes and state values, and `&`, `|`, `<<`, `>>` in expressions.
- Benefits: Removes the `u32` bits route for scales and the `/`, `%` unpacking of packed fields (gap ranks 11 and 14).
- Caveats: Floats in constraints need comparison rules; S6.2 overflow rules extend to shifts.
- Fail points: Float attribute rounding differs between oracle and backend; noticed by a test that round-trips scale values.
- Order and dependencies: L, then S, then B and O.
- Verification: Gemmini `mvout_acc` and Trainium2 `act_exp` take a float attribute and match the bits route.
