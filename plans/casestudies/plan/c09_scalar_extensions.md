# C9 Float attributes and bit operators

- What: No syntax; the `u32` bits route carries a float attribute, and `/` and `%` unpack packed fields.
- Evidence: Float scales at Trainium2 and Gemmini (gap rank 12); packed fields at AMX, Hopper, and Gemmini (gap rank 13); both routes are lossless.
- Caveats: Re-admission needs an idiom failure, for example a float comparison in a constraint.
- Fail points: A bits route rounds differently between oracle and backend; noticed by a test that round-trips scale values.
- Order and dependencies: C10 carries both idioms.
- Verification: Gemmini `mvout_acc` and Trainium2 `act_exp` take a scale through the bits route and match a reference.
