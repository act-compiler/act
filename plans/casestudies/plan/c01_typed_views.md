# C1 Typed views of storage

- What: Let a region give its own element type over a byte or untyped data model, as `[dm, starts, sizes, type]`, with a fixed little-endian byte order.
- Benefits: Removes the `u8` plus `bitcast_convert` pattern from 7 of 10 case studies (gap rank 1); fixes the byte order that S1.3 leaves implementation-defined.
- Caveats: The data model element type (S4.3) becomes a default; S7.2 must define a view whose type width does not divide the unit shape in bytes.
- Fail points: Generators disagree on byte order; noticed when the TAIDL-TO oracle and the compiled kernel differ on a mixed-type load test.
- Order and dependencies: L, F, then B and O, then T.
- Verification: Rewrite the TPU, Trainium2, and AMX sketches with views and check the oracle results equal the `bitcast_convert` versions bit for bit.
