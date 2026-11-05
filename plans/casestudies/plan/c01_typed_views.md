# C1 Typed-view idiom and byte order

- What: Document `u8` plus `bitcast_convert` as the way a region gives its own element type over byte storage; the little-endian byte order comes from F6.
- Benefits: Keeps the verbosity of 7 case studies (gap rank 5) out of the grammar; a typed-view syntax is verbosity, not an idiom failure (Q2.3).
- Caveats: S7.2 must state the result of a view whose type width does not divide the base element shape in bytes.
- Fail points: Generators disagree on byte order; noticed when the test oracle and the assembly code differ on a mixed-type load test.
- Order and dependencies: F6, then C10 carries the idiom text.
- Verification: The TPU, Trainium2, and AMX sketches run under the oracle with the fixed byte order and give the same results on a little-endian test vector.
