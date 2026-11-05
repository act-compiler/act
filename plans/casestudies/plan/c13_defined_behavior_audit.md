# C13 Defined-behavior audit of the case studies

- What: List every undefined or implementation-defined case of S1.3 that a case-study sketch reaches, and give each its v3 defined error or defined behavior under the owner's one-norm rule.
- Benefits: Grounds the no-undefined-behavior rule in real ISAs: constraint violations used as synchronization, the byte order of typed views, ZA after `smstop`, shift amounts out of range, and overlapping strided outputs.
- Caveats: A defined error at run time needs every generator to check it, which costs the compiler backend; some cases may need a defined value instead.
- Fail points: A case left implicit in one sketch survives into v3; noticed when the oracle meets an instruction whose result the spec does not give.
- Order and dependencies: L and F, then O, then T; C11 supplies the sketches.
- Verification: Each S1.3 row hit by a sketch has a v3 rule and an oracle test that triggers it.
