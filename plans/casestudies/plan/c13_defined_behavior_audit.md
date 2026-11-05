# C13 Defined-behavior audit of the case studies

- What: List every S1.3 case a sketch reaches and give each its v3 rule: a rejection, a fault, or a defined value, under the owner one-norm rule.
- Cases: constraint violations used as synchronization, the byte order of typed views, ZA access after `smstop` (a fault), shift amounts out of range, and overlapping strided outputs (a fault, S1.3.5).
- Constraint as wait: sound only for a sequential program order; the fault it produces is named.
- Caveats: A fault at run time needs every generator to check it; some cases may need a defined value instead.
- Fail points: A case left implicit in one sketch survives into v3; noticed when the oracle meets an instruction whose result the specification does not give.
- Order and dependencies: L, F, O, T; C11 supplies the sketches.
- Verification: Each S1.3 row hit by a sketch has a v3 rule and an oracle test that triggers it.
