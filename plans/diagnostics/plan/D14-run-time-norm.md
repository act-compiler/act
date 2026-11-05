# D14 Fault table and run-time norm

- What: write the one norm of `scheme.md` (every fault is a defined error that writes no region and no state) into S1.3, and list each fault with its `VF` code: region outside its data model, size below 1, zero divisor, an operator outside signed 64 bits, failed constraint, value shape differing from its region, read of an unwritten on-chip element, backtick constant outside its element type.
- What: state for each whether the oracle raises it, the backend rejects it, or the front end rejects it for literals.
- What: state which faults the program can decide from attributes, state, and region text; tensor data never faults.
- What: overlap rows fold into the failed-constraint row (`VF001`, S1.3.5): outputs apart, each input-output pair equal-or-apart by default, markers as declared; a marked pair across buffers is rejected (owner decision 24).
- Benefits: no "undefined" row remains; D4 and D8 implement a closed list.
- Caveats: an operator that leaves signed 64 bits is a fault, not a wrap (Q1.2).
- Fail points: a fault missing from the list; noticed by a test that mutates a case study to break each rule and expects the matching code.
- Order and dependencies: L5, L6.
- Verification: the S1.3 table has no row without a code; D8's tests cover every `V` code.
