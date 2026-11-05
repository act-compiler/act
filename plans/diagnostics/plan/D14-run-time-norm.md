# D14 Run-time violation table and norm

- What: write the one norm of `scheme.md` (every run-time violation is a defined error) into S1.3, and list each violation with its `V` code: region outside its data model, size below 1, zero divisor, integer outside signed 64 bits, overlapping outputs, failed constraint, value shape differing from its region.
- What: state for each whether the oracle raises it, the backend rejects it, or the front end rejects it for literals.
- What: state the in-place rule (output reuses the marked input's storage) as `R` for equal literal regions and `V` otherwise.
- Benefits: no "undefined" row remains; D4 and D8 implement a closed list.
- Caveats: owner decision on overflow (error versus wrap) is open; the default is error.
- Fail points: a violation missing from the list; noticed by a test that mutates a case study to break each rule and expects the matching code.
- Order and dependencies: L5, L6.
- Verification: the S1.3 table has no row without a code; D8's tests cover every `V` code.
