# D6 Unsupported-feature registries

- What: give each generator's unsupported rows `U` codes (UB001 to UB020, UO001 to UO003 in `registry-seed.md`) and one exception, `UnsupportedError`, raised by a rejection.
- What: mark each row as an exclusion (the instruction is left out, severity `warning`) or a rejection (D18); the oracle has no exclusion.
- What: each row cites the spec section of S1.2 and carries the status `partial`, `excluded`, or `rejected` the spec defines.
- What: generate the generator's unsupported-feature table (`docs/coverage.md`) from the registry.
- What: a backend never assumes an unsupported input away; each such input is a coded `U` row, an exclusion or a rejection.
- What: use one phrase, `<feature> is unsupported by <generator>`, in place of the seven current phrasings (`analysis/wording.md`).
- Benefits: a generator states clearly which spec features it does not support; closes E7 and the unsupported part of E13.
- Caveats: the backend's rejections come from three layers (Python, Rust reader, C++), so the registry feeds all three.
- Fail points: a feature rejected in code but missing from the table; noticed by the coverage test (B15, O17) that fails on an unlisted code.
- Order and dependencies: D1, L7, B15, O14, O17.
- Verification: for each `U` code a test builds the smallest description that triggers it and asserts the code.
