# D8 Oracle run-time and call diagnostics

- What: give the call-time rows of `catalog/act-oracle.md` codes: `VF001` to `VF010` (`VF007` unused) and `KO001` to `KO020`.
- What: add the instruction or kernel name where missing (`generic/utils.py:555`, `decorator.py:21, :150`, `utils.py:675-692`) (E15).
- What: turn `warn_width` (`generic/utils.py:268-271`) into a `VF005` error (E13), and check the implicit apart constraints and each pair's relation as `VF001`, loading all inputs before any store.
- What: raise `FaultError` and `InputError`; keep the `at i = v` suffix.
- What: add the faults of the norm that no code checks yet: an on-chip read of an element no earlier issue wrote (`VF008`, a written-set mask per buffer), and a backtick constant outside its element type (`VF009`).
- What: a test per `V` code asserts that the fault wrote no region and no state.
- Benefits: a kernel author sees one format for every failure of a run; the run-time norm is the same for every generator.
- Caveats: breaking for kernels that ran with a 64-bit warning; migration note.
- Fail points: iteration environments that grow long for nested loops; cap at the three innermost variables and say so.
- Order and dependencies: D1, D14, L6, O6, and the unroll outcome, O20 or O11 (O11 lands only if O19 misses the Q5.1 bound).
- Verification: `tests/test_kernel.py` and `tests/test_constraints.py` pass with code assertions; one test per `V` and `K` code.
