# D8 Oracle run-time and call diagnostics

- What: give the call-time rows of `catalog/act-oracle.md` codes: `VO001` to `VO007` and `KO001` to `KO020`.
- What: add the instruction or kernel name where missing (`generic/utils.py:555`, `decorator.py:21, :150`, `utils.py:675-692`) (E15).
- What: turn `warn_width` (`generic/utils.py:268-271`) into a `VO005` error (E13), and check overlapping outputs (`VO007`).
- What: raise `ViolationError` and `InputError`; keep the `at i = v` suffix.
- Benefits: a kernel author sees one format for every failure of a run; the run-time norm is the same for every generator.
- Caveats: breaking for kernels that ran with a 64-bit warning; migration note.
- Fail points: iteration environments that grow long for nested loops; cap at the three innermost variables and say so.
- Order and dependencies: D1, D14, L6, O6, O11.
- Verification: `tests/test_kernel.py` and `tests/test_constraints.py` pass with code assertions; one test per `V` and `K` code.
