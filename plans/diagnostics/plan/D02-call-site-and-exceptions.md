# D2 Call sites and exception classes in the front end

- What: every node of the model records the author's file and line, the first frame outside the `taidl` package; the 2.x string front end stores it on each `add_*` and `set_*` call, found with `inspect` as `Accelerator.__init__` does for `base_dir` (`taidl/taidl/accelerator.py:38-43`).
- What: a whole-instruction problem uses the first line of the instruction's function.
- What: replace the bare `ValueError`, `TypeError`, `SyntaxError` raises (23 sites in `instruction.py`, 4 in `accelerator.py`, 3 in `names.py`) with the classes of D1.
- What: generators read the stored call site, so a rejection or an exclusion raised in a generator still points into the ISA specification; the call site is a typed field of the `Model` that `check` returns (U3).
- What: rename the alias `Fault` (taidl/taidl/operations.py:167-168) to `Rejection`, the exception `check` raises with one entry per problem.
- Benefits: closes E8; line 2 of every message names the file and line the user edits.
- Caveats: the call site of a description built in a loop or a helper is the helper's line; store the first frame outside the `taidl` package.
- Fail points: frames inside a REPL or `exec` have no file; fall back to `<string>` and say so; noticed by a test that builds a description from `exec`.
- Order and dependencies: D1, U3; before D3, D5.
- Verification: a test per call checks the reported line equals the line of the faulty call; a generator rejection and an exclusion test check the same.
