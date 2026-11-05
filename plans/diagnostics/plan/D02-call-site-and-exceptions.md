# D2 Call sites and exception classes in the front end

- What: every `add_*` and `set_*` call stores the caller's file and line on the object it builds, found with `inspect` as `Accelerator.__init__` does for `base_dir` (`taidl/taidl/accelerator.py:38-43`).
- What: replace the bare `ValueError`, `TypeError`, `SyntaxError` raises (23 sites in `instruction.py`, 4 in `accelerator.py`, 3 in `names.py`) with the classes of D1.
- What: generators read the stored call site, so a rejection raised in a generator still points into the ISA description.
- Benefits: closes E8; line 2 of every message names the file and line the user edits.
- Caveats: the call site of a description built in a loop or a helper is the helper's line; store the first frame outside the `taidl` package.
- Fail points: frames inside a REPL or `exec` have no file; fall back to `<string>` and say so; noticed by a test that builds a description from `exec`.
- Order and dependencies: D1; before D3, D5.
- Verification: a test per call checks the reported line equals the line of the faulty call; a generator rejection test checks the same.
