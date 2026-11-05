# D15 Removed-names rejection

- What: one removed-names table lists each removed v2 member and tensor operator with its replacement and section id.
- What: a class-level `__getattr__` and the `OPERATIONS` lookup read the table and raise one coded `RF051` rejection; no stub per member and no converter in `taidl` 3.
- What: seed the table after the v3 API is fixed; `set_inplace` stays and is not a row (owner decision 30).
- What: the row for `add_semantics` and the other string-taking calls names the converter script on the 2.x line (U13); the table lists no other converter.
- Benefits: a v2 call raises a coded message with the fix, not a bare `AttributeError` (`set_inplace` and `set_apart` are methods, taidl/taidl/instruction.py:163, :174).
- Caveats: the table is a tested feature, one test per row; the migration note of U8 lists the same rows.
- Fail points: a row naming a replacement that is itself renamed; noticed by a test that resolves each replacement in the v3 API.
- Order and dependencies: D1, U2; after L15; the converter row waits for U13.
- Verification: each row raises `RF051` with the replacement and section id in the message.
