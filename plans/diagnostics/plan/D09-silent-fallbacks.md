# D9 Silent fallbacks become errors

- What: remove or reject the fallbacks of `analysis/fallbacks.md` that D4 and D7 do not cover: a second `set_inputs` (E-row in `fallbacks.md`), a kernel line outside a computation (E23), the backend's skip of instructions with no semantics (E1) once D4 rejects them.
- What: document the fallbacks that stay (last line is root when no ROOT, `base_dir`) as one defined sentence each in the generator or front-end docs.
- Benefits: nothing passes unremarked; the rest is stated.
- Caveats: rejecting a second `set_inputs` is breaking; the alternative is a documented replace.
- Fail points: a fallback found later; noticed by D10's lint that greps for `debug_assert`, bare `return None` in checks and `except: pass`.
- Order and dependencies: D1, B24, B26, L5.
- Verification: a test per removed fallback asserts the new code.
