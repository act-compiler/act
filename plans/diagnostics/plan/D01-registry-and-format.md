# D1 Registry, classes and message format

- What: create `taidl/diagnostics.json` and the Python classes of `scheme.md` (`TaidlError` and its eight subclasses) with one formatter that prints the three-line form.
- What: seed the registry from `registry-seed.md` (RF001 to RF034, UB, UO, VO rows); each generator gets its own registry for `U`, `K`, `I`, `X`.
- What: export the registries as JSON so the build can generate `diagnostics.rs` and `diagnostics.h`.
- Benefits: one place defines every code, its rule, its template and its spec section; generators stop inventing wording.
- Caveats: breaking in the v3 release only; S9 keeps `ValueError`, `TypeError`, `SyntaxError` as bases, so callers that catch them still work.
- Caveats: the registry lives in `taidl`, so generators already depend on it (`act_backend/spec_checks.py:11` imports `taidl.antlr4`).
- Fail points: a code renumbered or reused; noticed by a test that diffs the registry against its last released copy and rejects deletions.
- Fail points: template placeholders that no call fills; noticed by D10.
- Order and dependencies: L1, L25; before D2, D3, D6, D7, D8, D10.
- Verification: a unit test formats one code of each class; the registry loads in Python, Rust and C++ builds.
