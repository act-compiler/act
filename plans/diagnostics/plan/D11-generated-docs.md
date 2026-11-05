# D11 Generate S9 and the unsupported tables

- What: render S9 (`taidl/docs/conformance.md`) from the registry, with the code beside each row.
- What: render each generator's `docs/coverage.md` unsupported rows from its registry.
- Benefits: the spec, the code and the tests cannot disagree; the S9 "Not rejected" section shrinks as D4 lands.
- Caveats: S9 is prose-edited today; the generator must keep the page's one-sentence-per-line style.
- Fail points: a hand edit overwritten; the file header says generated and L8's consistency check fails on a diff.
- Order and dependencies: D3, D6, L7, L8, O14, B15.
- Verification: regenerating produces no diff in CI.
