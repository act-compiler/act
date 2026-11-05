# D4 Move run-time-independent checks into the front end

- What: add the checks RF040 to RF050 of `registry-seed.md`: output regions without semantics, an end-of-description completeness check, region value type and element count against the region, `constant` range, folded zero divisors, 64-bit dimensions and initial values, literal sizes below 1, `@a.` outside starts and constraints, shape and element-type agreement, element type per word.
- What: run the whole-description checks from the one `check(acc)` function that every generator entry calls once and that the user can call as `python -m taidl check spec.py` (U3, U10).
- What: `check` collects every problem and raises one `Rejection`; entries come in a fixed order (declarations, regions, constraints, bodies), so a description with several faults gives the same first entry in every generator (closes the order problem of `analysis/late.md`).
- What: the standalone command prints each entry in the two-line form and exits 1.
- Benefits: closes E1, E2, E5, E6, E14; the same description is accepted or rejected alike by every generator; an architect fixes several faults in one run.
- Caveats: breaking; descriptions that passed in v2 are rejected, so each new row needs a migration note.
- Caveats: shape agreement (RF049) is the large part and belongs to U4; this item attaches the codes RF040 to RF050 to U4's checks.
- Caveats: dims over `@c.` are decided at the oracle's call and at backend selection (`VF010`), not by this item.
- Fail points: a new rejection fires on a case study (C track) that is valid; noticed by running every spec in `tests/` and the tutorial specs through `check` before enabling.
- Order and dependencies: D14, D3, U4, L3, L5, L10, L14, L16, B24.
- Verification: a corpus verdict per RF040 to RF050 (T6 runner); both generators' suites pass after D5 deletes their copies; a specification with five faults gives five entries.
