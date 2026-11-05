# D4 Move run-time-independent checks into the front end

- What: add the checks RF040 to RF050 of `registry-seed.md`: output regions without semantics, an end-of-description completeness check, region value type and element count against the region, `constant` range, folded zero divisors, 64-bit dimensions and initial values, literal sizes below 1, `@a.` outside starts and constraints, shape and element-type agreement, element type per word.
- What: run the whole-description checks from one `Accelerator.validate()` that every generator calls first and that the user can call.
- What: report faults in a fixed order (declarations, regions, texts, semantics), so a description with several faults gives the same first message in every generator (closes the order problem of `analysis/late.md`).
- Benefits: closes E1, E2, E5, E6, E14; the same description is accepted or rejected alike by every generator.
- Caveats: breaking; descriptions that passed in v2 are rejected, so each new row needs a migration note.
- Caveats: shape agreement (RF049) is the large part and belongs to L12; this item wires it into `validate()`.
- Fail points: a new rejection fires on a case study (C track) that is valid; noticed by running every spec in `tests/` and the tutorial specs through `validate()` before enabling.
- Order and dependencies: D14, D3, L3, L5, L10, L12, L14, L16, B22, B24.
- Verification: a corpus verdict per RF040 to RF050 (T6 runner); both generators' suites pass after D5 deletes their copies.
