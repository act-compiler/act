# Overlap corpus

T30 turns the overlap rule set of S5.5 into cases: every declaration crossed with every placement, plus the rejection, size-defect, negative-attribute-start, and marker-versus-hand-written rows.
The cases are `program.toml` programs (T29), so attributes set the placement; the front-end rows are `case.toml` cases (T6).

## T30. Overlap matrix, rejection, and cross-check cases (S5.5, S1.3.5)

- What: the rule set under test is one relation per same-buffer input-output pair: tied, apart, exact-or-apart (the default), or may overlap; any finer rule is a written constraint.
- What: one ISA specification per declaration with a single elementwise instruction over one input and one output of a buffer `b`, size 16, starts `@a.s_in` and `@a.s_out`.
- What: a placement is a set of attribute values: apart (`s_in = 0`, `s_out = 32`), exact (`s_in = 0`, `s_out = 0`), partial (`s_in = 0`, `s_out = 8`).
- What: the semantics is order-sensitive under partial overlap (`out[i] = -in[i]`), so a stored value that matches the pre-issue state tells a snapshot read from any write-order model.
- What: the matrix has fifteen cells; a value cell stores the result of the semantics on the pre-issue state, and a fault cell expects `code`, clause S1.3.5, and unchanged buffers and state.

| declaration | apart | exact | partial |
| --- | --- | --- | --- |
| none (default exact-or-apart) | value | value | fault |
| `set_apart` | value | fault | fault |
| `set_no_partial_overlap` | value | value | fault |
| `set_inplace` | fault | value | fault |
| `set_may_overlap` | value | value | value |

- What: `set_no_partial_overlap` gives the same row as none; its cases pin it as the explicit spelling of the default.
- What: `set_inplace` apart faults because the tie requires equal starts and sizes; its exact value equals the apart value of the other rows.
- What: dimension rows use a two-dimensional region: overlap in one dimension and apart in the other is apart (value under `set_apart`); partial in one dimension and equal in the other is partial.
- What: each cell is named `overlap-<declaration>-<placement>` and cites S5.5 and, for a fault, S1.3.5; the S1.3.6 row of `norms.md` points at these cells.
- What: rejection rows are `rejected` cases in phase `build`, each citing its S9 clause: a tie across buffers, a tie whose size texts differ, `set_apart` on identical region text, and identical input and output text without `set_inplace`.
- What: size-defect rows hold equal starts and unequal sizes (`s_in = s_out = 0`, input size 16, output size 8): a fault under the default and a value under `set_may_overlap`, the narrowing model.
- What: a size-defect row on `set_inplace` with unequal size text is rejected at build, not faulted at issue.
- What: negative-attribute rows use start texts that differ and evaluate equal, as `1024 - @a.addr_out` against `@a.addr_in` with a negative `addr_out`: a value under the default, since the rules read evaluated starts and no attribute name.
- What: a negative attribute whose evaluated start falls below 0 faults on the bounds constraint (S1.3.3), not on overlap.
- What: written-constraint rows hold a positional rule (RVV widening, `@a.s_out >= @a.s_in + 16`) and an input-input rule (AMX distinctness, `@a.s1 != @a.s2`); each faults when it fails and is a value when it holds.
- What: cross-check rows run each marker and its hand-written expansion (`same`, `apart`, `same or apart`) over a swept attribute range and require the same fault set and the same values.
- What: a lint row expects a warning on an unmarked same-buffer input-output pair of an instruction whose semantics is not elementwise, with unchanged semantics.
- Benefits: every declaration meets every placement once, so a marker that widens or narrows the permitted set fails a named cell.
- Caveats: the cells use one instruction shape; a multi-output instruction gets the default on each pair, and its marker rows wait for a site in an in-scope ISA specification.
- Caveats: `set_may_overlap` is a working name; a rename edits the slugs and nothing else.
- Fail points: a cell whose value matches both a snapshot and a write-order model; the order-sensitive semantics and the partial placement rule it out.
- Order and dependencies: after T5, T6, T29, and D8; before T26 reports S1.3.6; the rule set comes from L.
- Verification: the matrix has fifteen cell cases and every row above has a case; the coverage report shows S5.5 and S1.3.5 above zero.
- Breaking: no.
- Files: new cases under `taidl/conformance/`; seeds from `taidl/tests/test_instruction.py:269-293` and :294-440, `act-backend/tests/test_refusals.py:383-396`, and `act-backend/tests/test_extraction.py:441-563`, each fact kept as a cell or row.
