# Verification review: response

Rows name the review's row id (`C<n>`, `D<n>`) or its section.

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| Questions table: size and constraint errors for `@s.` do not name the state | accepted | `spec_checks.py:287-290,364-367` say only "not supported" | coverage-table.md (`@s.` row); B27 |
| C5: old binary deleted first; one message names the buffer; cost library error is `FileNotFoundError` | accepted | `generator.py:90-98,117`; `spec_checks.py:297-305` | coverage-table.md (intro) |
| C12: a unit dimension past 64 bits is rejected too | accepted | `spec_checks.py:297-305` loops `var_dim`, addressed plus unit | coverage-table.md (buffer row) |
| C33: state update unchecked on a dropped instruction | accepted | `generator.py:119-122`; the row already says so | none |
| C35: the unread-value check covers inputs | accepted | `spec_checks.py:186-192` | coverage-table.md (semantics lines row) |
| C37: six element types accepted on a buffer `var_type` | accepted | `spec_checks.py:47-51` checks semantics lines only | coverage-table.md (new buffer row) |
| C44: float literals not range-checked | accepted | `spec_checks.py:106-107` | coverage-table.md (split row); B20 |
| D2: effect unverifiable | accepted | The row already says the effect is not verified | none |
| D7: reachability unverifiable | accepted | The row already says reachability is not verified | none |
| D9: "refuse" is in docstrings, not messages | accepted | `spec_checks.py:1,23,180` | plan/bugs.md D9 |
| D11: the mutation is at `slow.rs:24` | accepted | `slow.rs:24` | plan/bugs.md D11; B9 |
| Missed: `parameter` has no row | accepted | `spec_checks.py:53-55` | coverage-table.md (new row); plan/coverage.md intro |
| Missed: `concatenate` takes exactly two operands | accepted | `isel_ir2isa_rewrites_txt_generator.py:15`; `egraph.rs:13` | coverage-table.md (new row); B17 |
| C40 needs a `partial` row for `concatenate` | accepted | Same evidence | coverage-table.md (operation rows) |
| Missed: `@a.` in a size is rejected | accepted | `spec_checks.py:274-292` | coverage-table.md (new row) |
| Missed: constraint whose integers pass 64 bits is rejected | accepted | `spec_checks.py:346-348` | coverage-table.md (new row) |
| Missed: buffer `var_type` of the six narrow types accepted | accepted | `spec_checks.py:47-51` | coverage-table.md (new row) |
| Missed: float `constant` literals not range-checked | accepted | `spec_checks.py:106-107` | coverage-table.md; B20 |
| Missed: rank-0 shape has no row | accepted | `egraph.rs:406` treats an empty shape as unset | coverage-table.md (new row); B26 test |
| Missed: S1.3 implementation-defined cases | accepted | S1.2 asks each generator to document them | coverage-table.md (new section); B15 |
| S4.5 and S9 need no backend row | accepted | Front-end rules | coverage-table.md (Outside this table) |
| S8.5 needs no row | accepted | Outside the language | coverage-table.md (Outside this table) |
| One-element `broadcast` guard assumes the previous line | accepted | `isel_ir2isa_rewrites_rs_generator.py:348-350` | plan/bugs.md D13; coverage-table.md; B6 |
