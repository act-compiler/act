# Review of the current language and docs

This review covers the TAIDL front end and docs as they stand at `__version__ = '2.15'`.
The core design is sound: a small Python API for declarations, an XLA-HLO dialect for meaning, and an integer expression language for addresses.
Most weaknesses are unstated meaning, not wrong design.

## Strengths to keep

| component | what works | evidence |
| --- | --- | --- |
| Semantics grammar | Small, unambiguous, one line per value; reuses XLA-HLO spelling readers already know | `IDLV2.g4:6-28` |
| Expression grammar | Separate entry rule per place, precedence by rule nesting, no ambiguity | `IDLV2Expr.g4:5-24` |
| Operation table | One data table holds every word's operand count and attribute forms | `taidl/operations.py:41-165` |
| Eager checking | Each text is parsed at the call that takes it, with instruction, place, line, and column in the error | `taidl/instruction.py:331-383` |
| Reference checks | Every `@a.`, `@c.`, `@s.` is checked against declarations | `taidl/instruction.py:339-358` |
| Input and output naming | `%In<k>` and `%Out<k>` tie values to regions by position, checked both ways | `taidl/instruction.py:142-161`, `:290-307` |
| Names | One rule for every name, with a clear error | `taidl/names.py:7-14` |
| Placement marks | Defined as the constraints they append, so generators see one mechanism | `taidl/instruction.py:190-231` |
| Literal range | 64-bit range and literal-zero divisor checked at load | `taidl/instruction.py:403-437` |
| Old docs | Accurate against code; operations outside the language are listed with reasons | old `docs/operations.md` |

## Weaknesses

| component | weakness |
| --- | --- |
| Meaning of operations | Not checked by the front end and not tied to a pinned reference version (F1, F3, F4) |
| Regions | Bounds, zero sizes, overlap, and byte order were unstated (F5, F6) |
| Reference places | The `@a.` place rule is stated but not checked (F2) |
| Old docs style | Long multi-clause sentences; no section ids; facts spread across pages |
| Internal names | `parameters`, `comp_attr`, `var_type`, grammar rule `instruction` clash with the terms (F14 to F17) |
| `DataModel` | Carries two generator-only strings, `array_dim_str` and `num_dim_str` |
| `d0` | A special case in the region rule (F20) |

## Verdict per component

| component | verdict | reason |
| --- | --- | --- |
| `IDLV2.g4` | keep | Sound; small fixes F8, F13, F16 in v3 |
| `IDLV2Expr.g4` | keep | Sound; F8 only |
| `taidl/operations.py` | keep, extend | The table becomes the source of L3 shape rules |
| `taidl/instruction.py` | reshape | Add the place check (L10) and split shape checking into its own module (L12) |
| `taidl/accelerator.py` | keep | Add the order and range checks (L11, L16) |
| `taidl/data_model.py` | reshape | Rename fields; move generator strings out (L15) |
| `taidl/names.py` | keep | |
| docs | reshape | Section ids, one sentence per line, normative S1.3 and S9 |
