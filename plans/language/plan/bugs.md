# Bugs, divergences, and spec ambiguities

One row per issue in the front end, its docs, or the spec draft.
Severity: high means two generators can disagree or a wrong ISA description is accepted; medium means a rule is checked loosely; low means naming or messages.
No code was changed; each fix is a plan item.

| ID | Severity | Where (file:line) | Bug | Fix sketch |
| --- | --- | --- | --- | --- |
| F1 | high | `taidl/operations.py:3-4` | No shape or element type check of a semantics; generators decide alone | L3 table, then L12 checker |
| F2 | high | `taidl/instruction.py:339-358` | `@a.` is accepted in sizes, backtick expressions, and updates; only declared-ness is checked, while old `docs/expressions.md` called it an error | Check reference kind against place (L10) |
| F3 | high | `docs/operations.md` S8.4 | `reduce_<reducer>` initial value is stated nowhere | Fix per reducer and type (L4) |
| F4 | high | old `docs/semantics.md` | `constant` with a non-scalar shape had no stated meaning | S8.4 now says fill; O and B confirm (L4) |
| F5 | high | old `docs/declarations.md` | Region bounds, zero or negative sizes, and output overlap had no stated meaning | S1.3 names them undefined; decide in L5 |
| F6 | high | `docs/semantics.md` S7.2 | Byte order between `d0` and multi-byte types is not fixed | Pin in L2, state in S8.4 |
| F7 | medium | `taidl/accelerator.py:33-44` | `add_data_model` has no order check, unlike `add_initial_state` at `:46-52` | Same check (L11) |
| F8 | medium | `IDLV2.g4:89`, `IDLV2Expr.g4:59` | Semantics `INT` allows leading zeros, expression `INT` does not | One rule (L13) |
| F9 | medium | `taidl/instruction.py:296-297` | `parameter(01)` is accepted and read as input 1 | Falls out of F8 (L13) |
| F10 | medium | `taidl/instruction.py:410-415` | Literal-zero divisor check compares the next sibling's text to `0`, missing `(0)` and `-0` | Evaluate constant divisors (L14) |
| F11 | medium | `taidl/accelerator.py:55` | An initial value outside signed 64 bits is accepted | Range check (L16) |
| F12 | medium | `taidl/instruction.py:419-437` | `constant` literals are not range-checked by element type | Range check by type (L16) |
| F13 | medium | `IDLV2.g4:94`, `IDLV2Expr.g4:66`, `taidl/names.py:9` | Semantics `IDENTIFIER` allows dots, expression names and `check_name` do not | Decide in L13 |
| F14 | low | `taidl/instruction.py:36`, `:71` | `Instruction.parameters` holds the addressing attributes, while `parameter` is also an operation word | Rename (L15) |
| F15 | low | `taidl/accelerator.py:62`, `taidl/instruction.py:35` | `computation_attr` vs `comp_attr` vs the term "computational attribute" | Rename (L15) |
| F16 | low | `IDLV2.g4:8` | Grammar rule `instruction` names a semantics line | Rename to `line` (L15) |
| F17 | low | `taidl/data_model.py:34-40` | `access_dim`, `var_dim`, `var_type`, `array_dim_str`, `num_dim_str` do not match the glossary; the two `_str` fields serve only generators | Rename; move strings to generators (L15) |
| F18 | low | `taidl/instruction.py:311` | ROOT check defaults the last output to `%Out0` when no `%Out<k>` line exists, so the message names a missing value | Message only |
| F19 | low | `docs/*.md`, error texts | "refuse" and "buffer" used where the glossary says "reject" and "data model" | L26 |
| F20 | medium | `docs/declarations.md` S4.3 | `d0` declares no unit shape and no addressed dimension, yet takes one start and size: a special case in the data model rule | Keep, stated in S4.3; revisit in v3 |
| F21 | high | `docs/instructions.md` S5.5 | The meaning of an in-place mark beyond "overwrites" was unstated; B relies on it | State it as storage reuse of the output over the input, with B (L5) |
| F22 | medium | `docs/declarations.md` S4.3 | `d0` has no declared size, so "past its data model" has no bound there | Implementation-defined now (S1.3); add an optional size in v3 (L5) |
| F23 | low | `tests/test_grammar.py:317-342` | Doc tests pin a heading ("What an operation takes"), the text before the first heading, and the sentence "A type is one of", so the spec cannot carry section ids in those headings | Anchor tests on section ids (L8) |
