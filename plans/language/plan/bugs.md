# Bugs, divergences, and spec ambiguities

One row per issue in the front end, its docs, or the spec draft; ids are LB1 to LB29.
Severity: high means two generators can disagree or a wrong ISA specification is accepted; medium means a rule is checked loosely; low means naming or messages.
No code was changed; each fix is a plan item.

| ID | Severity | Where (file:line) | Bug | Fix sketch |
| --- | --- | --- | --- | --- |
| LB1 | high | `taidl/operations.py:3-4` | No shape or element type check of a semantics; generators decide alone | L3 rules, then the U4 checks |
| LB2 | high | `taidl/instruction.py:339-358` | `@a.` is accepted in sizes, backtick expressions, and updates; only declared-ness is checked | Check reference kind against place (L10) |
| LB3 | high | `docs/operations.md` S8.4 | Generators may start a reduction elsewhere than the identity S8.4 states | Test B and O against S8.4 (L4) |
| LB4 | high | `docs/operation_meaning.md` S8.4 | Generators may not fill a non-scalar `constant` as S8.4 states | Test B and O (L4) |
| LB5 | high | O and B | Region bounds, sizes below 1, and output overlap must fault (S1.3.3, S1.3.4) | L5 |
| LB6 | high | O | `d0` byte layout must follow S8.4 for multi-byte, sub-byte, and `pred` values | Test O's packing (`utils.py:880-908`) against S8.4 (L4) |
| LB7 | medium | `taidl/accelerator.py:33-44` | `add_data_model` has no order check, unlike `add_initial_state` at `:46-52` | Same check (L11) |
| LB8 | medium | `IDLV2.g4:89`, `IDLV2Expr.g4:59` | Semantics `INT` allows leading zeros, expression `INT` does not | Gone with the grammars (L45) |
| LB9 | medium | `taidl/instruction.py:296-297` | `parameter(01)` is accepted and read as input 1 | Falls out of LB8 (L13) |
| LB10 | low | `taidl/instruction.py:410-415` | The literal-zero divisor check matches only the token `0`, which is now the spec's rule | Test only (L14) |
| LB11 | medium | `taidl/accelerator.py:55` | An initial value outside signed 64 bits, or `True`, is accepted | Range and type check (L16) |
| LB12 | medium | `taidl/instruction.py:419-437` | `constant` literals are not range-checked by element type | Range check by type (L16) |
| LB13 | medium | `IDLV2.g4:94`, `IDLV2Expr.g4:66`, `taidl/names.py:9` | Semantics `IDENTIFIER` allows dots, expression names and `check_name` do not | Gone with the grammars; names follow S4.1 (L45) |
| LB14 | low | `taidl/instruction.py:36`, `:71` | `Instruction.parameters` holds the addressing attributes, while `parameter` is also a tensor operator | Rename (L15) |
| LB15 | low | `taidl/accelerator.py:62`, `taidl/instruction.py:35` | `computation_attr` vs `comp_attr` vs the term "computational attribute" | Rename to `computational_attrs` (L15, L30) |
| LB16 | low | `IDLV2.g4:8` | Grammar rule `instruction` names a semantics line | Gone with the grammars (L45) |
| LB17 | low | `taidl/data_model.py:34-40` | `num_dim_str` serves no generator; the other field names differ from the glossary | Delete `num_dim_str` (L15); renames in the L30 table |
| LB18 | low | `taidl/instruction.py:311` | ROOT check defaults the last output to `%Out0` when no `%Out<k>` line exists, so the message names a missing value | Message only; S9 lists the case |
| LB19 | low | `docs/*.md`, error texts | "refuse" and "data model" used where the glossary says "reject" and "buffer" | L26 |
| LB20 | medium | `taidl/accelerator.py:31`, `taidl/instruction.py:123` | `d0` is declared with no access dimension, then given one region start and size by a special case | One access dimension (L30) |
| LB21 | high | `taidl/instruction.py:163-172` | `set_inplace` checks only the flag count, not that the input matches output region 0 | `set_inplace` expands to `same(In k, Out)` and rejects a cross-buffer pair or unequal size text (L5, L36) |
| LB22 | medium | O | `d0` size must be the byte count of the `hbm` contents (S5.6) | O's `hbm` argument already sizes it; test in L5 |
| LB23 | low | `tests/test_grammar.py:317-342` | Doc tests pin a heading ("What an operation takes"), the text before the first heading, and the sentence "A type is one of", so the spec cannot carry section ids in those headings | Anchor tests on section ids (L8) |
| LB24 | medium | `taidl/instruction.py:136` | A start that evaluates below 0 is accepted as text; S1.3.3 makes it a fault at issue, while a negative attribute value stays legal | Test in L5 |
| LB25 | medium | `taidl/instruction.py:452-456` | `strings` checks the list but not that each entry is a `str` | L16 |
| LB26 | medium | `taidl/instruction.py:168`, `:203` | Mark flags are read by truthiness, so a non-`bool` flag is accepted | L16 |
| LB27 | medium | `taidl/instruction.py:99-100`, `:254-257` | An instruction with regions but no semantics, or the reverse, is never checked | L11 |
| LB28 | medium | `IDLV2.g4:23` | A literal shape dimension 0 is accepted | L16 |
| LB29 | high | `taidl/instruction.py:211-224` | `keep_off` compares starts only, so equal starts with unequal sizes pass as exact | Add the size term to `same` (L36) |
