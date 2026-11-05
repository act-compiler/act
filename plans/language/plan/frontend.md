# Plan: front-end fixes

These items bring the `taidl` front end (`taidl/taidl/*.py`) in line with the spec, and rename the confusing internals.
Each comes from reading the code against the spec and fixes rows F1 to F23.
No item changes what a valid ISA description means; most turn an undetected error into a rejection.
Sizes assume two engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L10 | Reject `@a.` outside starts and constraints | L1 | S | Q1 |
| L11 | Reject a data model declared after an instruction | L1 | S | Q1 |
| L12 | Shape and element type checker | L3, O7 | L | Q2 |
| L13 | Align integer literal lexing between the two grammars | L1 | S | Q1 |
| L14 | Catch every literal-zero divisor | L1 | S | Q1 |
| L15 | Rename confusing internal names | L25 | M | Q2 |
| L16 | Range-check `constant` literals by element type | L3 | S | Q2 |

## L10 Reject `@a.` outside starts and constraints

- What: `parse_text` (`instruction.py`) checks a reference's kind against its place: `@a.` only in a start or a constraint.
- Changes: `Instruction.parse_text` in `taidl/instruction.py` gains a place argument check; the declared-name check is kept.
- Breaking: yes, specs accepted today are rejected.
- Benefits: removes one S1.3 undefined row; B can rely on sizes being address-free.
- Caveats: may reject an ISA description that works today in O.
- Fail points: an example in A tutorials breaks; notice by `test_examples.py` and A's tutorial run.
- Order and dependencies: after L1; tell B and O; A checks tutorials.
- Verification: new tests per place; S9 row moves from "not rejected" to the texts table.

## L11 Reject a data model declared after an instruction

- What: `add_data_model` raises `ValueError` once an instruction exists, as `add_initial_state` already does.
- Changes: `Accelerator.add_data_model` in `taidl/accelerator.py`; its other checks are kept.
- Breaking: yes.
- Benefits: matches the declaration order S4.5 states.
- Caveats: none known in the examples.
- Fail points: a user ISA description interleaves declarations; notice by its error message.
- Order and dependencies: after L1.
- Verification: a test like the existing state-order test.

## L12 Shape and element type checker

- What: an inference pass over each semantics using the L3 table, rejecting lines whose type or shape disagree.
- Changes: new `taidl/shapes.py`, called from `Instruction.add_semantics`; `check_operations` is kept.
- Breaking: yes, specs accepted today are rejected.
- Benefits: removes the largest undefined row in S1.3; every generator gets the same accepted set.
- Caveats: shapes with backtick expressions need symbolic comparison or a check per attribute valuation.
- Fail points: a false rejection of a valid ISA description; notice by running every A, B, O example.
- Order and dependencies: after L3; O's `shapes.py` may move into the front end; coordinate with O.
- Verification: every row of L3 has a passing and a failing test.

## L13 Align integer literal lexing

- What: decide leading zeros: semantics `INT` allows `007`, expression `INT` rejects it.
- Changes: `INT` and `IDENTIFIER` in `IDLV2.g4`; every other rule is kept.
- Breaking: yes, `007` handling changes.
- Benefits: one rule for integers in both grammars.
- Caveats: a grammar change needs regeneration with ANTLR 4.9.2.
- Fail points: generated files out of date; notice by L8.
- Order and dependencies: after L1; batch with other grammar edits.
- Verification: grammar tests for `007` in both places.

## L14 Catch every literal-zero divisor

- What: the literal-zero check compares the next sibling's text to `0`, so `x / (0)` and `x / -0` pass.
- Changes: `expression_literal_fault` in `taidl/instruction.py`; the 64-bit check is kept.
- Breaking: yes.
- Benefits: the S9 rule holds as written.
- Caveats: or narrow S9 to "the literal `0` directly".
- Fail points: none beyond the test.
- Order and dependencies: after L1.
- Verification: tests for `(0)` and `-0`.

## L15 Rename confusing internal names

- What: `Instruction.parameters` holds addressing attributes; `comp_attr` vs `computation_attr`; `DataModel.access_dim`, `var_type`, `var_name`; grammar rule `instruction` is a semantics line.
- Changes: field names in `taidl/instruction.py` and `taidl/data_model.py`, grammar rule `instruction`; the public `add_*` and `set_*` API is kept.
- Breaking: yes, front-end attribute and grammar rule names change for B and O.
- Benefits: code reads with glossary terms.
- Caveats: B and O read these attributes, so they update in the same v3 release, with no aliases kept.
- Fail points: a generator breaks on a renamed attribute; notice by B and O test suites.
- Order and dependencies: after L25; B and O update together.
- Verification: grep for old names in all four repos is empty.

## L16 Range-check `constant` literals by element type

- What: `constant(300)` on `u8` is accepted by the front end; O checks it at generation.
- Changes: `add_initial_state` and `semantics_literal_fault`; their other checks are kept.
- Breaking: yes.
- Benefits: rejection at the call, the same for every generator.
- Caveats: float range per type needs care for `f8` types.
- Fail points: drift from O's `mlir_literal`; notice by sharing one table.
- Order and dependencies: after L3; coordinate with O.
- Verification: tests per element type.
