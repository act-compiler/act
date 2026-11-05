# Plan: front-end fixes

These items bring the `taidl` front end (`taidl/taidl/*.py`) in line with the spec, and rename the confusing internals.
Each comes from reading the code against the spec and fixes rows F1 to F23.
Most turn an undetected error into a rejection listed in S9.
Sizes assume the shared 3-4 engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L10 | Reject `@a.` outside starts and constraints | L1 | S | Q1 |
| L11 | Reject late buffers and partial instructions | L1 | S | Q1 |
| L12 | Shape and element type checker | L3 | L | Q2 |
| L13 | Align integer literal lexing between the two grammars | L1 | S | Q1 |
| L14 | Test the exact-token zero divisor rule | L1 | S | Q1 |
| L15 | Rename confusing internal names | L25 | S | Q2 |
| L16 | Range and type checks at the declaring call | L3 | S | Q1 |
| L30 | Owner vocabulary in the public API and `d0` as a one-dimension buffer | L1 | S | Q1 |

## L10 Reject `@a.` outside starts and constraints

- What: `parse_text` (`instruction.py`) checks a reference's kind against its place: `@a.` only in a start or a constraint.
- Changes: `Instruction.parse_text` in `taidl/instruction.py` gains a place check; the declared-name check is kept.
- Breaking: yes, ISA descriptions accepted today are rejected.
- Benefits: B can rely on sizes being address-free.
- Caveats: may reject an ISA description that works today in O.
- Fail points: an example breaks; notice by `test_examples.py`.
- Order and dependencies: after L1.
- Verification: new tests per place; the S9 row leaves the "not yet checked" table.

## L11 Reject late buffers and partial instructions

- What: `add_buffer` raises `ValueError` once an instruction exists; an instruction with regions but no semantics, or the reverse, is rejected.
- Changes: `Accelerator.add_data_model` in `taidl/accelerator.py`; a completeness check where generators take the accelerator, since `check_values` runs only when both are given (`instruction.py:99-100`, `254-257`).
- Breaking: yes.
- Benefits: matches S4.5 and S5.2; the data model list is no longer shared with later declarations (`accelerator.py:47-58`, `instruction.py:81`).
- Caveats: a state-only instruction has neither, and stays legal.
- Fail points: a user ISA description interleaves declarations; notice by its error message.
- Order and dependencies: after L1.
- Verification: tests like the existing state-order test.

## L12 Shape and element type checker

- What: an inference pass over each semantics using the L3 rules, rejecting lines whose type, shape, or axis disagree.
- Changes: new `taidl/shapes.py`, called from `Instruction.add_semantics`; `check_operations` is kept.
- Breaking: yes, ISA descriptions accepted today are rejected.
- Benefits: removes the largest gap in S9; every generator gets the same accepted set.
- Caveats: shapes with backtick expressions are checked at issue, as S1.3.8 faults.
- Fail points: a false rejection of a valid ISA description; notice by running every A, B, O example.
- Order and dependencies: after L3; moving O's `shapes.py` here is pending panel (R62).
- Verification: every L3 rule has a passing and a failing test.

## L13 Align integer literal lexing

- What: semantics `INT` allows `007`, expression `INT` rejects it; one rule for both.
- Changes: `INT` in `IDLV2.g4`; every other rule is kept.
- Breaking: yes, `007` handling changes, and `parameter(01)` (F9) is rejected.
- Benefits: one rule for integers in both grammars.
- Caveats: a grammar change needs regeneration with ANTLR 4.9.2.
- Fail points: generated files out of date; notice by L8.
- Order and dependencies: after L1; batch with other grammar edits.
- Verification: grammar tests for `007` in both places.

## L14 Test the exact-token zero divisor rule

- What: S6.2 rejects only the token `0` as a right operand; `(0)` and `-0` fault at issue, so `instruction.py:414` stays as it is.
- Changes: tests only.
- Breaking: no.
- Benefits: the S9 row holds as written with no code change.
- Caveats: none.
- Fail points: none beyond the test.
- Order and dependencies: after L1.
- Verification: tests that `x / 0` is rejected and `x / (0)` is accepted.

## L15 Rename confusing internal names

- What: rename `Instruction.parameters`, which holds the addressing attributes, and delete `num_dim_str`, which no generator reads.
- Changes: `taidl/instruction.py` and `taidl/data_model.py`; S4.6 and L29 follow.
- Breaking: yes, B and O read `parameters`.
- Benefits: `parameters` no longer collides with the `parameter` operation word.
- Caveats: the wider renames of the reconciled glossary are pending panel (R2).
- Fail points: a generator breaks on the rename; notice by B and O test suites.
- Order and dependencies: after L25; B and O update together.
- Verification: grep for the old names in all four repos is empty.

## L16 Range and type checks at the declaring call

- What: reject a `constant` literal its element type cannot hold, an initial value outside signed 64 bits or of type `bool`, a literal shape dimension 0, a non-`str` list entry, and a non-`bool` flag.
- Changes: `add_state`, `semantics_literal_fault`, `strings` (`instruction.py:452-456`), and the flag setters (`instruction.py:168`, `203`).
- Breaking: yes.
- Benefits: rejection at the call, the same for every generator; B and O disagree on literals today (`spec_checks.py:94-107`, `utils.py:398-426`).
- Caveats: float range per type needs care for `f8` types.
- Fail points: drift from O's literal check; notice by sharing one table.
- Order and dependencies: after L3.
- Verification: tests per element type and per S9 row.

## L30 Owner vocabulary in the public API

- What: `add_data_model` becomes `add_buffer` and `add_initial_state` becomes `add_state`, per owner decision 19; `d0` gets one access dimension; `add_constraints` is removed (D34).
- Changes: `taidl/accelerator.py:31-58`; `instruction.py:123` drops the `d0` special case; `instruction.py:321-329` goes.
- Breaking: yes, every ISA description changes its declaring calls.
- Benefits: the API speaks the owner's vocabulary; regions of `d0` follow the general rule.
- Caveats: the old names are removed, not aliased, in the one v3 release.
- Fail points: a tutorial still uses the old names; notice by A's migration.
- Order and dependencies: after L1; B and O update together; L29 follows.
- Verification: grep for `add_data_model` and `add_initial_state` in all four repos is empty.
