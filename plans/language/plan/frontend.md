# Plan: front-end fixes

These items bring the `taidl` front end (`taidl/taidl/*.py`) in line with the spec, and rename the confusing internals.
Each comes from reading the code against the spec and fixes rows F1 to F29.
Most turn an undetected error into a rejection listed in S9.
Every rejection carries a code and its section id; the front-end alias `Fault` is renamed `Rejection` (Q3.10).
Sizes assume the shared 3-4 engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L10 | Reject `@a.` outside starts and constraints | L1 | S | Q1 |
| L11 | Reject late buffers and partial instructions | L1 | S | Q1 |
| L12 | Shape and element type checker: merged into U4 | U4 | - | - |
| L13 | Align integer literal lexing between the two grammars | L1 | S | Q1 |
| L14 | Test the exact-token zero divisor rule | L1 | S | Q1 |
| L15 | Rename confusing internal names | L25 | S | Q1 |
| L16 | Range and type checks at the declaring call | L3 | S | Q1 |
| L30 | v3 rename table: public API, token, grammar rule, and `d0` as a one-dimension buffer | L1 | M | Q1 |
| L34 | Rename "refus" to "reject" in messages, docstrings, READMEs, and identifiers | L30 | S | Q1 |
| L35 | Add `set_may_overlap` and the pair-relation record | L36 | S | Q1 |
| L36 | Expand each pair's relation with the size term and the exact-or-apart default | L1 | S | Q1 |

## L10 Reject `@a.` outside starts and constraints

- What: `parse_text` (`instruction.py`) checks a reference's kind against its place: `@a.` only in a start or a constraint.
- Changes: `Instruction.parse_text` in `taidl/instruction.py` gains a place check; the declared-name check is kept.
- Breaking: yes, ISA descriptions accepted today are rejected.
- Benefits: B can rely on sizes being address-free.
- Scope: the check is by reference kind (`@a.` or `@c.`), never by attribute name or value; a negative attribute value is legal (owner decision 34).
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

- Merged into shared item U4 (Q3.1), which moves O's shape module into `taidl` and then adds the checks; L3 stays the spec text it follows.

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

- What: rename `Instruction.parameters`, `comp_attr`, and `var_type` (Q2.12); delete `num_dim_str`, and move `array_dim_str` into the oracle.
- Changes: `taidl/instruction.py` and `taidl/data_model.py`; the new names follow owner decisions 19 and 20 and enter L30's rename table.
- Breaking: yes, B and O read `parameters`.
- Benefits: `parameters` no longer collides with the `parameter` operation word; U2 documents the contract once under the renamed members.
- Caveats: none beyond the table.
- Fail points: a generator breaks on the rename; notice by B and O test suites.
- Order and dependencies: Q1, after L25 and before U2 (Q3.3); B and O update together.
- Verification: grep for the old names in all four repos is empty.

## L16 Range and type checks at the declaring call

- What: reject a `constant` literal its element type cannot hold, an initial value outside signed 64 bits or of type `bool`, a literal shape dimension 0, a non-`str` list entry, and a non-`bool` flag.
- Changes: `add_state`, `semantics_literal_fault`, `strings` (`instruction.py:452-456`), and the flag setters (`instruction.py:168`, `203`).
- Breaking: yes.
- Benefits: rejection at the call, the same for every generator; B and O disagree on literals today (`spec_checks.py:94-107`, `utils.py:398-426`).
- Caveats: a float literal rounds once from its exact decimal (`Fraction` or hex bits, never via f64), and one that rounds to an infinity, or NaN at `f8e4m3fn`, is rejected (Q1.10).
- Fail points: drift from O's literal check; notice by sharing `literal_fits`, a listed contract function (Q3.5).
- Order and dependencies: after L3.
- Verification: tests per element type and per S9 row.

## L30 v3 rename table

- What: one grep-generated per-repo table of old name, new name, and consumer files (Q7.10), citing symbols, not line numbers.
  - `add_data_model` to `add_buffer(name, access_dims, base_element_dims, element_type)`, `add_initial_state` to `add_state(name, value)`, `DataModel` to `Buffer`.
  - `dimensions` to `access_dims`, `unit_dim` to `base_element_dims`, `var_type` to `element_type`.
  - `computation_attr` to `computational_attrs`, `addressing_attr` to `addressing_attrs`.
  - Token `OPERATION` to `OPERATOR`, grammar rule `instruction` to `operation`.
  - `targets/<Name>/` to `accelerators/<name>/` (Q7.11).
- Also: `d0` is declared with one access dimension, so the special case in `instruction.py` goes (Q2.9).
- Kept: `update`, `update_asts`, `update_text`, `Region`, `@s.`, and the marks (owner decision 30).
- Changes: `taidl/accelerator.py`, `instruction.py`, `IDLV2.g4`, `IDLV2Expr.g4`, the regenerated parsers, and the call sites of `.instruction()`.
- Breaking: yes, every ISA specification changes its declaring calls.
- Benefits: the API speaks the owner's vocabulary; regions of `d0` follow the general rule.
- Caveats: no working alias and no converter; each removed name leaves a stub that only raises a coded rejection naming the replacement and the section id (Q7.12, L33).
- Fail points: a tutorial still uses an old name; notice by the CI grep that every old name returns nothing after v3.
- Order and dependencies: after L1; batch the grammar change with L17 and L13; B and O update together; the table feeds L33, L34, and the migration note.
- Verification: the CI grep of old names is empty in all four repos.

## L34 Rename "refus" to "reject"

- What: "fault" at run time, "reject" at generation and kernel compile time (Q7.8); rename "refus" in messages, docstrings, READMEs, and public identifiers.
- Changes: taidl, act-oracle, and act-backend texts and identifiers, from the L30 table; `test_refusals.py` follows panel 6's test-suite item.
- Breaking: yes for tools that match message text or import a renamed identifier.
- Benefits: one static-versus-dynamic word pair across the ecosystem.
- Caveats: coverage status says "unsupported".
- Fail points: a missed spelling; notice by the banned-column CI grep.
- Order and dependencies: after L30; its own v3 item.
- Verification: a grep of "refus" in the four repos is empty outside test fixture names awaiting panel 6.

## L35 Add `set_may_overlap` and the pair-relation record

- What: the new mark `set_may_overlap(may_overlap)` declares the may-overlap relation, whose expansion is empty (S5.5, Q9.1).
- Changes: `taidl/instruction.py` gains the mark beside `set_apart`; the front end records each same-buffer input-output pair's relation and the tie beside `inplace`; S4.6 names both.
- Breaking: no, additive.
- Benefits: a partial overlap a real ISA permits, such as RVV narrowing, has a spelling once the default forbids it (OD31).
- Caveats: no in-scope tensor site yet; panel 7 may rename the mark without changing its meaning.
- Fail points: B ignores the record and allocates by constraint text alone; notice by the backend rows of Q9.4.
- Order and dependencies: after L36; B reads the record in its touch exception.
- Verification: T rows: RVV narrowing (equal start, smaller output) legal under `set_may_overlap` and a fault unmarked; a written positional constraint beside it.

## L36 Expand each pair's relation with the size term and the default

- What: each same-buffer input-output pair expands to exactly one relation's constraint, from its mark or the exact-or-apart default (S5.5, Q9.2).
- Changes: `taidl/instruction.py`: `set_inplace` appends `same(In k, Out)`; `keep_off` compares sizes as well as starts, so `same` holds only on equal starts and equal sizes (F29); unmarked pairs get the default expansion.
- Changes: reject `set_apart` on identical text, unequal size text on a tie, and identical input and output text without `set_inplace`; a lint warns on an unmarked non-elementwise pair.
- Breaking: yes, an unmarked partial overlap, or equal starts with unequal sizes, now faults.
- Benefits: one constraint list for O and B; a forgotten `set_apart` faults loudly instead of giving a corrupt kernel the oracle passes.
- Caveats: terms are evaluated expressions (OD34); a dimension with identical start and size text is skipped.
- Fail points: a mark and its hand-written expansion disagree; notice by the cross-check row over a swept attribute range.
- Order and dependencies: after L1; L5 and L35 build on it.
- Verification: the declaration-by-placement matrix, the unequal-size row, and the OD34 row of Q9.4 pass in O.
