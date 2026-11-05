# Plan: front-end fixes

These items bring the `taidl` front end (`taidl/taidl/*.py`) in line with the spec, and rename the confusing internals.
Each comes from reading the code against the spec and fixes rows LB1 to LB29.
Most turn an undetected error into a rejection listed in S9.
Every rejection carries a code and its section id; the front-end alias `Fault` is renamed `Rejection` (Q3.10).
Sizes assume the shared 3-4 engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L10 | Reject `@a.` outside starts and constraints | L1 | S | Q1 |
| L11 | `check` reports a late buffer, a late state variable, and a partial instruction at the declaring call's site | L1 | S | Q1 |
| L12 | Shape and element type checker: merged into U4 | U4 | - | - |
| L13 | Integer literal lexing: no target after U14 | U14 | - | - |
| L14 | Test the plain-zero divisor rule | U12 | S | Q1 |
| L15 | Rename confusing internal names | L25 | S | Q1 |
| L16 | `check` reports a range or type error at the declaring call's site | L3 | S | Q1 |
| L30 | v3 rename table: public API and `d0` as a one-dimension buffer | L1 | M | Q1 |
| L34 | Rename "refus" to "reject" in messages, docstrings, READMEs, and identifiers | L30 | S | Q1 |
| L35 | Add `set_may_overlap` and the pair-relation record | L36 | S | Q1 |
| L36 | Expand each pair's relation with the size term and the exact-or-apart default | L1 | S | Q1 |

## L10 Reject `@a.` outside starts and constraints

- What: `parse_text` (`instruction.py`) checks a reference's kind against its place: `@a.` only in a start or a constraint.
- Changes: `check` (L43) checks each reference's kind against its place in the `Model`; the declared-name check is kept.
- Breaking: yes, ISA specifications accepted today are rejected.
- Benefits: B can rely on sizes being address-free.
- Scope: the check is by reference kind (`@a.` or `@c.`), never by attribute name or value; a negative attribute value is legal (owner decision 34).
- Caveats: may reject an ISA specification that works today in O.
- Fail points: an example breaks; notice by `test_examples.py`.
- Order and dependencies: after L1.
- Verification: new tests per place; the S9 row leaves the "not yet checked" table.

## L11 `check` reports late declarations and partial instructions

- What: `check` reports a late buffer, a late state variable, and a partial instruction at the declaring call's site, collected with every other problem (S4.5, S9).
- Changes: `Accelerator.add_data_model` in `taidl/accelerator.py`; a completeness check where generators take the accelerator, since `check_values` runs only when both are given (`instruction.py:99-100`, `254-257`).
- Breaking: yes.
- Benefits: matches S4.5 and S5.2; the data model list is no longer shared with later declarations (`accelerator.py:47-58`, `instruction.py:81`).
- Caveats: a state-only instruction has neither, and stays legal.
- Fail points: a user ISA specification interleaves declarations; notice by its error message.
- Order and dependencies: after L1.
- Verification: tests like the existing state-order test.

## L12 Shape and element type checker

- Merged into shared item U4 (Q3.1), which moves O's shape module into `taidl` and then adds the checks; L3 stays the spec text it follows.

## L13 Align integer literal lexing

- No target: both grammars leave with U14, and plain integers in the Python surface follow S6.2.

## L14 Test the plain-zero divisor rule

- What: S6.2 raises at the author's line on a `//` or `%` whose divisor is the plain integer 0; a symbolic divisor that evaluates to 0 faults at issue.
- Changes: tests only.
- Breaking: no.
- Benefits: the S9 row holds as written with no code change.
- Caveats: none.
- Fail points: none beyond the test.
- Order and dependencies: after L1.
- Verification: tests that `x // 0` raises and `x // y` with `y` evaluating to 0 faults in O.

## L15 Rename confusing internal names

- What: rename `Instruction.parameters`, `comp_attr`, and `var_type` (Q2.12); delete `num_dim_str`, and move `array_dim_str` into the oracle.
- Changes: `taidl/instruction.py` and `taidl/data_model.py`; the new names follow owner decisions 19 and 20 and enter L30's rename table.
- Breaking: yes, B and O read `parameters`.
- Benefits: `parameters` no longer collides with the `parameter` tensor operator; U2 documents the contract once under the renamed members.
- Caveats: none beyond the table.
- Fail points: a generator breaks on the rename; notice by B and O test suites.
- Order and dependencies: Q1, after L25 and before U2 (Q3.3); B and O update together.
- Verification: grep for the old names in all four repos is empty.

## L16 Range and type checks reported at the declaring call's site

- What: `check` reports, at the declaring call's site, a `constant` literal its element type cannot hold, an initial value outside signed 64 bits or of type `bool`, a literal shape dimension 0, a non-`str` list entry, a non-`bool` flag, a plain start below 0 or size below 1, a plain start plus size past a plain access dimension, and an axis attribute that is not a plain integer.
- Changes: `add_state`, `semantics_literal_fault`, `strings` (`instruction.py:452-456`), and the flag setters (`instruction.py:168`, `203`).
- Breaking: yes.
- Benefits: rejection at the call, the same for every generator; B and O disagree on literals today (`spec_checks.py:94-107`, `utils.py:398-426`).
- Caveats: a plain number rounds once from the exact value received, a `float` as its binary64 value and an `int` or `Fraction` exactly, never through `float()`; one that rounds to an infinity, or NaN at `f8e4m3fn`, is rejected (Q1.10, Q13b.9).
- Fail points: drift from O's literal check; notice by sharing `literal_fits`, a listed contract function (Q3.5).
- Order and dependencies: after L3.
- Verification: tests per element type and per S9 row.

## L30 v3 rename table

- What: one grep-generated per-repo table of old name, new name, and consumer files (Q7.10), citing symbols, not line numbers.
  - `add_data_model` to `acc.buffer(name, access_dims, base_element_dims, element_type)`, `add_initial_state` to `acc.state(name, value)`, `DataModel` to `Buffer`.
  - `add_instruction` and its attribute lists to `@acc.instruction` with `Comp` and `Addr` parameters (S5.1).
  - `set_inputs`, `set_outputs`, `add_semantics`, and the marks to `read`, `write`, `ops`, and `write` markers (S5.2, S5.5, U14).
  - `targets/<Name>/` to `accelerators/<name>/` (Q7.11).
- Also: `d0` is declared with one access dimension, so the special case in `instruction.py` goes (Q2.9).
- Also: the findings of the outside architect's ISA specification on the traced surface (A24, by M3) enter the table, beside the removed-names table (D15).
- Kept: `update`, `Region`, and the markers as `write` keywords (owner decision 30).
- Changes: `taidl/accelerator.py`, `instruction.py`, and the call sites of `.instruction()`.
- Breaking: yes, every ISA specification changes its declaring calls.
- Benefits: the API speaks the owner's vocabulary; regions of `d0` follow the general rule.
- Caveats: no working alias; the U13 converter moves the corpus; each removed name leaves a stub that only raises a coded rejection naming the replacement and the section id (Q7.12, L33).
- Fail points: a tutorial still uses an old name; notice by the CI grep that every old name returns nothing after v3.
- Order and dependencies: after L1; lands with U12; B and O update together; the table feeds L33, L34, U13, and the migration note.
- Verification: the CI grep of old names is empty in all four repos.

## L34 Rename "refus" to "reject"

- What: "fault" at run time, "reject" at generation and kernel compile time (Q7.8); rename "refus" in messages, docstrings, READMEs, and public identifiers.
- Changes: taidl, act-oracle, and act-backend texts and identifiers, from the L30 table; `test_refusals.py` follows panel 6's test-suite item.
- Breaking: yes for tools that match message text or import a renamed identifier.
- Benefits: one static-versus-dynamic word pair across the ecosystem.
- Caveats: coverage status says "unsupported".
- Fail points: a missed spelling; notice by the banned-column CI grep.
- Order and dependencies: after L30; its own v3 item.
- Verification: a grep of "refus" in the four repos is empty outside test fixture names.

## L35 Add `set_may_overlap` and the pair-relation record

- What: the new mark `set_may_overlap(may_overlap)` declares the may-overlap relation, whose expansion is empty (S5.5, Q9.1).
- Changes: `taidl/instruction.py` gains the mark beside `set_apart`; the front end records each same-buffer input-output pair's relation and the tie beside `inplace`; S4.6 names both.
- Breaking: no, additive.
- Benefits: a partial overlap a real ISA permits, such as RVV narrowing, has a spelling once the default forbids it (OD31).
- Caveats: no in-scope tensor site yet; panel 7 may rename the mark without changing its meaning.
- Fail points: B ignores the record and allocates by constraint text alone; notice by the backend rows of Q9.4.
- Order and dependencies: after L36; B reads the record in its touch exception.
- Verification: T rows: RVV narrowing (one start text, smaller output) legal under `set_may_overlap`; unmarked, a fault with a symbolic size and rejected with plain sizes; a written positional constraint beside it.

## L36 Expand each pair's relation with the size term and the default

- What: each same-buffer input-output pair expands to exactly one relation's constraint, from its mark or the exact-or-apart default (S5.5, Q9.2).
- Changes: `taidl/instruction.py`: `set_inplace` appends `same(In k, Out)`; `keep_off` compares sizes as well as starts, so `same` holds only on equal starts and equal sizes (LB29); unmarked pairs get the default expansion.
- Changes: reject `set_apart` on identical text, unequal size text on a tie, and identical input and output text without `set_inplace`; a lint warns on an unmarked non-elementwise pair.
- Changes: reject two outputs of one buffer with equal start text, an exact-or-apart pair with equal start text and unequal plain sizes, and an input tied to more than one output (S5.5, Q13b.3).
- Breaking: yes, an unmarked partial overlap, or equal starts with unequal sizes, now faults, or is rejected where the text decides it.
- Benefits: one constraint list for O and B; a forgotten `set_apart` faults loudly instead of giving a corrupt kernel the oracle passes.
- Caveats: terms are evaluated expressions (OD34); a dimension with identical start and size text is skipped.
- Fail points: a mark and its hand-written expansion disagree; notice by the cross-check row over a swept attribute range.
- Order and dependencies: after L1; L5 and L35 build on it.
- Verification: the declaration-by-placement matrix, the unequal-size row, and the OD34 row of Q9.4 pass in O.
