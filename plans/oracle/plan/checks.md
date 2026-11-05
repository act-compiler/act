# act-oracle shapes and utils simplification

`taidl_to/generic/shapes.py` (543 lines) holds one shape rule per operation word, used both at generation and at each call; `taidl_to/generic/utils.py` (989 lines) holds loop algebra, call checks and emitters.
Shape agreement and element types per word are language rules (S7.4, S8.3), so a second generator needs the same checks.
This plan deletes what is dead, fixes error classes, and moves the language rules to the front end in Q2.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O5 | Delete dead branch and the double attribute visit | T | S | Q1 |
| O6 | Spec errors raise `ValueError`; S8.3 table from the front end | L3 | S | Q1 |
| O7 | Move `shapes.py` rules to the front end, keep call-time evaluation | L12, B23, O6 | L | Q2 |

## O5 Dead code and double visit

- Breaking: no.
- Changes: `taidl_to/oracle_visitor.py`; preserves all reachable behavior.
- What: delete the unreachable "word not in `TYPES`" branch, `taidl_to/oracle_visitor.py:237-241`.
- What: visit a line's attributes once; today lines 266-274 visit them twice, toggling `self.holding`.
- Benefits: removes hidden state from the visitor; about 10 lines fewer.
- Caveats: the second visit yields per-call text that the shape check needs; the single visit must return both forms.
- Fail points: a held value where a per-call one is needed passes generation but checks the wrong shape at the call; loop tests in `tests/test_kernel.py` cover it.
- Order and dependencies: first behavioral-neutral step after O1.
- Verification: generated `api.py` byte-identical for both examples; suite green.

## O6 Error classes and the S8.3 table

- Breaking: yes.
- Changes: `taidl_to/oracle_visitor.py`, `README.md`, `tests/test_refusals.py`; preserves which specs are rejected.
- What: a word at a type outside its S8.3 row and a non-integer start index are spec errors; raise `ValueError` instead of `NotImplementedError` (`oracle_visitor.py:254-264`).
- What: when the front end exports the S8.3 table (L track), import it and delete `TYPES` (`oracle_visitor.py:19-98`), which equals it row by row today.
- What: give `sh_copy`'s rejection (`generic/utils.py:551-556`) the instruction name, or move the check into `shapes.py`.
- Benefits: `NotImplementedError` then means only "the oracle does not support this", which `docs/coverage.md` lists; one copy of S8.3.
- Caveats: changing an exception class changes the public contract in `README.md:54-71`; tests in `tests/test_refusals.py` assert the class.
- Fail points: a test expecting `NotImplementedError` fails loudly; update it with the README in the same commit.
- Order and dependencies: L3 states the table; until the front end exports it, do only the error-class change.
- Verification: `tests/test_refusals.py` green with updated classes; `grep -n NotImplementedError taidl_to` lists only coverage gaps.

## O7 Shape rules into the front end

- Breaking: yes.
- Changes: `taidl_to/generic/shapes.py`, `taidl_to/generic/utils.py`; preserves each rule and the call-time entry point `disagreement`.
- What: `shapes.disagreement` (`generic/shapes.py:541`) and its `RULES` table (`shapes.py:450`) check S7.4 shape agreement; the front end checks form only, per S7 and S9.
- What: move the rules for literal shapes to the front end, so a spec whose shapes disagree is rejected at load for every generator.
- What: the oracle keeps only the call-time evaluation, for shapes given by expressions (`generic/utils.py:771-786`), calling the same rule functions.
- Benefits: the oracle loses most of 543 lines; ACT-Backend gets the same rejections; one definition of shape agreement per word.
- Caveats: the runtime copy in each oracle still needs the rules for expression shapes, so the front end must expose them as a plain module the generated oracle can import or vendor.
- Caveats: cross-track; the front end is frozen in the shared venv, so this lands with a TAIDL release and a version pin in `pyproject.toml`.
- Fail points: a rule that differs between the two copies during migration; run both on every test spec and diff verdicts until the old one is deleted.
- Order and dependencies: L12 builds the front-end checker; B23 consumes the same rules; after O6.
- Verification: every spec in `tests/` gets the same verdict from old and new rules; `shapes.py` reduced to the call-time entry point.
