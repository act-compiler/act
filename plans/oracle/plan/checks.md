# act-oracle checks

`taidl_to/generic/shapes.py` (543 lines) and the checks in `oracle_visitor.py` and `semantic_generator.py` restate language rules that every generator needs.
Under owner decision 16 every run-time violation must be detected and reported, and today several are not.
This plan deletes dead code, fixes error classes, moves static rules to the front end, and closes the call-time gaps.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O5 | Delete the dead branch and the dead `sh_copy` check | none | S | Q1 |
| O6 | Spec errors raise `ValueError`; S8.3 table from the front end | L3 | S | Q1 |
| O7 | Move static checks to the front end: shapes, region values, constant ranges, `compare` type | L12, B23, O6 | M | Q2 |
| O24 | Call-time errors: checked 64-bit values, overlapping outputs, constraint overflow | L5, L6, O20 | S | Q1 |

## O5 Dead code

- Breaking: no.
- Changes: `taidl_to/oracle_visitor.py`, `taidl_to/generic/utils.py`; preserves all reachable behavior.
- What: delete the unreachable "word not in `TYPES`" branch (`oracle_visitor.py:237-241`).
- What: delete the `copy` type check in `sh_copy` (`generic/utils.py:554-555`): a copy type mismatch is rejected earlier with the instruction named (`shapes.py:98-103`, `shapes.py:136-140`).
- What: the double attribute visit (`oracle_visitor.py:266-274`) goes with O2.
- Benefits: no dead path to keep in step.
- Fail points: none; the branches cannot run.
- Order and dependencies: first.
- Verification: generated `api.py` byte-identical; suite green.

## O6 Error classes and the S8.3 table

- Breaking: yes.
- Changes: `taidl_to/oracle_visitor.py`, `README.md`, `tests/test_refusals.py`; preserves which ISA descriptions are rejected.
- What: a word at a type outside its S8.3 row and a non-integer start index are spec errors; raise `ValueError`, not `NotImplementedError` (`oracle_visitor.py:254-264`).
- What: when the front end exports the S8.3 table, import it and delete `TYPES` (`oracle_visitor.py:19-97`), equal to it row by row today.
- Benefits: `NotImplementedError` then means only an oracle gap, which `docs/coverage.md` lists.
- Caveats: the exception class is public contract (`README.md:54-71`); tests assert it.
- Fail points: a test expecting `NotImplementedError` fails loudly; update it with the README.
- Order and dependencies: L3 states the table; the class change lands first.
- Verification: `grep -n NotImplementedError taidl_to` lists only coverage gaps.

## O7 Static checks to the front end

- Breaking: yes.
- Changes: `taidl_to/generic/shapes.py`, `taidl_to/semantic_generator.py`, `taidl_to/oracle_visitor.py`; preserves each rule and the call-time entry point `disagreement`.
- What: move to the front end the literal-shape rules of `shapes.py` (`RULES` at `shapes.py:450`), the region value type and count check (`semantic_generator.py:41-56`), the `compare` type fit (`oracle_visitor.py:276-283`), literal shape rules (`oracle_visitor.py:285-289`) and the constant range check (`oracle_visitor.py:373-389`).
- What: the oracle keeps only call-time evaluation of shapes given by expressions, calling the front end's rule functions.
- Benefits: every generator rejects the same ISA descriptions; `semantic_generator.py` shrinks to the visitor call.
- Caveats: the generated oracle needs the rules at run time, so it imports or vendors the front end's module at a pinned taidl version.
- Fail points: verdicts differ between the two copies during migration; run both on every test ISA description and diff verdicts.
- Order and dependencies: L12 builds the checker; B23 consumes it; after O6.
- Verification: same verdict per test ISA description, old and new.

## O24 Call-time errors

- Breaking: yes.
- Changes: `runtime/` checks, `runtime/decorator.py`, `tests/`.
- What: every addressing value past signed 64 bits raises at the call, replacing `warn_width`'s `UserWarning` (`generic/utils.py:268-271`); today state-update results (`templates.txt:363`) and constraint arithmetic (`templates.txt:353-360`) are not checked at all.
- What: two output regions of one call that overlap raise at the call, from their concrete regions; today nothing checks it.
- What: a constraint whose arithmetic leaves 64 bits does not hold, as S6.2 says; today it is evaluated exactly and can hold.
- What: an initial state value outside signed 64 bits is rejected (`api_generator.py:43-44` stores it exactly).
- What: one test per S1.3 case, each naming its row.
- Benefits: every addressing violation is detected and reported, as decision 16 requires.
- Caveats: without O20 the checks run on the symbolic loop values per iteration.
- Fail points: a false rejection of a value that never overflows; property-test against exact arithmetic.
- Order and dependencies: L5 and L6 state the rules; after O20 when it lands.
- Verification: the S1.3 test set green; `grep -n UserWarning runtime` empty.
