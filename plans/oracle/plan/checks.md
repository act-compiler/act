# act-oracle checks

`taidl_to/generic/shapes.py` (543 lines) and the checks in `oracle_visitor.py` and `semantic_generator.py` restate language rules that every generator needs.
Under owner decision 16 every run-time violation must be detected and reported, and today several are not.
Under owner decision 40 the oracle stops only on invalid assembly code or binary input, with an error naming the instruction and the clause, and otherwise runs correctly.
This plan deletes dead code, fixes error classes, moves static rules to the front end through shared item U4, closes the call-time gaps, and audits every raise site against decision 40.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O5 | Delete the dead branch and the dead `sh_copy` check | none | S | Q1 |
| O6 | Spec errors raise `ValueError`; S8.3 table from the front end | L3 | S | Q1 |
| O7 | Delete the oracle's static checks; run the front end's; copy its shape module | U4, O6 | M | Q2 |
| O24 | Call-time errors: checked 64-bit values, constraint overflow, initial state | L5, L6, O20 | S | Q1 |
| O26 | Unwritten on-chip reads raise at the call; zero fill goes | L5, O20 | M | Q2 |
| O31 | Failure policy: errors only on invalid input, naming instruction and clause; audit every raise site | O6, O20 | M | Q1 |

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
- Changes: `taidl_to/oracle_visitor.py`, `README.md`, `tests/test_refusals.py`; preserves which ISA specifications are rejected.
- What: a spec error raises `ValueError`; `NotImplementedError` means only a coverage gap (Q5.8); this part lands now.
- What: a reserved name raises `ValueError` naming the name and the reserved list (D30); the architect can rename it.
- What: a word at a type outside its S8.3 row and a non-integer start index are spec errors (`oracle_visitor.py:254-264`).
- What: once U4 generates the S8.3 table, import it and delete `TYPES` (`oracle_visitor.py:19-97`); this part is a named exit test of U4's taidl step (Q8.4).
- Caveats: the exception class is public contract (`README.md:54-71`); tests assert it.
- Fail points: a test expecting `NotImplementedError` fails loudly; update it with the README.
- Order and dependencies: L3 states the table; the class change lands first.
- Verification: `grep -n NotImplementedError taidl_to` lists only coverage gaps.

## O7 Static checks from the front end

- Breaking: yes.
- Changes: `taidl_to/generic/shapes.py`, `taidl_to/semantic_generator.py`, `taidl_to/oracle_visitor.py`, `taidl_to/generator.py`; preserves each verdict and the call-time entry `disagreement`.
- What: a consumer deletion under U4: U4 moves the rules into taidl; this item deletes the oracle's copies (Q3.1, Q8.4).
- What: the region, type, compare-fit, literal-shape, and constant-range checks run in the front end at generation (Q5.8).
- What: the copies deleted are `RULES` (`shapes.py:450`), `semantic_generator.py:41-56`, `oracle_visitor.py:276-289` and `oracle_visitor.py:373-389`.
- What: the generator copies the front end's shape module into each generated oracle, stamped with its taidl version (owner decision 25, Q3.7).
- What: the oracle keeps call-time evaluation of shapes given by expressions, through the copied module, and faults at the call (Q3.6).
- Benefits: every generator rejects the same ISA specifications; `semantic_generator.py` shrinks to the visitor call.
- Caveats: the copied module imports nothing from `taidl` or ANTLR; a test imports it alone and asserts `antlr4` is not loaded.
- Fail points: verdicts differ between the two copies during migration; run both on every test ISA specification and diff verdicts.
- Order and dependencies: after U4's taidl step and O6; U4 owns the size of the move.
- Verification: same verdict per test ISA specification, old and new; a drift test compares the copy with the source at the stamped version.

## O24 Call-time errors

- Breaking: yes.
- Changes: `runtime/` checks, `runtime/decorator.py`, `tests/`.
- What: every intermediate of an addressing value, state update or constraint is a checked signed 64-bit integer on concrete values (Q5.3, Q1.2).
- What: the check walks the expression tree as parsed, never the canonical sum of products of `utils.py:54-65`.
- What: an overflow raises at the call; `warn_width`'s `UserWarning` (`generic/utils.py:268-271`) goes; state-update results (`templates.txt:363`) and constraints (`templates.txt:353-360`) are not checked today.
- What: an overflowing constraint raises the same error class as a broken constraint, with the overflow named (Q5.3, Q1.12).
- What: an initial state value outside signed 64 bits is rejected (`api_generator.py:43-44` stores it exactly).
- What: two output regions of one call that overlap violate an implicit constraint of the list that O10 checks.
- What: one test per S1.3 case, each naming its row.
- What: a test calls an instruction with a negative addressing attribute and a start `1024 - @a.addr_out`, and expects only the evaluated address to be checked (owner decision 34).
- Benefits: every addressing violation is detected and reported, as decision 16 requires.
- Fail points: a false rejection of a value that never overflows; property-test against exact arithmetic with `hypothesis`.
- Order and dependencies: L5 and L6 state the rules; after O20.
- Verification: the S1.3 test set green; `grep -n UserWarning runtime` empty.

## O26 Unwritten on-chip reads

- Breaking: yes.
- Changes: `runtime/` checks, `taidl_to/templates/templates.txt`, `tests/`.
- What: a read of an on-chip element that no earlier instruction of the kernel wrote raises at the call; `d0` holds the caller's `hbm` in full (Q1.4).
- What: the oracle tracks the written set on the host with a boolean mask per buffer, from the concrete regions after O20.
- What: the zero-fill prologue (`templates.txt:469-470`, `sh_zeros`) goes.
- Benefits: an oracle pass no longer certifies a kernel that reads stale storage.
- Caveats: a stale read of a buffer reused across kernels is not caught, since "written" means written during the kernel.
- Caveats: decision 40 stops the oracle only on invalid assembly code; the item stands only if L5 states an unwritten read as a violation, and real hardware may allow it (decision 31); the owner confirms.
- Fail points: example kernels that read before writing now fail; each needs a zeroing first instruction.
- Order and dependencies: L5 states the rule; after O20, so every region is concrete.
- Verification: a test reads one unwritten element and expects the error; the examples pass.

## O31 Failure policy

- Breaking: yes.
- Changes: `taidl_to/oracle_visitor.py`, `taidl_to/generic/`, `runtime/`, `README.md`, `tests/`; preserves every error on invalid input.
- What: the oracle's errors are of two kinds, and no third path exists.
- What: a generation error rejects a problem the architect can correct, such as a typo, a reserved name or an invalid ISA specification, with a clear message (O6, O7).
- What: a call-time error rejects invalid assembly code, and later invalid binary input; its first line names the instruction and the clause id (S1.3.x), with the iteration when a loop is unrolled.
- What: audit every `raise`, `assert`, warning and jax or MLIR exception that can reach a caller (grep `raise`, `assert`, `NotImplementedError`, `TypeError`); give each a row: kind, input class, clause.
- What: a path that raises on valid input is a bug, fixed or listed in `bugs.md`: D28 and D29 today; a path that computes a wrong result on valid input is the same class (D11, D19, D21).
- What: an exception from jax or MLIR is never the user-facing error: wrap it, name the instruction, and treat it as a bug unless the input was invalid (D20).
- What: the README lists the error classes and states the policy in one place (`README.md:54-71`).
- Benefits: an oracle pass or stop always means something about the kernel.
- Caveats: the exception class is public contract (O6); changes land with the README and the migration note.
- Fail points: an input class missing from the audit; generate valid kernels with `hypothesis` and assert no exception outside the table.
- Order and dependencies: after O6 and O20, since unrolling deletes D11, D19, D20, D28 and D29 at the source.
- Verification: the audit table has no row of kind "valid input"; the property test is green; each invalid-input test asserts instruction and clause in the message.
