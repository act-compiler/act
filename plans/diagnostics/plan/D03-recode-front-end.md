# D3 Recode and reword the front-end messages

- What: give each of the current rejections that are seeded (`registry-seed.md`, RF001 to RF013 and RF018 to RF034) its code, its spec reference and a template in the wording rules.
- What: drop `buffer` and `machine state variable` from messages and docstrings (`taidl/taidl/instruction.py:117-119, :208`, `accelerator.py:47`), and "refuse" (4 lines).
- What: keep the `semantics line L, column C` position form unchanged.
- Benefits: one wording per condition; S9 rows map one to one to codes; closes the front-end part of E17.
- Caveats: tests that pin message text change with it (E16); do D10's change to code-based assertions in the same commit.
- Fail points: two rules sharing one code (`names.py:14` serves four owners); noticed by the registry test that every code has one rule sentence.
- Order and dependencies: D1, D2, L26.
- Verification: S9 table regenerated from the registry equals the hand-written one row for row; `grep -rniE "refus|honou?r|buffer" taidl/taidl` is empty.
