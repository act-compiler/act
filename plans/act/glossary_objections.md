# Objections to the TAIDL glossary (act track)

- The glossary has no entries for words the act repo uses everywhere: oracle, test oracle, TAIDL-TO, ACT-Backend, backend, ISA, kernel, ecosystem.
  README.md:16-21 and tests/test_end_to_end.py:14,69 use them with no canonical form.
  Default: add `oracle`, `backend`, `kernel` (a program written with an accelerator's instructions) and `ecosystem`.
- `kernel` is a banned synonym for `semantics`, yet it names an instruction program in tests/test_end_to_end.py:14 and tutorials/pldi26/exercise2/README.md.
  Default: define `kernel` for programs, keep the ban for semantics.
- `target` is banned for `accelerator`, but the oracle writes `targets/<Name>/` (tests/test_end_to_end.py:78-79 uses `tmp_path / 'targets' / 'MM'`).
  Default: rename the generated directory in act-oracle and act-backend, or exempt the path in the glossary.
- `region` bans `tile`, but tests/test_end_to_end.py:43 and the tutorials say "tile" for a 4x4 or 64x64 block.
  Default: the glossary wins; the tutorials follow when the current one is revised.
- `off-chip memory` bans HBM and `data model` bans scratchpad and buffer; the current tutorial uses all three (tutorials/pldi26/exercise1/README.md:23,180,313).
  Default: the glossary wins.
- `spec` is defined as a Python program, while README.md:41 and the tutorials say "ISA specification" and "TAIDL specification".
  Default: say "spec" and drop "ISA specification" except when naming the paper.
