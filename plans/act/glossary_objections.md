# Objections to the glossary (act track)

Each objection from the act repo's wording and the row of [../glossary/terms.md](../glossary/terms.md) that answers it.

| Objection | Where the act repo shows it | Answer |
| --- | --- | --- |
| No entries for oracle, backend, kernel, ecosystem | README.md:16-21; tests/test_end_to_end.py:14,69 | Rows "test oracle", "test oracle generator", "compiler backend", "compiler backend generator", "kernel function", "ACT ecosystem". |
| "kernel" names an instruction program | tests/test_end_to_end.py:14; the tutorials | "kernel function"; bare "kernel" names only the `@kernel` decorator and its call. |
| "target" is banned, yet the oracle writes `targets/<Name>/` | tests/test_end_to_end.py:78-79 | The generated directory becomes `accelerators/<name>/`. |
| "tile" for a 4x4 block | tests/test_end_to_end.py:43 | "region" (Decision 20). |
| "HBM", "scratchpad", and "buffer" in one tutorial | tutorials/pldi26/exercise1/README.md:23,180,313 | Rows "hbm" and "buffer"; the archived trees keep their words. |
| "ISA specification" and "TAIDL specification" in the README | README.md:12,18,20 | "ISA specification" is the user's file; "TAIDL language specification" is the language document (Decision 32). |
