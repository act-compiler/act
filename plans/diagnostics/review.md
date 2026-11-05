# What works and should stay

## Front end

- Every rejection happens at the call that takes the faulty input (`taidl/taidl/instruction.py:81-110`, `accelerator.py:47-93`), as S1.1 requires.
- Messages name the instruction (`{instruction}: ...`) and, for a text, the place, line and column counted from 1 (`instruction.py:345-368`).
- ANTLR errors are collected, not printed (`instruction.py:23-30`, `:440-449`), so a bad text gives one exception with a position.
- One table drives operand and attribute checks: `OPERATIONS` (`taidl/taidl/operations.py:42-164`), with one row per word and one function per check (`:171-239`).
- One function checks every name (`taidl/taidl/names.py:7-14`).
- Placement marks give one message per condition and replace what they appended before (`instruction.py:178-253`).
- The first fault is reported with its token (`Fault = tuple[CommonToken, str]` today, renamed `Rejection`; `operations.py:167-168`); a checker can return a list later without a redesign.

## act-backend

- All Python checks run before anything is written (`generator.py:125-136`), pinned by `tests/test_refusals.py:439`.
- Each Python message names the instruction and quotes the offending text (`spec_checks.py:26-415`).
- Every tensor operator and element type is tested as handled or rejected (`tests/test_refusals.py:149, :185`) and the tables are tested against the e-graph's (`:164, :200`); keep these table-sync tests.
- Rust reader faults are `Result<_, String>` from the line parser to one exit (`pipeline.rs:449`), not panics.
- The output is probed for writing before the search starts (`pipeline.rs:405-414`), and a failed compile leaves no output (`pipeline.rs:55-58`).
- A non-finite cost is rejected (`cost/c.rs:20-22`).
- The first dropped candidate's reason is kept and reported when none survives (`pipeline.rs:208-218`).
- The C++ allocator returns a reason string through a buffer (`include/act_malloc.h:13-15`, `src/act_malloc.cc:16-18`) rather than throwing across the FFI.
- A failed compile ends in one stderr line and status 1, tested through the real binary (`tests/test_endings.py:117`).

## act-oracle

- Generation builds the API text first and removes the previous oracle before any rejection, so a rejected generation leaves none (`generator.py:30-48`, tested at `tests/test_refusals.py:467-488`).
- The oracle is built in a staging directory and moved into place (`generator.py:57-85`).
- Call-time errors name the instruction and the loop iteration (`generic/utils.py:662-783`).
- A broken constraint names the instruction, the constraint text, the reference values and the iteration (`generic/utils.py:796-804`).
- Reserved names are rejected before any file is written (`generator.py:36-42`, `api_generator.py:52-59`).
- A refusal test exists per rejection (`tests/test_refusals.py`, `tests/test_constraints.py`), pinning message text.

## act

- `act/tests/check_versions.py:105-119` collects every drift and exits once, listing them all.
- `act/act/generators/__init__.py:4-16` imports each generator on call, so one missing package does not break the other.

## Keep as the base of the scheme

- The `{instruction}: ... position: ...` shape of the front end becomes line 1 of the common format.
- The `at i = v` suffix of the oracle becomes the `V` iteration field.
- The Rust single-exit `Result` flow and the C++ reason buffer are the `Diagnostic` carriers for generators.
- The tests that sync tables with code become the registry coverage test.
