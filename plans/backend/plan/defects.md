# Defect fixes

Each item fixes one bug, named by its D id.
Turning an accepted-but-wrong ISA specification into a rejection breaks one that generated before, so those items are marked breaking and ride act-backend 2.0 in the coordinated major release (owner decisions 10 and 26).
D15 is fixed by B38 (encoding.md), D6 and D13 by B6, D9 by B27, D10 by B2, D11 by B9, D12 by B16, D14 and D16 by B30.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B21 | Entry takes a `Model` or calls `check` once; delete the backend's region count check (D1, D2) | U3 | S | Q1 |
| B22 | Merged into B21 | -- | -- | -- |
| B23 | Cut: line agreement (D3) is U4's | -- | -- | -- |
| B24 | Reject instructions with no semantics, exclude state-only ones (D4) | L5 | S | Q1 |
| B25 | C++ errors returned, never `assert` (D5) | -- | S | Q1 |
| B26 | Extraction errors as dropped candidates, unset shape as `Option`, shape disagreement as a coded fault (D7, D8) | B7 | S | Q1 |

## B21 The checked model (D1, D2)

- What: `generate_backend` takes a `Model` or an `Accelerator`; given an `Accelerator` it calls `check` once (U10, Q11.3); its body reads only the `Model`.
- What: `check_regions` (`spec_checks.py:394-418`) is deleted.
- What: the element count and element type of each region value (D1, D2) are U3's checks, raised with their shared codes.
- Preserves: every backend limit and its message text, now the reason of an exclusion (B14).
- Benefits: one rejection per rule across generators; the backend keeps no parallel checker (Q3.2).
- Caveats: the literal-only count check stays until U3 ships, and no interim check is written (Q3.2, Q4.8).
- Fail points: a valid example rejected by `check`; noticed by example generation.
- Order and dependencies: after U3; no v3 backend ships without U3 (owner decision 10).
- Verification: a test that a count or type mismatch raises from `check`; `grep check_regions act_backend` empty; the entry gives equal output for an `Accelerator` and its `Model`.
- Breaking: yes.

## B22 Merged into B21

- The element-type check of a region value (D2) is U3's; B21 makes the call.

## B23 Cut

- Shape and element-type agreement between semantics lines (D3) folds into U4 (Q3.1).
- The backend asserts typed semantics from the `Model` and keeps no rule table of its own (Q4.8).

## B24 Instructions with no semantics (D4)

- What: an instruction with regions and no semantics is rejected, since the architect can add the body.
- What: one that only updates or reads state is excluded with a reason until B16, and an update stays excluded after it (B14, B46).
- Preserves: leaving out a state-only instruction, now with a printed and stored reason.
- Benefits: a forgotten body is not silently dropped (`generator.py:119-122`); decision 16 makes the silent drop a defect (Q4.7).
- Caveats: L5 may make the first part a front-end rejection; then B24 shrinks to the state-only exclusion.
- Fail points: an ISA specification that declares unused instructions; noticed by its owner's generation.
- Order and dependencies: after L5; first of the state work (Q4.7).
- Verification: rejection test; exclusion-report test for a state-only instruction.
- Breaking: yes.

## B25 C++ errors returned (D5)

- What: replace `assert(false && "unreachable")` in `parser.cc:14,188` and `constraint.cc:115` with error returns through `why`.
- Preserves: each error's message text.
- Benefits: no abort of the compiler from the allocator; no wrong value under `NDEBUG`.
- Caveats: `parser.cc` mostly goes with B4, so the edit there is small.
- Fail points: an error path that now continues; noticed by a C++ unit test per path.
- Order and dependencies: first item of lane B, before B2 and B4 (Q4.11).
- Verification: `grep 'assert(false' cpp/` empty.

## B26 Extraction errors, unset shapes, shape disagreement (D7, D8)

- What: `PiiGraph` construction returns `Result` instead of panicking (`ir/pii.rs:36,42,48`, `ir/buffer.rs:15`); class shape is `Option<Vec<i64>>`, set by B7.
- What: when `merge` joins a declared and a computed dim that are both bound and differ, it raises the shared coded shape fault at selection, not a panic (Q3.6, Q3.11).
- Preserves: candidate node fields; every non-error path; a kernel-versus-pattern mismatch stays a non-match.
- Benefits: a bad candidate is dropped with a reason; rank-0 merges are checked; no panic is left in selection (owner decision 16).
- Caveats: touches every shape reader; the computed dim must be expressible for each supported tensor operator.
- Fail points: a `None` shape read as rank 0; noticed by a rank-0 kernel test.
- Order and dependencies: after B7.
- Verification: no `panic!` in `ir/pii.rs` or `merge`; a rank-0 kernel test; a disagreeing-description test raises the coded fault; B3 gate identical.
