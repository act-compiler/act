# Defect fixes

Each item fixes one bug, named by its D id.
Turning an accepted-but-wrong spec into a rejection breaks a spec that generated before, so those items are marked breaking and ride the coordinated major release.
D6 and D13 are fixed by B6, D10 by B2, D9 by B27, D11 by B9, D12 by B16, D14 by B30.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B21 | Reject a region value whose element count differs from its region (D1) | B3, L5 | S | Q1 |
| B22 | Reject a region value whose element type differs from its buffer (D2) | B3, L5 | S | Q1 |
| B23 | Check shape and element-type agreement of semantics lines (D3) | B7, B13, L3, L12 | M | Q2 |
| B24 | Check, or reject, instructions with regions and no semantics (D4) | L5 | S | Q1 |
| B25 | C++ errors returned, never `assert` (D5) | B4 | S | Q1 |
| B26 | Extraction errors as dropped candidates, unset shape as `Option` (D7, D8) | B7 | S | Q1 |

## B21 Element count of a region value (D1)

- What: in `check_regions` (`spec_checks.py:394-418`) reject `tile != room` for literals, and an expression-sized region whose count cannot be shown equal.
- Preserves: the literal-only fast path of `check_regions`.
- Benefits: an S7.2 spec error is rejected, not compiled.
- Caveats: the expression case may reject specs that are fine for every attribute value; alternative is a selection condition.
- Fail points: a valid example rejected; noticed by example generation.
- Order and dependencies: after B3; with L5, which may move this check into the front end.
- Verification: rejection tests for fewer and more elements.
- Breaking: yes.

## B22 Element type of a region value (D2)

- What: compare each `%In<k>`/`%Out<k>` type with `find_data_model(buffer).var_type`.
- Preserves: `check_regions` structure; one comparison added.
- Benefits: closes an unchecked rule of S7.2.
- Caveats: `d0` holds `u8`; a value read from `d0` in another type is a presumed reinterpretation the spec must define (L).
- Fail points: examples that rely on `d0` reads; noticed by example generation.
- Order and dependencies: after L5 rules on values read from `d0`.
- Verification: rejection test with a mismatched type.
- Breaking: yes.

## B23 Line agreement (D3)

- What: infer each line's shape and type for the supported words and reject a mismatch, from the same rule table as B7.
- What: an ill-typed semantics is a defined error (owner decision 16); which track raises it is open for the panel.
- Preserves: `SemanticsChecker`'s existing rejections.
- Benefits: an ill-typed semantics is rejected with its line instead of yielding an instruction that is never selected.
- Caveats: one rule set in Python and one in Rust unless generated from one table.
- Fail points: a rule stricter than the oracle's; noticed by O's conformance specs.
- Order and dependencies: after B7 and B13; if L12 lands a front-end checker, B23 shrinks to deleting the backend's copy.
- Verification: rejection tests per rule.
- Breaking: yes.

## B24 Instructions with no semantics (D4)

- What: an instruction with regions and no semantics is rejected; one with only state updates stays left out until B16, with a message.
- Preserves: leaving out a state-only instruction.
- Benefits: a forgotten `add_semantics` is not silently dropped (`generator.py:119-122`).
- Caveats: L5 may make this a front-end rejection.
- Fail points: a spec that declares unused instructions; noticed by its owner's generation.
- Order and dependencies: after L5.
- Verification: rejection test.
- Breaking: yes.

## B25 C++ errors returned (D5)

- What: replace `assert(false && "unreachable")` in `parser.cc:14,188` and `constraint.cc:115` with error returns through `why`.
- Preserves: each error's message text.
- Benefits: no abort of the compiler from the allocator; no wrong value under `NDEBUG`.
- Caveats: `parser.cc` mostly goes with B4.
- Fail points: an error path that now continues; noticed by a C++ unit test per path.
- Order and dependencies: with or after B4.
- Verification: `grep 'assert(false' cpp/` empty.

## B26 Extraction errors and unset shapes (D7, D8)

- What: `PiiGraph` construction returns `Result` instead of panicking (`ir/pii.rs:36,42,48`, `ir/buffer.rs:15`); class shape is `Option<Vec<i64>>`, set by B7.
- Preserves: candidate node fields; every non-error path.
- Benefits: a bad candidate is dropped with a reason; rank-0 merges are checked.
- Caveats: touches every shape reader.
- Fail points: a `None` shape read as rank 0; noticed by `merge` on the examples.
- Order and dependencies: after B7.
- Verification: no `panic!` in `ir/pii.rs`; a rank-0 kernel test; B3 gate identical.
