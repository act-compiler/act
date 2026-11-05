# Coverage gaps

`act-backend/docs/coverage.md` lists each TAIDL feature as supported, rejected, excluded or partial.
An instruction that uses a feature the backend cannot model is excluded from the generated backend with a printed, stored reason (B14, B46); a problem the architect can correct is rejected.
State is excluded wherever an instruction with semantics reads or updates it (`spec_checks.py:249-251,274-292,328-367,374-385`); an instruction with no semantics is left out unchecked today (`generator.py:119-122`).
The other gaps: 18 tensor operators plus `parameter` (`isel_ir2isa_rewrites_txt_generator.py:11-30`, `spec_checks.py:53-55`), two-operand `concatenate`, six element types (`isel_ir2isa_rewrites_rs_generator.py:24-38`), one output only (`spec_checks.py:22-29`).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B15 | Keep `docs/coverage.md` in sync by a test; document S1.3 cases | B14, L7, T | S | Q1 |
| B16 | Read-only state folded at compile time | L1, B13, B24 | M | Q2 |
| B17 | Operations beyond the 18, `concatenate` of any arity | B6, B7, B31 | M | Q2 |
| B18 | Element types `s2`, `s4`, `u2`, `u4`, `f8e4m3fn`, `f8e5m2` | B7, L17 | S | Q2 |
| B19 | Several output regions | B2, B11 | L | Q2 |
| B20 | `slice` stride, `constant` expressions, float literal range | B6 | S | Q2 |

## B15 Keep `docs/coverage.md` in sync by a test

- What: a test that maps each rejected or excluded row to an ISA specification that triggers it and each supported row to an example or a generated one.
- What: the doc states each S1.3 case the backend fixes: `bitcast_convert` byte order, initial value of `reduce_add`, size of `d0`, and what an in-place mark and each pair relation do to allocation (B30).
- What: the doc lists each precondition the backend cannot see, such as state values and the size of `d0` (Q1.12).
- Preserves: `docs/coverage.md` columns and status words.
- Benefits: the coverage doc cannot go stale; S1.2 conformance is checkable.
- Caveats: row wording must carry a stable key.
- Fail points: a row with no test; the test lists untested rows and fails.
- Order and dependencies: after B14; serves L7's unsupported-feature tables; with T.
- Verification: test passes and fails when a row is removed.

## B16 State in the backend

- What: a kernel compiles under one known state; read-only `@s.` in starts, sizes and constraints folds to an integer at compile time (Q4.7).
- What: an instruction with any other use of state stays excluded, and its reason names the state (B27, B46).
- Not in this plan: state updates and state-only instruction emission follow after v3, since an e-graph orders no effects (Q4.7).
- Preserves: today's exclusion of `@s.` until each use is supported.
- Benefits: ISA specifications with modes (dataflow, layout) become usable for read-only state.
- Caveats: the `@x.` writer path would compile `@s.n` as attribute `n` (D12), so each fold needs its own test.
- Fail points: a fold under a state other than the kernel's; noticed only by oracle simulation, so add state kernels with T and O.
- Order and dependencies: after B24, B13, and L1's ruling on state.
- Verification: new state examples simulate equal to their NumPy references; the read-only coverage rows flip to supported.
- Breaking: no for existing ISA specifications.

## B17 Operations beyond the 18

- What: add ops in priority order of the examples' needs: `subtract`, `multiply`, `maximum`, `minimum`, `compare`, `select`, `reduce_max`, `negate`; each needs a `TensorOp` arm, a reader arm and a shape rule.
- What: `concatenate` takes one or more operands (S8.2), not exactly two (`egraph.rs:13`).
- Preserves: the 18 supported operations' matching and payloads.
- Benefits: more accelerators expressible.
- Caveats: each op touches reader, e-graph and allocator; cheaper after B6 and B7.
- Fail points: an op whose payload omits an attribute; noticed by an attribute-mismatch test per op.
- Order and dependencies: after B31, B7, B6.
- Verification: per-op exclusion test flips to a selection test.

## B18 Narrow element types

- What: add the six types to `Dtype` and to `ELEMENT_TYPES`.
- Preserves: the 13 supported element types' byte sizes.
- Benefits: closes S7.3.
- Caveats: sub-byte types need packing rules in the allocator's byte sizes.
- Fail points: a byte size rounded wrong; noticed by an allocator test with an `s4` region.
- Order and dependencies: after B7.
- Verification: exclusion tests for these types become selection tests.

## B19 Several output regions

- What: lift the one-output exclusion (`check_outputs`); patterns get several roots and the allocator places several outputs.
- Preserves: single-output selection as the one-root case.
- Benefits: closes S5.2.
- Caveats: egg patterns have one root; needs a multi-pattern or a tuple op.
- Fail points: an output left unplaced; noticed by oracle simulation.
- Order and dependencies: after B2, B11; may slip past Q2.
- Verification: a two-output example simulates correctly.

## B20 Slice stride and constant expressions

- What: carry the stride in the slice payload; fold a backtick `constant` at selection; reject a float literal outside its element type (`spec_checks.py:106-107` accepts any).
- Preserves: slice payload layout for stride 1.
- Benefits: two exclusions removed; one missing rejection added.
- Caveats: kernel reader must read strided slices too.
- Fail points: stride ignored in matching; noticed by an attribute-mismatch test.
- Order and dependencies: after B6.
- Verification: rows flip in `docs/coverage.md` with tests.
