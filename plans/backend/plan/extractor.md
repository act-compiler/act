# Candidate extraction and in-place storage

Every iteration, `pass` (`generic/src/pipeline.rs:226-262`) runs a fast and a slow extractor and allocates each candidate.
The fast extractor is a cost-blind first-feasible search (`isel/extractor/fast.rs:88-128`); the slow one enumerates every tree up to a growing node limit by cartesian product (`slow.rs:117`, `pipeline.rs:502`).
The default cost is the count of `api.` lines in the assembly (`cpp/cost/count.cc:17`), so a node-count proxy ranks candidates close to the cost model.
The allocator lets an in-place output overlap its marked input but does not force it (`cpp/malloc/src/constraint.cc:214-230`); owner decision 17 says the output reuses that storage.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B10 | One cost-guided extractor keeping the best few candidates | B9 | M | Q2 |
| B11 | In-place copies as a step of candidate building | B10 | M | Q2 |
| B30 | In-place storage reuse as a solver equality | B2 | M | Q1 |

## B10 One cost-guided extractor

- What: one `extraction` module; a k-best enumeration on (class, buffer) ranked by a node-count proxy feeds the allocator; the real cost model picks among those that allocate.
- Preserves: the (class, buffer) state and child buffer rule (`fast.rs:50-54,99-106,200-208`); drop reasons (`pipeline.rs:207-219`).
- Benefits: replaces two strategies (283 and 208 lines) with one; candidates come in cost order.
- Caveats: k must cover candidates the allocator drops; the order of `--log` files changes once.
- Fail points: a cost rise on an example; noticed by the B3 cost comparison.
- Order and dependencies: after B9; a behavioral step with a new pinned order.
- Verification: equal or lower cost on all examples; oracle run equal; time per kernel reported.

## B11 In-place copies as a step of candidate building

- What: `fix_inplace` (`isel/extractor/inplace_fix.rs:9`) edits each candidate after extraction; build the copy of a still-live marked input during candidate building, with its cost visible.
- Preserves: the copy rule and its two error cases.
- Benefits: the marked input is dead after the instruction, as B30 requires.
- Caveats: needs a copy instruction per buffer that holds a shared overwritten input.
- Fail points: a candidate that overwrites a live value; noticed only by the oracle, so add a kernel test with a shared overwritten input.
- Order and dependencies: after B10; T adds the kernel test.
- Verification: oracle run equal on the new kernel test.

## B30 In-place storage reuse as a solver equality

- What: for an in-place pair, add an equality constraint on the output's and the marked input's offsets in the solver, replacing the skipped interference edge (`constraint.cc:214-230`).
- Preserves: the interference edges of every other pair.
- Benefits: the output reuses the marked input's storage on every allocation (owner decision 17).
- Caveats: until B11, the extractor's fix-up keeps the input dead; a candidate where it is live is dropped with a reason.
- Fail points: an allocation that becomes infeasible; noticed as a dropped candidate with the reason.
- Order and dependencies: after B2.
- Verification: oracle test with a shared overwritten input; output offset equals input offset in the `--log` allocation.
