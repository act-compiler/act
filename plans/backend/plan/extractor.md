# Candidate extraction and in-place storage

Every iteration, `pass` (`generic/src/pipeline.rs:226-262`) runs a fast and a slow extractor and allocates each candidate.
The fast extractor is a cost-blind first-feasible search (`isel/extractor/fast.rs:88-128`); the slow one enumerates every tree up to a growing node limit by cartesian product (`slow.rs:117`, `pipeline.rs:502`).
After saturation the loop re-extracts an unchanged e-graph until the time limit (`pipeline.rs:544-547`).
The default cost is the count of `api.` lines in the assembly (`cpp/cost/count.cc:17`), known only after allocation, so a node-count proxy can only approximate it.
The allocator reads each same-buffer input-output pair's relation and applies it where live ranges touch (B39, B30); the copy pass reads the tie only (B11).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B37 | One extraction module and no re-extraction after saturation | B1, B29 | S | Q1 |
| B10 | One k-best extractor on (class, buffer) | B9, B29, B37 | M | Q2 |
| B36 | Drop a candidate that reads unwritten on-chip storage | B10 | S | Q2 |
| B11 | Copy of a shared tied input as a step of candidate building | B10, B30 | M | Q2 |
| B30 | Pair relations in the allocator: tie equality and a narrowed touch exception | B2, B39 | M | Q1 |
| B39 | Pair-relation record in the typed handoff | B2, L35 | S | Q1 |
| B40 | Allocation rows per relation | B30, T | S | Q1 |

## B37 One extraction module, no re-extraction after saturation

- What: move `fast.rs`, `slow.rs`, `utils.rs` and `lib.rs` into one `extraction` module with no logic change; stop the loop once the e-graph is saturated instead of re-extracting it (Q4.5).
- Preserves: both strategies, their candidates and their order.
- Benefits: removes repeated passes over an unchanged e-graph; one place for B10.
- Caveats: the stop must still report its stop reason under `--verbose`.
- Fail points: a candidate lost because a later pass found it; noticed by the B3 gate.
- Order and dependencies: after B1 and B29; structural.
- Verification: B3 gate identical; fewer passes logged after saturation.

## B10 One k-best extractor

- What: one k-best enumeration on (class, buffer) ranked by a node-count proxy; the k candidates are re-ranked through the cost ABI after allocation (Q4.5).
- What: the enumerating extractor stays as a test-only cross-check; k is measured on the examples before it is fixed.
- Preserves: the (class, buffer) state and child buffer rule (`fast.rs:50-54,99-106,200-208`); drop reasons (`pipeline.rs:207-219`).
- Benefits: replaces two strategies (283 and 208 lines) with one; candidates come in cost order; allocation feasibility is a filter on them.
- Caveats: k must cover candidates the allocator drops; the order of `--log` files changes once.
- Fail points: a cost rise on an example; noticed by the B3 cost comparison.
- Order and dependencies: after B9 and B29 (gate in place) and B37; behavioral.
- Verification: equal or lower cost on all examples; oracle run equal; the test-only enumerator finds no cheaper candidate on the examples.

## B36 Unwritten on-chip storage

- What: a candidate that reads an element of an on-chip buffer no earlier issue wrote is dropped with a named reason; no zeroing is inserted (Q4.9, Q1.4).
- What: a buffer with a declared initial value is exempt, and the backend may rely on that value instead of a zeroing issue (joint decision on unwritten reads).
- What: the backend emits the zeroing or overwriting first issue an ISA provides where a kernel needs one; it never emits a faulting issue (Q1.12).
- Preserves: every candidate whose reads all have a producer.
- Benefits: the oracle's unwritten-read fault never fires on a backend program.
- Caveats: arises only from partial writes and storage reuse, both visible on the candidate.
- Fail points: a valid candidate dropped; noticed by the oracle run and a partial-write kernel test.
- Order and dependencies: after B10.
- Verification: a partial-write kernel test; every backend program runs on the oracle with no fault.

## B11 Copy of a shared tied input as a step of candidate building

- What: `fix_inplace` (`isel/extractor/inplace_fix.rs:9`) edits each candidate after extraction; build the copy of a still-live tied input during candidate building, with its cost visible.
- What: the pass reads the tie only; no other relation (apart, exact-or-apart, may overlap) causes a copy (Q9.4).
- Preserves: the copy rule and its two errors, an HBM-pinned operand and a shared-view operand (`inplace_fix.rs:6-17,59-80`).
- Benefits: the copy's cost is visible to the extractor.
- Caveats: needs a copy instruction per buffer that holds a shared overwritten input.
- Fail points: a candidate that overwrites a live value; noticed only by the oracle, so B40 adds a kernel test with a shared tied input.
- Order and dependencies: after B10 and B30; T adds the kernel test.
- Verification: oracle run equal on the new kernel test.

## B30 Pair relations in the allocator

- What: for a tensor pair that only touches (an input dying at instruction u and the output of u, in one storage), `build_interference_graph` adds the pair's relation from the record (B39) where `constraint.cc:206` adds nothing today.
- Relations on the pair's offsets and sizes:
  - tied: equal offsets in every dimension (owner decision 17);
  - exact-or-apart: equal offsets where the sizes are equal, or the apart disjunction of `constraint.cc:88-103`;
  - apart: the apart disjunction;
  - may overlap: no constraint.
- What: delete the exemption of `constraint.cc:214-230`; a pair whose live ranges overlap keeps its edge under every relation, because the input is read after the output is written.
- What: if exact-or-apart overloads the solver on large kernels, allocate default pairs apart, stricter and never faulting; this is a backend limit stated in the coverage table, not a language rule (Q9.4).
- Preserves: the interference edges of every other pair; the apart disjunction.
- Benefits: the output reuses a tied input's storage on every allocation (D14); no allocation overlaps a pair beyond its relation (D16); no issue faults (Q1.12).
- Caveats: until B11, the extractor's fix-up keeps a tied input dead; a candidate where it is live is dropped with a reason.
- Caveats: the solver cost of the default disjunction is unmeasured; measure it on the examples first.
- Fail points: an allocation that becomes infeasible, noticed as a dropped candidate with the reason; a forbidden overlap, noticed by the B40 rows.
- Order and dependencies: after B2 and B39.
- Verification: B40 rows pass; output offset equals input offset in the `--log` allocation of a tied pair.

## B39 Pair-relation record in the typed handoff

- What: the instruction model takes, per instruction, each input's relation to the output and the tie from the front end's record (L35, L36), replacing `inplace: list[bool]` (`instruction_info.py:15,112`).
- What: the typed FFI carries one relation per input to the allocator in place of `get_rhs_inplace` (`cpp/malloc/include/instructions.h:44,145,279`).
- What: address-naming constraints, marker expansions included, still reach the solver as constraints (`generator.py:65-68`); the backend writes no second copy of a relation's text.
- Preserves: the tie flags that `inplace_children` (`ir/egraph.rs:130`) gives the copy pass.
- Benefits: the allocator works on tensors, so it reads the relation, not constraint text; the oracle and the backend start from one list.
- Caveats: a tensor that feeds two inputs of one instruction takes both relations; a front end without the record fails the contract version check (U2).
- Fail points: the backend ignores the record and allocates by constraint text alone; noticed by the B40 rows.
- Order and dependencies: after B2 and L35; before B30.
- Verification: an instruction-model test yields the right relation for each of the five declarations and the default.

## B40 Allocation rows per relation

- What: per relation, a kernel whose input dies at the instruction; assert the allocation overlaps only as the relation permits and the oracle agrees on the assembled program (Q6.2, Q9.4).
- What: a tied input with another reader gets a copy; a tied operand pinned in HBM is an error.
- Preserves: the existing allocator tests.
- Benefits: a forbidden overlap cannot return unnoticed.
- Caveats: the may-overlap row needs a synthetic ISA specification with equal start and smaller output (RVV narrowing model).
- Fail points: rows pass while the exemption of `constraint.cc:214-230` or the missing touch constraint is back; the rows must fail then.
- Order and dependencies: after B30; with T.
- Verification: each row fails when B30 is reverted.
