# In-house C++ e-graph library and the engine swap

Owner decisions 45 and 46 replace egg with an in-house C++ e-graph library, reached first from the Rust driver through the B2 C ABI (Q12.3).
The library is a standalone C++ library with no MLIR header; extraction stays in the backend, since candidates are ranked by the external C cost model and keyed by (class, buffer).
The library drops explanations, a run-time rule language, Datalog functions, threading, `BackoffScheduler`, and the `modify` hook.

| ID | Item | Depends on | Size (S/M/L) | Quarter |
| --- | --- | --- | --- | --- |
| B49 | E-graph library core: arena, typed key, union-find, deferred rebuild, class data, read-only view, property tests | B29, B31, B7, B9, B26, B57 | L | later |
| B51 | Matcher and rules: backtracking per operator, early preconditions, matched ids, applied memo, C++ rule emitter | B49, B6, B8 | L | later |
| B52 | Dual-engine harness and swap through the B2 C ABI, three-tier gate, egg deleted | B51, B2, B3, B10, B32, B34, B35 | M | later |

## Swap gate

| Tier | Kernels | Passes when |
| --- | --- | --- |
| 1 | none: library property tests without egg | congruence, hash-cons, and analysis-conflict properties hold |
| 2 | kernels that saturate or stop at the iteration bound | equal canonical e-graph dumps per iteration, then identical chosen assembly and `--log` files |
| 3 | kernels that still stop at `NODE_LIMIT` | chosen cost no worse than egg's and an equal oracle result |

- Tier 2 needs egg's `SimpleScheduler` (B32) and canonical extraction order (B10).
- Generic join stays a later row: it comes forward only when a B34 profile after B32 puts matching above one third of search time.

## B49 E-graph library core

- What: arena e-nodes with the typed payload (B31) in the hash-cons key; union-find with path compression and union by size; deferred rebuild.
- What: one class-data record (B7) with `make` and a `merge` that is a coded fault on disagreement and reports change for both sides.
- What: a read-only class view, insertion-ordered containers, and three limits (iterations, nodes, time) with stop reasons.
- Benefits: deferred rebuild is the one egg algorithm with measured value; the payload joins the key, so patterns can match payloads.
- Caveats: no MLIR header, no threading, no explanations; ACT's e-graphs stay under `NODE_LIMIT` 10000 (`generic/src/lib.rs:20`).
- Fail points: a congruence missed after a deferred rebuild; noticed by the tier-1 property tests.
- Order and dependencies: after B29, B31, B7, B9, and B26; runs beside B47 and B48 in wave 8.
- Verification: property tests for congruence, hash-cons, and analysis conflict pass without egg.

## B51 Matcher and rules

- What: a backtracking matcher over a per-operator index; each precondition is evaluated as soon as its variables bind; matches yield e-node ids.
- What: an applied-substitution memo stops a rule re-firing on a match it has applied.
- What: a C++ rule emitter reads B6's language-neutral rule records (name, pattern, guard tree, right-hand side) beside the Rust emitter.
- Preserves: each rule's pattern and guard as B6 and B8 name them.
- Benefits: the re-application waste of rules 2, 3, 4, and 7 goes with B32's guards and the memo; once-per-class firing needs no `modify` hook.
- Caveats: generic join waits for its trigger (see the swap gate).
- Fail points: a guard evaluated before a variable binds; noticed by B8's near-miss tests on the library.
- Order and dependencies: after B49, B6, and B8.
- Verification: every B8 test passes on the library; the emitted C++ rules match the Rust ones rule for rule.

## B52 Dual-engine harness and swap

- What: a harness runs egg and the library on each example under B29's deterministic mode and checks the three-tier gate.
- What: the Rust driver then calls the library through the B2 C ABI; egg and its rule glue are deleted.
- Preserves: the Rust driver, extraction, the allocator, and `--log` files.
- Benefits: owner decision 45 read literally: the engine changes before the driver, under one gate.
- Caveats: tier 2 compares only kernels that saturate or stop at the iteration bound.
- Fail points: a cost rise on a node-limit kernel; noticed by tier 3.
- Order and dependencies: after B51, B2, B3, B10, B32, B34, and B35.
- Verification: all three tiers pass on every example; no `egg` in `Cargo.toml`.
