# Search: determinism, rule guards, budget

`search` runs saturation and extraction under a one-second wall clock (`TIME_LIMIT`, `generic/src/lib.rs:18`) and stops on no improvement for ten times the last best (`pipeline.rs:155-180`, `SATURATION_FACTOR`), so the chosen candidate can depend on machine speed.
Rules 2, 3, 4 and 7 have a bare-variable LHS (`(?x) => ...`, `ir2ir_rewrites.txt` lines 3, 4, 5, 8) and fire on every class, so saturation ends at `NODE_LIMIT` 10000 (`lib.rs:20`).
egg's default `BackoffScheduler` is already in force (`Runner::default()`, `pipeline.rs:491-495`).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B29 | Deterministic search mode and `search` as a library function | -- | M | Q1 |
| B32 | Guards on bare-variable IR-to-IR rules; phases only if node-limit stops remain | B8, B29 | M | Q2 |
| B34 | Search budget measurement and candidate deduplication | B29, B10 | S | Q2 |

## B29 Deterministic search mode

- What: a flag that bounds search by iteration and candidate counts instead of `TIME_LIMIT` and `SATURATION_FACTOR`; `search` returns `Result`, and `main.rs` only parses flags.
- What: the deterministic mode is the only test and CI mode; the wall-clock mode stays the user default (owner decision 27, Q4.2).
- What: one equivalence check, inside B29, compares the two modes on the examples; if they differ, goldens are re-recorded once in a commit of their own.
- Preserves: the wall-clock mode; stop reasons and their messages.
- Benefits: the B3 gate can compare the chosen assembly across runs and machines.
- Caveats: where the wall clock is hit, the choice may vary between runs, but never to a wrong kernel (owner decision 27).
- Fail points: a hidden nondeterminism, such as hash-map iteration order; noticed by two runs that differ.
- Order and dependencies: first item of the plan.
- Verification: two runs on two machines give identical assembly and `--log` files.

## B32 Rule guards, then phases if needed

- What: guard rules 2, 3, 4 and 7 on the ISA having a consumer of the operator they produce (Q4.10).
- What: if B29 still records node-limit stops after the guards, run expansion and selection rules as separate phases, each its own egg `Runner`.
- Preserves: every rule and its condition; egg as the engine; no egglog.
- Benefits: saturation stops for a stated reason, not at the node limit.
- Caveats: a phase order can lose a candidate that interleaved firing finds, so phases are a behavioral step.
- Fail points: a cost rise on one example; noticed by the B3 cost comparison.
- Order and dependencies: after B8 names the rules and B29 makes runs comparable.
- Verification: equal or lower cost on every example; stop reason is saturation or iteration limit, not node limit.

## B34 Search budget and candidate deduplication

- What: report where compile time goes (saturation, extraction, allocation, cost) per pass under `--verbose`; skip a candidate whose content equals one already allocated.
- What: a CI test records compile time per example on the standard CI machine; each kernel over 1 s is a named exception (owner decision 27).
- What: a kernel that finishes under the bound never fails because the bound is longer; the wall-clock stop returns the best candidate found.
- Preserves: candidate numbering in `--log` for the candidates kept.
- Benefits: no repeated OR-Tools runs on identical candidates; budget limits set from measurement.
- Caveats: the content key must include the buffer of every node.
- Fail points: two distinct candidates hashed equal; noticed by a test with two candidates that differ only in buffer.
- Order and dependencies: after B10 produces candidates in one place.
- Verification: allocation count per kernel drops; B3 gate identical; the exception list matches the timing test.
