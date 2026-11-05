# Search: determinism, rule phasing, budget

`search` runs saturation and extraction under a one-second wall clock (`TIME_LIMIT`, `generic/src/lib.rs:18`) and stops on no improvement for ten times the last best (`pipeline.rs:155-180`, `SATURATION_FACTOR`), so the chosen candidate can depend on machine speed.
Rules 2, 3, 4 and 7 have a bare-variable LHS (`(?x) => ...`, `ir2ir_rewrites.txt` lines 3, 4, 5, 8) and fire on every class, so saturation ends at `NODE_LIMIT` 10000 (`lib.rs:20`).
Extraction and allocation run after every iteration (`N` is 1, `lib.rs:24`); after saturation the loop re-runs extraction over an unchanged e-graph (`pipeline.rs:544-547`).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B29 | Deterministic search mode and `search` as a library function | -- | M | Q1 |
| B32 | IR-to-IR rule phases | B8, B29 | M | Q2 |
| B34 | Search budget measurement and candidate deduplication | B29, B10 | S | Q2 |

## B29 Deterministic search mode

- What: a flag that bounds search by iteration and candidate counts instead of `TIME_LIMIT` and `SATURATION_FACTOR`; `search` returns `Result`, and `main.rs` only parses flags.
- Preserves: the wall-clock mode as the default; stop reasons and their messages.
- Benefits: the B3 gate can compare the chosen assembly across runs and machines.
- Caveats: the deterministic budgets must reach the same choice as the default mode on the examples, or the goldens differ.
- Fail points: a hidden nondeterminism, such as hash-map iteration order; noticed by two runs that differ.
- Order and dependencies: first item of the plan.
- Verification: two runs on two machines give identical assembly and `--log` files.

## B32 IR-to-IR rule phases

- What: measure node growth per IR-to-IR rule, then run expansion rules (bare-variable LHS) and selection rules as separate phases, each its own egg `Runner` with its own limits.
- Preserves: every rule and its condition; egg as the engine.
- Benefits: saturation stops for a stated reason, not at the node limit.
- Caveats: a phase order can lose a candidate that interleaved firing finds; compare costs on all examples.
- Fail points: a cost rise on one example; noticed by the B3 cost comparison.
- Order and dependencies: after B8 names the rules and B29 makes runs comparable.
- Verification: equal or lower cost on every example; stop reason is saturation or iteration limit, not node limit.

## B34 Search budget and candidate deduplication

- What: report where compile time goes (saturation, extraction, allocation, cost) per pass under `--verbose`; skip a candidate whose content equals one already allocated.
- Preserves: candidate numbering in `--log` for the candidates kept.
- Benefits: no repeated OR-Tools runs on identical candidates; budget limits set from measurement.
- Caveats: the content key must include the buffer of every node.
- Fail points: two distinct candidates hashed equal; noticed by a test with two candidates that differ only in buffer.
- Order and dependencies: after B10 produces candidates in one place.
- Verification: allocation count per kernel drops; B3 gate identical.
