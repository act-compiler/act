# egg and egglog

## What it is

- egg is the Rust e-graph library: `EGraph`, `Rewrite` with `Pattern`s, `Analysis` for per-class facts, `Runner` with `with_iter_limit` (30), `with_node_limit` (10000), `with_time_limit` (5 s), `BackoffScheduler`, `StopReason` (`Saturated`, `IterationLimit`, `NodeLimit`, `TimeLimit`), `Extractor` with a `CostFunction`, and explanations via `with_explanations_enabled`.
- egglog is its successor: a Datalog with equality saturation, relational e-matching, functions with merge, rulesets with run schedules (`run`, `saturate`, `seq`, `repeat`), and built-in `extract`.
- Sources: https://docs.rs/egg/latest/egg/, https://docs.rs/egg/latest/egg/struct.Runner.html, https://arxiv.org/abs/2304.04332, https://egglog-python.readthedocs.io/latest/.

## TAIDL today, for comparison

- The backend uses egg: one `Runner` with `NODE_LIMIT` 10000, `ITERATION_LIMIT` 30, `TIME_LIMIT` 1 s (act-backend/act_backend/generic/src/lib.rs:18-30, pipeline.rs:491-495).
- Extraction is custom: `extract_fast` picks one PII graph with each e-class once per buffer and `extract_slow` enumerates repeat-free PII graphs up to a node limit (act-backend/act_backend/generic/src/isel/extractor/fast.rs:13-40, slow.rs:14-50), then an external C cost model scores whole candidates (act-backend/act_backend/generic/src/cost/c.rs:6-24).
- `TensorInfo` is the only analysis: shape, dtype and a constant flag (act-backend/act_backend/generic/src/ir/egraph.rs:371-416).

## Borrow

- Report `StopReason` to the user every run, with the limit hit; pipeline.rs:520-550 already branches on it, so surface it in a diagnostics line.
- Rulesets and schedules from egglog: run expansion rules to saturation or a limit, then mapping rules, instead of one mixed set.
- Explanations: enable egg explanations in a debug mode so a selected instruction can be traced to the rewrite chain that produced it.
- Cost in extraction: give the extractor a per-e-node cost derived from the C cost model, so candidate enumeration in `extract_slow` is bounded by cost rather than by node count alone.
- Analyses as facts: layout, buffer and constant facts are lattice analyses; egglog's merge functions are the model for adding them without panics.

## Avoid

- Migrating to egglog wholesale: it is a different runtime and language; borrow the schedule and analysis ideas inside egg first.
- Node and time limits as the only termination story: a 1 s limit makes results machine-dependent; prefer saturation of a phase with a stated bound.

## Relevance

- High: the backend is built on egg.
