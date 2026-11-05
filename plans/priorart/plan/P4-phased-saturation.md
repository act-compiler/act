# P4 Phased saturation schedule

## What

- Split rules into rulesets, expansion (slice through convert, reshape pairs, broadcast removal), mapping (ir2isa), cleanup, and run them in sequence, each with its own node and iteration limit and stop reason.
- Drop the wall-clock `TIME_LIMIT` as a correctness-affecting bound; keep it as a watchdog that reports.

## Benefits

- Smaller e-graphs per phase, the Isaria result; deterministic results across machines.
- Per-phase stop reasons make "instruction never selected" explainable.

## Caveats

- Some expansions are only useful after mapping exposes a shape mismatch; a second expansion pass may be needed.
- The `SATURATION_FACTOR` logic in pipeline.rs goes away; keep its progress reporting.

## Fail points

- A kernel that selected under one mixed run fails under phases, noticed by the B3 golden corpus.
- Phase limits too small for a large kernel, noticed by the stop-reason report naming the phase.

## Order and dependencies

- Depends on P3, B6; feeds P6.

## Verification

- Same assembly on two machines for the golden corpus.
- Each phase's stop reason appears in the log for every compile.
