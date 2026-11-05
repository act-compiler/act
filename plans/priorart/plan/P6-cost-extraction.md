# P6 Cost-aware extraction

## What

- Give the extractor a per-e-node cost from a declared per-instruction latency or from the C cost model, and prune `extract_slow` enumeration by cost instead of a node count alone.
- Keep `c_bridge` as the final arbiter over the surviving candidates.

## Benefits

- Fewer candidates written and scored; compile time bounded by cost search, not by `SLOW_LIMIT_CUTOFF`.

## Caveats

- Buffer placement affects cost, so cost is per e-node and buffer, which the fast extractor already keys on.
- Builds on B10's single extractor module.

## Fail points

- A cheaper candidate pruned because its per-node estimate was high, noticed by comparing against exhaustive extraction on the golden corpus.

## Order and dependencies

- Depends on P4, B10.

## Verification

- Best C cost on the golden corpus is not worse than before for any kernel.
- Candidate count per kernel is reported and lower than before.
