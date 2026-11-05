# P8 Normative execute function and reference-interpreter status

## What

- Write S5.4 as a pseudo-code `execute(instruction, state, memory)` with numbered steps, in Sail's style.
- Declare the oracle the reference interpreter of that function and state the backend's obligation: every emitted program equals the oracle's result on the kernel.

## Benefits

- The formal track has one function to formalize; the test track has one property to check.

## Caveats

- The oracle's known partial rows (float constant rounding, `triangular_solve` at f16) must be listed as deviations.

## Fail points

- A step order that the oracle does not implement, noticed by a P7 example on state-dependent shapes.

## Order and dependencies

- Depends on P2, O, F; feeds P11.

## Verification

- Each oracle check cites a step number; each coverage.md partial row cites the step it deviates from.
