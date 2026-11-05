# P5 Skipped-pattern and stop-reason reports

## What

- At generation time, list every instruction without a selectable rule and the reason, as GlobalISel's `--warn-on-skipped-patterns` does.
- At compile time, print each phase's egg `StopReason` and, in a debug mode, egg explanations for each selected instruction.

## Benefits

- The owner sees why an instruction is never chosen without reading Rust.
- Diagnostics track gets a stable set of message ids.

## Caveats

- Explanations cost memory; keep them behind a flag.

## Fail points

- A report that names a reason wrongly, noticed when a case study contradicts it.

## Order and dependencies

- Depends on P3, D.

## Verification

- A spec with one unselectable instruction produces exactly one named report line.
- Every compile log contains one stop-reason line per phase.
