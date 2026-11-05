# P10 Reusable check extractors

## What

- Generate shape recovery, dtype check, attribute solving and constant detection once as named Rust functions, and compose them per rule, replacing the per-rule bodies `generate_precond_function` emits.

## Benefits

- Generated Rust shrinks; a bug is fixed in one function.

## Caveats

- Builds on B13's selectable-instruction model; the extractors are its methods.

## Fail points

- A composed check weaker than the old inline one, noticed by the golden corpus and by P7 rejection examples.

## Order and dependencies

- Depends on P3, B13.

## Verification

- `isel_ir2isa_rewrites_rs_generator.py` emits no shape or dtype check text; the Rust helpers have unit tests.
