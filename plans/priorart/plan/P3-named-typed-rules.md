# P3 Named typed rewrite rules with priorities

## What

- Replace `ir2ir_rewrites.txt` and `ir2isa_rewrites.txt` numbered strings and the three parallel `match name` tables in `applier.rs` with one generated Rust value per rule: name, pattern with typed variables (value, constant, attribute), side conditions, result data, priority.
- Reject at generation time a rule whose pattern can join values of unlike shape or element type, so `TensorInfo::merge` never panics.

## Benefits

- Rules are greppable by name; a priority is declared, not an e-node-order accident.
- Removes the string split and `unwrap` parsing in `get_applier`.

## Caveats

- Builds on B6 and B8; coordinate so the Rust value shape is shared.
- Typing pattern variables needs shape variables in patterns, a small extension to egg `Pattern` use.

## Fail points

- A rule that was selected before stops matching after typing, noticed by the golden corpus from B3.
- Two rules at equal priority still overlap, noticed by a generation-time overlap report.

## Order and dependencies

- Depends on B6, B8; feeds P4, P5, P10, P12.

## Verification

- `applier.rs` has no `match name` on strings.
- A rule listing command prints name, priority and operand kinds for every rule.
