# C4 Mode-dependent semantics

- What: No syntax; `select` over the modes is the idiom.
- Evidence: 5 public sites (gap rank 7); the one failure case is the `wgmma` transpose flags, where the arms read differently shaped regions and `select` cannot join them.
- Caveats: `select` computes every mode on every issue; the backend loses per-mode selection, which is a B-track cost.
- Fail points: An arm left unreachable by constraints hides a wrong semantics; noticed by oracle coverage counting arms hit by random tests.
- Order and dependencies: C11 runs the sketches; re-admission needs a second ran failure case.
- Verification: Gemmini `mvout_acc` and Tensix `mvmul` run with `select` and the oracle covers every mode.
