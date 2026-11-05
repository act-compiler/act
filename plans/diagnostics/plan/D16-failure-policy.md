# D16 Failure policy (decision 40)

- What: write `failure-policy.md` into `scheme.md`: the allowed stop per tool, the Input class extended to assembly and binary input (`KO021` to `KO040`), and `KB041` for a valid kernel no candidate implements, whose message lists excluded instructions as a hint outside the match key (D18).
- What: in the backend, a candidate failure, including an `I`-coded one, becomes a drop record (candidate id, `code`, reason) in a drop log; the search continues and the run summary counts the drops.
- What: replace `StopReason::Other` and the `timeout()` stop (`pipeline.rs:551-563`) with drop or `KB041` endings; the wall-clock bound only ends the search.
- What: the oracle's assembly and binary reader raises `KO021` to `KO040` naming the line or word offset; the decode error for a word matching no instruction is one of them.
- What: seed `registry-seed.md` with the layer rows `RF052` to `RF057` and the `KO` and `KB041` rows.
- Benefits: no stop without a cause; each stop names what the user or architect fixes.
- Caveats: lands with D7 (backend runtime) and D8 (oracle); the drop log shape is shared with T36.
- Fail points: a drop that hides a wrong answer; the same-assembly test of T36 and the T13 simulation catch it.
- Order and dependencies: D1, D7, D8, D18, B2; Q2.
- Verification: `act-backend/tests/test_endings.py` has a row per ending class of `failure-policy.md`; `act-oracle` has one test per `KO021` to `KO040` code, as D8 does for its codes; no `I` code stops a run.
- Breaking: yes: the backend exit status and stderr for a dropped candidate change, released with v3.
