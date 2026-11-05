# Long-term platform tests

T39 to T41 test the long-term platform of roadmap waves 8 to 10: the StableHLO importer, the torch path, and the C++ driver on fsim (Q12.4).
They reuse the differential harness of T13 and the program runner of T29.

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter |
| --- | --- | --- | --- | --- |
| T39 | StableHLO corpus exported once from jax, differential against the HLO path | B47, T13 | S | later |
| T40 | Torch tests: eager PyTorch against ACT on the oracle per operation and dtype, coverage table | A29, T29 | M | later |
| T41 | Platform gate: B3 on the C++ driver, oracle on fsim | B56, O36 | M | later |

## T39. StableHLO corpus

- What: each example kernel is exported once from jax as a StableHLO portable artifact at U15's version and checked in beside its HLO text.
- What: a differential test compiles each artifact through B47 and its HLO form through the reader, and expects equal assembly.
- Benefits: the importer is tested on the kernels the backend already compiles, with no torch dependency.
- Caveats: the export is regenerated only when U15's version moves, so jax is not a test-time dependency.
- Fail points: a jax export that differs from the HLO kernel's semantics; noticed by an oracle run of both forms.
- Order and dependencies: after B47 and T13.
- Verification: every example has an artifact; each pair compiles to equal assembly.
- Breaking: no.
- Files: new artifacts beside the backend's example kernels; one test module in act-backend.

## T40. Torch tests

- What: per operation and dtype of the input specification, a torch module runs in eager PyTorch and through A29; the ACT assembly runs on the generated oracle and matches eager within the per-dtype tolerance.
- What: a coverage table lists each core ATen operation as `supported`, `partial`, or `rejected` with its code, generated from the test rows.
- What: a module with an operation outside the input specification stops with a coded error in strict mode and prints each undelegated node under A30's partition.
- Benefits: the torch path is checked end to end against PyTorch's own results.
- Caveats: one CI row per supported torch minor (A29); per-dtype tolerances come from the numerics norms (T26).
- Fail points: a tolerance loose enough to hide a wrong lowering; noticed by an integer-dtype row, which must match exactly.
- Order and dependencies: after A29 and T29.
- Verification: every row passes on each supported torch minor; the coverage table has no operation without a row.
- Breaking: no.
- Files: a new test module in act; the coverage table under act's docs.

## T41. Platform gate

- What: the B3 gate runs on the C++ driver (B56): identical chosen assembly and `--log` files in deterministic mode, then the oracle-differential run.
- What: the oracle side runs on fsim (O36): every golden and norm row holds.
- Benefits: one gate certifies that the Rust crate and the jax run path are gone with no change in result.
- Caveats: fsim's arithmetic is XLA's inside the plugin, so the norm rows hold as measured.
- Fail points: a golden re-recorded to hide a driver change; noticed by review of any golden diff in the gate's commit.
- Order and dependencies: after B56 and O36.
- Verification: two runs on two machines are identical; every golden and norm row passes.
- Breaking: no.
- Files: the B3 test entry and the oracle's golden tests, pointed at the new driver and run path.
