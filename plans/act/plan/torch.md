# PyTorch entry points and partition

Owner decision 44 asks for a usable PyTorch backend; panel 12 schedules the compile path only (Q12.2).
A29 and A30 wrap the backend's torch adapter (B53) for users; an eager PrivateUse1 device and a PJRT plugin are later rows with no wave.
The torch package is strict by default under owner decision 40, sets `dynamic=False`, and never falls back silently.

| ID | Item | Depends on | Size (S/M/L) | Quarter |
| --- | --- | --- | --- | --- |
| A29 | `torch.compile` and `torch.export` entry points, artifact-hash cache, one released torch per release, CI row per minor | B53, A23 | M | later |
| A30 | Partition opt-in: supported subgraphs to ACT, every undelegated node printed | A29, B46 | M | later |

## A29 Torch entry points

- What: a `torch.export` entry takes an `ExportedProgram` through B53; a `torch.compile` backend registers by entry point and wraps B53 through `aot_autograd`.
- What: compiled kernels are cached by the portable artifact's hash.
- What: each act release pins one released torch, with a CI row per supported torch minor; A16 records them.
- Benefits: a PyTorch user reaches the backend through the two standard doors.
- Caveats: the `aot_autograd` hook is private and torch minors ship about every two months, so each minor is retested; PyTorch/XLA is archived, so `torch_xla` plays no part.
- Fail points: a torch minor that changes the hook; noticed by its CI row before release.
- Order and dependencies: after B53 and A23.
- Verification: T40 passes on each supported minor; a second compile of one module hits the cache.

## A30 Partition opt-in

- What: an opt-in flag sends supported subgraphs to ACT and runs the rest in eager PyTorch.
- What: every undelegated node is printed with its reason, in the form of the B46 exclusion report.
- What: without the flag, the strict default stops with a coded error naming the first unsupported node.
- Benefits: real models run while the backend covers few core ATen operations.
- Caveats: legal under owner decision 40 because the compiler only sees supported subgraphs and the report keeps the split visible.
- Fail points: a node delegated to eager without a printed line; noticed by a T40 row that counts printed nodes against the graph.
- Order and dependencies: after A29 and B46.
- Verification: a model with one unsupported node runs under the flag and prints exactly that node; without the flag it stops naming it.
