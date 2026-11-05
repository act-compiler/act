# Owner decisions

Decisions the owner made during the meeting; every plan and ruling follows them.

| No. | Topic | Decision |
| --- | ----- | -------- |
| 1 | Where the plans live | One branch, `future-planning`, on act-compiler/act, under `plans/`; nothing else is edited |
| 5 | Edits outside `plans/` | None; the meeting is planning only |
| 6 | Current tutorial | None; `pldi26` is the latest and dates from about ACT 1.2 |
| 10 | Breaking changes | Allowed, batched into one coordinated major release (TAIDL v3 and matching generator releases) |
| 11 | Capacity | Size the roadmap for 3-4 engineers over two quarters |
| 12 | GitHub issues | None |
| 13 | Pull request | None |
| 14 | Terminology precedence | The owner's papers first when precise, otherwise StableHLO and MLIR |
| 15 | Accelerator scope | Tensor accelerators plus GPU tensor-core instructions (mma, wgmma, TMA); no SIMT thread model |
| 16 | Undefined behavior | None in v3: every run-time violation is a defined error or a defined behavior under one simple norm that is easy to explain |
| 17 | In-place mark | The output reuses the marked input's storage |
| 19 | API vocabulary | A data model has buffers (not "tensor buffers") and state (not "initial state"); `hbm` stays `hbm` |
| 20 | Instruction's view of a buffer | "Region" (not "data slice") |
| 21 | State reference prefix | `@s.` stays |
