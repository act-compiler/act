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
| 22 | In-place mark | Overlap alone does not settle read-after-write: in `r0 = r1 + r0` an implementation may or may not reuse the source storage, and `r0 = r1 + r2` allows `r2` to equal or differ from `r0`; `set_inplace` stays unless the panels show it is truly redundant |
| 23 | Oracle execution | The oracle will later move to an all-C/C++ simulator through fsim-lib (act-compiler/fsim-lib); this plan does not pivot to it yet, and records the fsim analysis as the follow-up direction |
| 24 | In-place across buffers | An in-place mark that pairs an input and an output in different buffers is rejected |
| 25 | Shape rules in the generated oracle | The generator copies the front end's shape module into each generated oracle, stamped with its taidl version |
| 26 | v3 versions | taidl, taidl-to, and act become 3.0; act-backend becomes 2.0 |
| 27 | Backend search limit | A wall-clock bound stays the user default and should almost never be hit; where it is hit, non-determinism is acceptable, but a wrong answer never is, and a kernel that compiles in 400 ms must not fail at 1 s; tests use the deterministic mode; most kernels compile within 1 s on the standard CI machine, with named exceptions |
| 28 | New dependencies | Allowed when their item lands: cbindgen (act-backend build), hypothesis and mutmut (tests); the StableHLO interpreter is not added |
| 29 | Generated backend code | rustfmt runs on the generated crate at generation time |
