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
| 30 | Markers | Markers such as `set_inplace`, `set_apart`, and `set_no_partial_overlap` stay as shorthand: each expands to the constraints it stands for, so users need not write those constraints by hand; this is the same mechanism as the implicit address-space constraints that every region gets (it fits within its buffer's bounds, and no address is negative), and a violated marker is handled exactly like any violated constraint |
| 34 | Attributes | An addressing attribute may be negative, and a region start is any expression over attributes (for example `1024 - @a.addr_out`); the implicit constraints apply to the evaluated address, not to attribute values; attribute names are plain variable names, and no tool infers meaning from a name |
| 35 | Attribute widths and encodings | The plan covers bit widths per attribute and binary-encoding annotations for an assembler; an ISA specification may hold attributes that never appear in the binary, tied to encoded attributes by constraints (for example addresses written as `B*8` with a constraint linking `B` to the encoded attribute `A`) |
| 38 | Layers of the ISA specification | Bit widths, signedness, and encodings are optional layers: by default attributes and operands are 64-bit integers, and an author who wants to be more exact adds the width layer and, where needed, the encoding layer on top of it |
| 39 | More generators and generator metadata | The system is easy to extend with new generators (for example act-allo, which generates an Allo specification from the ISA specification); an ISA specification may carry optional generator metadata at instruction, buffer, or accelerator level, which other generators ignore and which reaches that generator's entry point (for example `generate_allo()`) |
| 40 | Failure policy | Nothing fails without a genuine cause: the front end and the generators reject only problems the architect can correct, such as typos or an invalid ISA specification, with a clear error; a generated oracle stops with an error only on invalid assembly code or binary input and otherwise runs correctly; a generated backend never stops because one candidate fails (it drops that candidate with a recorded reason) and stops only when the input kernel (StableHLO, XLA HLO, torch-mlir, or torch.fx) is invalid or outside the input specification |
| 41 | Authoring surface | ISA specifications move off semantics written as text inside Python strings to a clean embedded language (for example a Python meta-DSL, or C++ or MLIR); nothing fancy, as few dependencies as possible, and comfortable for an architect to read and write |
| 42 | Check before generation | A check function validates an ISA specification on its own, without any generator; it always runs once before any generator, followed by that generator's own additional checks (for example the backend's machine-state requirements; none for the oracle) |
| 43 | Long-term platform | The plans include porting the stack to C++ on MLIR infrastructure, starting with the backend, so it reads StableHLO, MLIR, and torch-mlir inputs natively |
| 44 | Torch backend | The plans include a portable PyTorch backend built on that infrastructure |
| 45 | E-graph library | The backend later moves from egg through a Rust FFI to a lightweight C++ e-graph library built for tensor computations: union-find, fast e-matching, complex rewrites with little code, and native e-node metadata with preconditions over it, without proof machinery |
| 46 | E-graph library source | The C++ e-graph library is built in-house, independent of egg and egglog, which were designed more for proof systems than for compiler optimization; it keeps only the machinery compiler rewriting needs |
| 31 | Overlap and hazards | TAIDL assumes no hazard model such as read-before-write; the ISA author declares how an instruction behaves when its inputs and outputs overlap, and the language must not fail scenarios real architectures allow |
| 32 | The user's file | "ISA specification" |
| 33 | Releases | Public releases follow the act 2.x structure: each release pins released versions of its dependencies; no pre-release tags and no commit or branch pins in public; development happens in private repos |
