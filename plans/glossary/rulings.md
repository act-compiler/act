# Rulings

Each row settles one disagreement between the language glossary (`taidl/docs/glossary.md`, `work/spec/glossary.md`), the terminology reference (`work/terminology/`), or a track's objections.
The owner's rule decides each row: the papers' precise term first, then the StableHLO or MLIR term.
Renames land together in the v3 release; the identifiers are listed in `terms.md`.

## Description and memory

| Conflict | Ruling | Reason |
| --- | --- | --- |
| One storage unit: "data model" (glossary) or "tensor buffer" (terminology #1) | "tensor buffer", short "buffer"; "data model" names the set; `add_data_model` becomes `add_tensor_buffer`. | [ACT] Sec. 3.2 and [TAIDL] Sec. 3.1 call the set "data model" and each unit a "tensor buffer"; the paper term is precise. |
| Backend objection: Rust `Buffer` enum names data models | Withdrawn by the ruling above: `Buffer` already matches "tensor buffer". | The enum now uses the canonical short form. |
| Oracle objection: errors say "buffer" for a data model | Keep "buffer" in those errors; plan item O15 is no longer needed for this word. | "buffer" is the canonical short form. |
| Persistent integer: "state variable" (glossary) or "control register" (terminology #2) | "control register"; `add_initial_state` becomes `add_control_register`; "register" alone stays banned. | [TAIDL] Sec. 3.1 defines the data model as tensor buffers and control registers; hardware registers such as AMX tiles are tensor buffers, so the qualifier is required. |
| `d0`: "off-chip memory" (glossary), "HBM" (papers, `hbm` keyword), or "global memory" | "global memory"; "HBM" only for a concrete chip; the `hbm` keyword becomes `global_memory`. | Both papers define `d0` as "global memory" ([ACT] Sec. 3.2, [TAIDL] Sec. 3.1) and use HBM as an alias; HBM is a memory technology, wrong for AMX hosts, and CUDA and MLIR `gpu` also say global memory. |
| Oracle objection: keep `hbm` as a public keyword, as `var_type` is kept | Rejected: v3 renames both `hbm` and `var_type`. | Breaking renames are allowed in v3, so no exemption is needed. |
| Region dimensions: "addressed dimensions" (glossary) or "access dimensions" (terminology #3) | "access dimensions"; parameter `dimensions` and field `access_dim` become `access_dims`. | [ACT] Sec. 3.2 defines "access dimensions (S0)". |
| Shape read whole: "unit shape" (glossary) or a paper term | "base element" for the unit, "base element shape" for its shape; `unit_dim` becomes `base_element_dims`. | [TAIDL] Sec. 3.1 defines tensor buffers as arrays of base elements; [ACT] Sec. 3.2 calls it the granularity of data access, which names no shape. |
| Element type: "element type" (glossary) or "data type", "dtype" (papers) | "element type"; `var_type` becomes `element_type`. | The papers use "dtype" loosely, not as a defined term; [SHLO] and [MLIR] say element type. |
| Description name: "ISA description", "TAIDL definition" ([TAIDL]), "TAIDL specification" (repos) | "ISA description". | [ACT] title and Sec. 3.2 use it 54 times; it is the newer paper's term. |
| Act objection: say "spec" for the ISA description | Rejected: "the spec" names only the language document. | One word cannot name both the language and a program written in it. |
| Act objection: `targets/<Name>/` uses the banned "target" | Rename the generated directory to `accelerators/<name>/` in both generators. | The ban stands, and v3 allows the rename. |
| Scratchpad, buffer, HBM in tutorials (act objection) | The glossary wins; "scratchpad" stays only for the hardware a tensor buffer models. | [ACT] Sec. 3.2 uses "scratchpad" for hardware and "tensor buffer" for its model. |
| Scope: GPU tensor-core instructions | The same terms apply: shared memory and register fragments are tensor buffers, device memory is global memory. | One vocabulary covers both kinds of accelerator in scope; vendor words such as "fragment" stay only in a vendor quote. |

## Instructions

| Conflict | Ruling | Reason |
| --- | --- | --- |
| "instruction" or "abstract instruction" / "concrete instruction" | "instruction" for the declared one; "concrete instruction" once attributes have values; "abstract" only where the contrast matters. | [ACT] Sec. 3.2 defines both. |
| Attributes: "computational / addressing" ([ACT]) or "calling attributes" ([TAIDL]) | "computational attribute", "addressing attribute"; `computation_attr`, `comp_attr`, `parameters` become `computational_attrs`, `addressing_attrs`. | [ACT] Sec. 3.2 is newer and splits the two kinds. |
| Terminology #7: keep the API name `computation_attr` | Overruled: rename in v3. | The identifier should spell the canonical term; v3 allows it. |
| Oracle objection: front end fields `parameters`, `comp_attr` | Rename in the front end first, generators follow in the same v3 release. | One coordinated release removes the mismatch. |
| Data an instruction touches: "region" (glossary, terminology #5) or "data slice" | "data slice", never "slice" alone; `Region` becomes `DataSlice`. | [ACT] Sec. 3.2 defines data slices and data slice address maps; the qualified term is precise and does not collide with the `slice` operator. |
| Oracle objection: add "region value" | Use the existing "input value / output value" instead. | Those two terms already name `%In<k>` and `%Out<k>`. |
| Extent words: "start, size" (glossary) or "start, end" ([ACT] `d[Is:Ie]`) | "start" and "size". | The API takes sizes; [ACT] Fig. 6 also writes `len=`, and [SHLO] `dynamic_slice` takes `slice_sizes`. |
| Constraint: "constraint" (glossary) or "validity constraint" ([ACT]) | "validity constraint", short "constraint" within the spec; `add_constraints` keeps its name. | [ACT] Sec. 3.2 term; the backend also has allocation constraints, so the full term is used there. |
| Backend objection: "precondition" for a rewrite condition | Allowed for generator internals; banned only for a validity constraint. | The bans cover TAIDL concepts only. |
| State change: "state update" (glossary) or "control register assignment" ([TAIDL]) | "control register assignment"; keyword `update` becomes `assignments`. | [TAIDL] Sec. 3.2 names it and follows from the control register ruling. |
| "semantics" or "instruction semantics" | "instruction semantics", short "semantics". | Section title in [TAIDL] Sec. 3.2 and [ACT] Fig. 6. |

## Semantics language

| Conflict | Ruling | Reason |
| --- | --- | --- |
| One `%x = ...` text: "line" (glossary) or grammar rule `instruction` (terminology #6) | "operation"; grammar rule `instruction` becomes `operation`. | The papers name no unit here; [SHLO] and [MLIR] call it an operation, and the rule name collides with the ISA instruction. |
| Its kind: "operation word" (glossary) or "tensor operator" | "tensor operator"; grammar token `OPERATION` becomes `OPERATOR`. | [ACT] Sec. 3.2 Table 1 and [TAIDL] Sec. 2.4 say tensor operators. |
| "reference operation" | "reference operator". | Matches "tensor operator". |
| `broadcast` vs XLA `Broadcast` (terminology #8) | Keep `broadcast`; its reference operator is `broadcast_in_dim`. | [XLA] `Broadcast` adds leading dimensions only. |

## Tools, compilation, and errors

| Conflict | Ruling | Reason |
| --- | --- | --- |
| Compiled output: "kernel" (glossary) or "assembly code" (terminology #4) | "assembly code" for the compiler backend output; "machine code" banned. | [ACT] Sec. 3.2 defines it as concrete instructions plus a constant tensor. |
| Compiler backend input | "tiled kernel"; "tensor kernel" and bare "kernel" banned for it. | [ACT] Sec. 1 and Sec. 3.1 use it 48 times. |
| Act objection: "kernel" for an instruction program | Accepted as "kernel function", short "kernel": a `@kernel` Python function run on a test oracle. | [TAIDL] Sec. 6 defines a kernel function as the Python form of assembly code. |
| Oracle objection: no term for the generated artifact | "test oracle" names it; "oracle" alone stays for repo, package, and directory names. | [TAIDL] title and Sec. 2.3 use "test oracle" 71 times. |
| Act objection: add oracle, backend, ISA, ecosystem | Add "ACT ecosystem"; oracle and backend fold into "test oracle" and "compiler backend"; ISA needs no entry. | [ACT] Sec. 1 uses "ecosystem"; the bare forms stay banned in prose. |
| Generator of compiler backends | "generator" for the general case; ACT is a "compiler backend generator". | [ACT] Sec. 3.3. |
| Backend phases and data | "instruction selection", "memory allocation", "pii graph", "e-graph", "e-class", "e-node", "rewrite rule", "extraction", "identity instruction", "sound", "complete". | All are [ACT] terms (Sec. 3.3 to Sec. 6). |
| Backend objection: bans apply to kernel-IR and e-graph terms | Accepted: the bans cover TAIDL concepts only; "off-chip offset" may become "global memory offset" in the backend's own docs. | The XLA-HLO tensor and rewrite-pattern variables are not TAIDL concepts. |
| "refuse" in repos (terminology words list) | "reject" everywhere; `test_refusals.py` becomes `test_rejections.py`. | The glossary chose it, and every track accepted. |
| Oracle objection: rejection at kernel compile time | "reject" covers every stage; the sentence names the stage. | No new term is needed. |
| Backend objection: status words in `docs/coverage.md` | Add "coverage status": `supported`, `partial`, `rejected`. | Both generators already use them. |
| Oracle objection: where implementation-defined choices live | Each generator's `docs/coverage.md`. | Both generators have that file. |
