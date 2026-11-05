# Reconciled glossary

One canonical term per concept, for the TAIDL spec and the four repos `taidl`, `act-oracle`, `act-backend`, and `act`.
Docs, error messages, and user-visible names use the canonical term and none of its banned synonyms when they name a TAIDL concept.
The bans do not apply to a generator's own concepts, such as a rewrite condition, an XLA-HLO tensor, or an off-chip placement offset.
A short form in parentheses may replace the term once the full term has appeared in the same section.
The last column lists public identifiers, file names, and user-visible strings that change in the v3 release; private internals follow only when touched.

## Description and memory

| Term | Definition | Source | Banned synonyms | Identifiers to rename in v3 (file:line) |
| --- | --- | --- | --- | --- |
| ISA description | A Python program that builds one `Accelerator`. | [ACT] title, Sec. 3.2 | spec, ISA specification, TAIDL specification, TAIDL definition, ISA definition, model file, config | `act-backend/act_backend/spec_checks.py` -> `description_checks.py` |
| the spec | The TAIDL language document, sections S1 to S10. | current glossary | standard, reference manual | none |
| tensor accelerator (accelerator) | The `Accelerator` object an ISA description builds: its data model and its instructions; it covers tensor accelerators and GPU tensor-core instructions. | [ACT] title; [TAIDL] title | target, device, chip | generated directory `targets/` -> `accelerators/`: `act-oracle/taidl_to/generator.py:22,31`; `act-backend/act_backend/generator.py:95`; `act/tests/test_end_to_end.py:81` |
| data model | All storage units of an accelerator together: its tensor buffers and its control registers. | [ACT] Sec. 3.2; [TAIDL] Sec. 3.1 | data model (for one storage unit) | none; `Accelerator.data_model` (`taidl/taidl/accelerator.py:44`) already names the set |
| tensor buffer (buffer) | One named storage unit of the data model, declared by `add_tensor_buffer`: a tensor of type `E[S0, S1]`. | [ACT] Sec. 3.2; [TAIDL] Sec. 3.1 | data model (for one unit), scratchpad (except for the hardware it models), memory, bank | `taidl/taidl/accelerator.py:33` `add_data_model` -> `add_tensor_buffer`; `taidl/taidl/data_model.py:33` `DataModel` -> `TensorBuffer`; `act-backend/act_backend/instruction_info.py:71` prefix `Model` -> `Buffer` |
| global memory | The built-in tensor buffer `d0`: byte-addressable, one-dimensional, base element `u8`, reachable by the host. | [ACT] Sec. 3.2; [TAIDL] Sec. 3.1 | off-chip memory, DRAM, main memory, host memory; HBM except for a concrete chip's memory | `act-oracle/taidl_to/generic/decorator.py:142` parameter `hbm` -> `global_memory` (and `:64,70-72`); `act-backend/act_backend/instruction_info.py:71` `'HBM'`; `act-backend/act_backend/generic/src/ir/buffer.rs:13` `Buffer::HBM` -> `Buffer::GlobalMemory` |
| access dimensions | The leading dimensions `S0` of a tensor buffer: the ones a data slice gives a start and a size for. | [ACT] Sec. 3.2 | addressed dimensions, outer dims, rows | `taidl/taidl/accelerator.py:34` `dimensions` -> `access_dims`; `taidl/taidl/data_model.py:36,42,45` `access_dim` -> `access_dims` |
| base element | One addressable unit of a tensor buffer, always read whole: its base element shape with its element type, such as `256xi8`. | [TAIDL] Sec. 3.1; [ACT] Sec. 3.2 calls it the granularity of data access | unit, row, cell | none |
| base element shape | The trailing dimensions `S1` of a tensor buffer: the shape of one base element. | [TAIDL] Sec. 3.1; [ACT] Sec. 3.2 | unit shape, unit dim, inner dims, element shape, row shape | `taidl/taidl/accelerator.py:34` and `taidl/taidl/data_model.py:37,42,46` `unit_dim` -> `base_element_dims` |
| element type | One of the 19 type words of S7.3, such as `bf16`. | [SHLO] "element type"; [MLIR] | dtype (outside NumPy, JAX, or PyTorch names), data type, precision | `taidl/taidl/accelerator.py:34,42-44` and `taidl/taidl/data_model.py:38,42` `var_type` -> `element_type` |
| control register | A named integer of the data model that persists across instructions; read as `@s.<name>`. | [TAIDL] Sec. 3.1, Sec. 3.2 | state variable, register (unqualified), global, machine state variable, flag | `taidl/taidl/accelerator.py:46` `add_initial_state` -> `add_control_register`; `taidl/taidl/data_model.py:28` `State` -> `ControlRegister`; `taidl/taidl/instruction.py:39` `state` -> `control_registers`; `taidl/taidl/antlr4/IDLV2Expr.g4:36` token `STATE` -> `CONTROL_REGISTER` |
| initial value | The integer a control register holds before the first instruction. | current glossary | reset value, default | none |
| memory state | The contents of every tensor buffer and the value of every control register at one point of a run. | [ACT] Sec. 5.2 | machine state, memory contents | none |

## Instructions

| Term | Definition | Source | Banned synonyms | Identifiers to rename in v3 (file:line) |
| --- | --- | --- | --- | --- |
| instruction (abstract instruction where the contrast matters) | One operation of the accelerator's ISA, declared by `add_instruction` and parameterized by its instruction attributes. | [ACT] Sec. 3.2 | op, command, intrinsic | none |
| concrete instruction | An instruction with a value for every instruction attribute, such as `load_rm(n=4, addr_in=0, addr_out=2)`. | [ACT] Sec. 3.2 | call (for the instruction itself), instance, invocation | none |
| partially instantiated instruction (pii) | An instruction with its computational attributes set and its addressing attributes not yet set. | [ACT] Sec. 4.1 | partial instruction, plan node | none |
| instruction attribute | A computational attribute or an addressing attribute. | [ACT] Sec. 3.2 | calling attribute, parameter, field, operand, immediate | none |
| computational attribute | An instruction attribute that fixes what an instruction computes, such as a matrix size; read as `@c.<name>`. | [ACT] Sec. 3.2 (alpha) | computation attribute, comp attr, config attribute, immediate | `taidl/taidl/accelerator.py:62,71` `computation_attr` -> `computational_attrs`; `taidl/taidl/instruction.py:35,62,70,342` `comp_attr` -> `computational_attrs`; `act-oracle/taidl_to/instruction_codegen.py:149,158` |
| addressing attribute | An instruction attribute that says where a data slice starts; read as `@a.<name>`. | [ACT] Sec. 3.2 (beta) | address parameter, address attribute, pointer, parameter | `taidl/taidl/accelerator.py:63,72` `addressing_attr` -> `addressing_attrs`; `taidl/taidl/instruction.py:71,72,341` `parameters` -> `addressing_attrs`; `act-oracle/taidl_to/instruction_codegen.py:149,153` |
| validity constraint (constraint) | A condition that must hold for a concrete instruction to be valid, declared by `add_constraints`. | [ACT] Sec. 3.2; [TAIDL] Sec. 3.2 | guard, predicate, assertion, precondition | none |
| control register assignment | A text `@s.<name> = <expression>` an instruction runs to set a control register. | [TAIDL] Sec. 3.2 | state update, side effect, write-back | `taidl/taidl/accelerator.py:65` `update` -> `assignments`; `taidl/taidl/instruction.py:40,47` `update`, `update_asts`; `taidl/taidl/antlr4/IDLV2Expr.g4:8,11` rules `update_text`, `update` -> `assignment_text`, `assignment` |
| data slice | `[tensor buffer, starts, sizes]`: the sub-tensor of a tensor buffer that an instruction reads or writes. | [ACT] Sec. 3.2 | region, slice (alone), tile, window, access, operand, buffer | `taidl/taidl/instruction.py:18` `Region` -> `DataSlice`; `taidl/taidl/instruction.py:115,131` `check_regions`, `parse_regions`, `RegionTrees` |
| input data slice / output data slice | The k-th data slice given to `set_inputs` / `set_outputs`, counting from 0. | [ACT] Sec. 3.2 | input region / output region, source / destination, src / dst | none |
| start | The first index of a data slice along one access dimension. | [ACT] Sec. 3.2 | offset, base, address, origin | none |
| size | The count of indices of a data slice along one access dimension. | [SHLO] `dynamic_slice` `slice_sizes` | extent, length, `len`, count, span, end | none |
| data slice shape | The sizes of a data slice followed by its tensor buffer's base element shape. | derived from [ACT] Sec. 3.2 | region shape, tile shape, access shape | none |
| in-place mark | The `set_inplace` flag saying an instruction overwrites an input data slice. | current glossary | aliasing, donation | none |
| placement mark | The `set_apart` or `set_no_partial_overlap` flag; each appends validity constraints. | current glossary | overlap hint, alias constraint | none |
| instruction semantics (semantics) | The `ENTRY` block an instruction's `add_semantics` takes. | [TAIDL] Sec. 3.2; [ACT] Sec. 3.2 | body, computation, abstract computation, HLO, kernel | none |

## Semantics language

| Term | Definition | Source | Banned synonyms | Identifiers to rename in v3 (file:line) |
| --- | --- | --- | --- | --- |
| operation | One `%name = type[shape] operator(...)` of an instruction semantics. | [SHLO] ops; [MLIR] operation | line, statement, instruction (in an instruction semantics), node | `taidl/taidl/antlr4/IDLV2.g4:6,8` rule `instruction` -> `operation` |
| tensor operator | The XLA-HLO operator an operation applies, named by a word such as `dot`. | [ACT] Sec. 3.2, Table 1; [TAIDL] Sec. 2.4 | operation word, opcode, op, HLO op, primitive | `taidl/taidl/antlr4/IDLV2.g4:8,28,45` token `OPERATION` -> `OPERATOR` |
| reference operator | The XLA-HLO or StableHLO operator whose meaning a tensor operator takes, such as `broadcast_in_dim` for `broadcast`. | [SHLO]; [XLA] | reference operation, backing op, lowering | none |
| value | What an operation defines, named by `%` and a name. | [SHLO] value | SSA value, variable, tensor, register | none |
| input value / output value | `%In<k>` / `%Out<k>`: the value of input / output data slice k. | current glossary | region value, tile, parameter, argument, result | none |
| operand | A value an operation reads, between its parentheses. | [SHLO]; [MLIR] | argument, input (in an instruction semantics) | none |
| operation attribute | A `name=value` after an operation's operands, such as `dimensions={1,0}`. | [MLIR] operation attribute | attribute (unqualified, in an instruction semantics), option, flag | none |
| contracting dimensions / batch dimensions | The dimensions `dot` sums over / keeps paired, given by `lhs_contracting_dims` and `lhs_batch_dims`. | [SHLO] `dot_general`; [XLA] DotGeneral | reduction dims, inner dims | none; the operation attribute names follow XLA-HLO text |
| expression | An integer text over integers and references (S6.2). | current glossary | formula, index expression | none |
| condition | A true-or-false text of comparisons joined by `and` and `or` (S6.3). | current glossary | predicate, boolean expression | none |
| reference | `@a.<name>`, `@c.<name>`, or `@s.<name>` inside an expression or a condition. | current glossary | variable, placeholder, symbol | none |
| presumed reshape | Reading or writing a data slice's value in another shape of the same element count, as `reshape` would. | current glossary | implicit reshape, auto-reshape | none |
| XLA-HLO | The IR whose text the instruction semantics use and that tiled kernels arrive in. | [ACT] Sec. 3.1; [TAIDL] Sec. 2.4 | HLO (alone), XLA IR | none |

## Tools, compilation, and errors

| Term | Definition | Source | Banned synonyms | Identifiers to rename in v3 (file:line) |
| --- | --- | --- | --- | --- |
| front end | The `taidl` package: it parses and checks an ISA description. | current glossary | parser (for the whole), frontend, compiler | none |
| generator | A program that turns an accelerator into another tool: ACT is a compiler backend generator, TAIDL-TO a test oracle generator. | [ACT] Sec. 3.3 | backend (for the general case), emitter, tool | none |
| test oracle | The executable functional model of the ISA that TAIDL-TO generates; gloss "functional simulator" on first use. | [TAIDL] title, Sec. 2.3 | oracle (in prose), generated oracle, simulator (unqualified), emulator | none; `oracle` stays in repo, package, and directory names |
| kernel function (kernel) | A Python function decorated with `@kernel` that writes assembly code as calls of concrete instructions and runs on a test oracle. | [TAIDL] Sec. 6 | program, test program, assembly (for the function) | none |
| compiler backend | The compiler ACT generates: it takes a tiled kernel and emits assembly code. | [ACT] Sec. 3.3 | backend (unqualified), codegen | none |
| tiled kernel | An XLA-HLO sub-graph, typically 10 to 50 nodes, that a compiler backend takes. | [ACT] Sec. 1, Sec. 3.1 | tensor kernel, input kernel, kernel (unqualified), program | none |
| assembly code | The stream of concrete instructions and the constant tensor a compiler backend emits. | [ACT] Sec. 3.2 | machine code, compiled kernel, binary | none |
| instruction selection | The compiler backend phase that sets computational attributes and yields a pii graph, by equality saturation. | [ACT] Sec. 4.1 | lowering, isel (in prose) | none |
| memory allocation | The compiler backend phase that sets addressing attributes by integer constraint programming. | [ACT] Sec. 4, Sec. 6 | bufferization, malloc (in prose) | none |
| e-graph / e-class / e-node | The equality-saturation graph, its classes of equal terms, and its nodes. | [ACT] Sec. 5 | egraph, eclass, enode | none |
| rewrite rule (IR-to-IR, IR-to-ISA) | A rule that adds an equal term to the e-graph; IR-to-ISA rules come from the ISA description. | [ACT] Sec. 5.2 | rewrite (alone), pattern | none |
| identity instruction | A rewrite based on a no-op: an instruction that leaves the memory state unchanged. | [ACT] Sec. 5.2 | no-op instruction, nop | none |
| extraction / extractor | Choosing one pii graph from the e-graph, and the module that does it. | [ACT] Sec. 5 | selection (for this step), picker | none |
| sound / complete | A compiler backend whose assembly code always equals its tiled kernel / that compiles every tiled kernel some assembly code equals. | [ACT] Sec. 3.3 | correct, valid (for either) | none |
| unsupported feature | An ISA description feature a generator documents it does not handle, citing a section id. | current glossary | limitation, gap, TODO | none |
| coverage status | One of `supported`, `partial`, or `rejected`, per feature, in a generator's `docs/coverage.md`. | `act-backend/docs/coverage.md` | handled, works, missing | none |
| reject | To raise an error for an ISA description or a concrete instruction, naming the stage: generation, kernel compile time, or run time. | current glossary | refuse, deny, disallow, error out | `act-oracle/tests/test_refusals.py` -> `test_rejections.py`; `taidl/taidl/instruction.py:453`; `act-backend/act_backend/generator.py:94` |
| conforming | Said of a front end or generator that meets the spec (S1.1, S1.2). | `taidl/docs/overview.md` | compliant, compatible | none |
| undefined behavior | A case whose result the spec does not give; an ISA description that reaches it is in error. | [SHLO] | unspecified, UB | none |
| implementation-defined | A case each generator decides and documents in its `docs/coverage.md`. | [SHLO] | platform-specific, vendor-specific | none |
| ACT ecosystem | The four repos `taidl`, `act-oracle`, `act-backend`, and `act`, built around TAIDL. | [ACT] Sec. 1 | toolchain, stack, suite | none |
