# Glossary

One canonical term per concept, for the TAIDL language specification and the four repos `taidl`, `act-oracle`, `act-backend`, and `act`.
Docs, error messages, and user-visible names use the canonical term and none of its banned synonyms when they name a TAIDL concept.
The bans do not apply to a generator's own concepts, such as a rewrite condition, a precondition of a rewrite, an XLA-HLO tensor, or an off-chip placement offset.
A short form in parentheses may replace the term once the full term has appeared in the same section.
Where the owner's papers are precise, their term wins; otherwise the StableHLO or MLIR term wins; the owner decisions cited in the source column override both.
The last column lists public identifiers, file names, and user-visible strings that change in the v3 release, by symbol; private internals follow only when touched.
[words-to-avoid.md](words-to-avoid.md) lists every banned synonym below, and a CI grep checks docs against it.

## Description and memory

| Term | Definition | Source | Banned synonyms | Identifiers to rename in v3 |
| --- | --- | --- | --- | --- |
| ISA specification | A Python program that builds one `Accelerator`: the file a user writes. | Decision 32; [ACT] title and Sec. 3.2 write "ISA description" | ISA description, spec (for the file), TAIDL specification, TAIDL definition, ISA definition, model file, config | none; `act-backend/act_backend/spec_checks.py` keeps its name |
| TAIDL language specification (language specification) | The TAIDL language document, sections S1 to S10. | Decision 32 | the spec, standard, reference manual | none |
| tensor accelerator (accelerator) | The `Accelerator` object an ISA specification builds: its data model and its instructions; it covers tensor accelerators and GPU tensor-core instructions. | [ACT] title; [TAIDL] title; Decision 15 | target, device, chip | generated directory `targets/` -> `accelerators/` in `act-oracle/taidl_to/generator.py`, `act-backend/act_backend/generator.py`, and `act/tests/test_end_to_end.py` |
| data model | All storage of an accelerator together: its buffers and its state. | Decision 19; [ACT] Sec. 3.2; [TAIDL] Sec. 3.1 | none | none; `Accelerator.data_model` in `taidl/taidl/accelerator.py` names the set |
| buffer | One named storage unit of the data model, declared by `add_buffer`: a tensor of type `E[S0, S1]`. | Decision 19; [ACT] and [TAIDL] write "tensor buffer" | tensor buffer, data model (for one buffer), scratchpad (except for the hardware it models), memory, bank | `add_data_model` -> `add_buffer` in `taidl/taidl/accelerator.py`; `DataModel` -> `Buffer` in `taidl/taidl/data_model.py`; prefix `Model` -> `Buffer` in `act-backend/act_backend/instruction_info.py` |
| hbm | The built-in buffer `d0`: byte-addressable, one-dimensional, base element `u8`, reachable by the host, sized by the `hbm` kernel argument. | Decision 19; Q7.7; the papers write "global memory (HBM)", a pointer and not a synonym | global memory, off-chip memory, DRAM, main memory, host memory, HBM (as a name for `d0`) | none; `hbm` stays in `act-oracle/taidl_to/generic/decorator.py` and `act-backend/act_backend/instruction_info.py` |
| access dimensions | The leading dimensions `S0` of a buffer: the ones a region gives a start and a size for. | [ACT] Sec. 3.2 | addressed dimensions, outer dims, rows | `dimensions` -> `access_dims` in `taidl/taidl/accelerator.py`; `access_dim` -> `access_dims` in `taidl/taidl/data_model.py` |
| base element | One addressable unit of a buffer, always read whole: its base element shape with its element type, such as `256xi8`. | [TAIDL] Sec. 3.1; [ACT] Sec. 3.2 calls it the granularity of data access | unit, row, cell | none |
| base element shape | The trailing dimensions `S1` of a buffer: the shape of one base element. | [TAIDL] Sec. 3.1; [ACT] Sec. 3.2 | unit shape, unit dim, inner dims, element shape, row shape | `unit_dim` -> `base_element_dims` in `taidl/taidl/accelerator.py` and `taidl/taidl/data_model.py` |
| element type | One of the 19 type words of S7.3, such as `bf16`. | [SHLO] "element type"; [MLIR] | dtype (outside NumPy, JAX, or PyTorch names), data type, precision | `var_type` -> `element_type` in `taidl/taidl/accelerator.py` and `taidl/taidl/data_model.py` |
| state variable (state) | A named value of a declared kind, an integer by default, that persists across instructions; declared by `add_state`, read as `@s.<name>`; state is all of them. | Decision 19; Decision 21; Q7.6; [TAIDL] writes "control register" | control register, initial state, register (unqualified), machine state variable, global, flag | `add_initial_state` -> `add_state` in `taidl/taidl/accelerator.py`; `State` and the `@s.` prefix stay |
| initial value | The integer a state variable holds before the first instruction. | current glossary | reset value, default | none |
| memory state | The contents of every buffer and the value of every state variable at one point of a run. | [ACT] Sec. 5.2 | machine state, memory contents | none |

## Instructions

| Term | Definition | Source | Banned synonyms | Identifiers to rename in v3 |
| --- | --- | --- | --- | --- |
| instruction (abstract instruction where the contrast matters) | One operation of the accelerator's ISA, declared by `add_instruction` and parameterized by its instruction attributes. | [ACT] Sec. 3.2 | op, command, intrinsic | none |
| concrete instruction | An instruction with a value for every instruction attribute, such as `load_rm(n=4, addr_in=0, addr_out=2)`. | [ACT] Sec. 3.2 | call (for the instruction itself), instance, invocation | none |
| partially instantiated instruction (pii) | An instruction with its computational attributes set and its addressing attributes not yet set. | [ACT] Sec. 4.1 | partial instruction, plan node | none |
| instruction attribute | A computational attribute or an addressing attribute. | [ACT] Sec. 3.2 | calling attribute, parameter, field, operand (for an attribute), immediate | none |
| computational attribute | An instruction attribute that fixes what an instruction computes, such as a matrix size; read as `@c.<name>`. | [ACT] Sec. 3.2 (alpha) | computation attribute, comp attr, config attribute | `computation_attr` -> `computational_attrs` in `taidl/taidl/accelerator.py`; `comp_attr` -> `computational_attrs` in `taidl/taidl/instruction.py`; `act-oracle/taidl_to/instruction_codegen.py` follows |
| addressing attribute | An instruction attribute that a region start uses; read as `@a.<name>`; its value may be negative, and its name is a plain variable name that no tool reads meaning from. | [ACT] Sec. 3.2 (beta); Decision 34 | address parameter, address attribute, pointer | `addressing_attr` -> `addressing_attrs` in `taidl/taidl/accelerator.py`; `parameters` -> `addressing_attrs` in `taidl/taidl/instruction.py`; `act-oracle/taidl_to/instruction_codegen.py` follows |
| derived attribute | An attribute that never appears in the binary and is tied to one encoded attribute by one affine constraint, such as `B` with address `B*8` linked to encoded `A`. | Decision 35; Q10.3 | virtual attribute, hidden field, add-on attribute | none |
| width layer | The optional layer giving each attribute and operand a bit width and signedness (`s<w>`, `u<w>`); absent, they are `s64`; every generator gives it one meaning. | Decision 38; Q10.1; LLVM TableGen `bits<N>` fields | type system, precision, bits, add-on | none |
| encoding layer | The optional layer of binary-encoding declarations an assembler reads, added on top of the width layer; the front end validates it. | Decision 35; Decision 38; LLVM TableGen `Inst` encoding fields | binary format, bit layout, bits and stuff, add-on | none |
| encoding block | The optional per-instruction declaration of the encoding layer: the fixed word width, the fixed bits, and per encoded attribute an ordered list of slices; it is pure bit placement and only the assembler and disassembler generator reads it. | Decision 35; Decision 38; Q10.2; LLVM TableGen `Inst` field assignments | encoding annotation, bit layout, format block, add-on | none |
| encoded attribute | An attribute a slice of the encoding block names, so that it appears in the binary; it has a declared type whose width equals its slice total. | Decision 35; Q10.2; Q10.3 | binary field, encoded field, real attribute | none |
| validity constraint (constraint) | A condition that must hold for a concrete instruction to be valid, declared by `add_constraints`. | [ACT] Sec. 3.2; [TAIDL] Sec. 3.2 | guard, predicate, assertion, precondition (for a validity constraint) | none |
| state update | A text `@s.<name> = <expression>` an instruction runs to set a state variable. | [TAIDL] Sec. 3.2 writes "assignment" | control register assignment, side effect, write-back | none; `update`, `update_asts`, and `update_text` stay |
| region | `[buffer, starts, sizes]`: the sub-tensor of a buffer that an instruction reads or writes. | Decision 20; [ACT] Sec. 3.2 writes "data slice" | data slice, slice (alone), tile, window, access, operand (for a region), buffer (for a region) | none; `Region` in `taidl/taidl/instruction.py` stays |
| input region / output region | The k-th region given to `set_inputs` / `set_outputs`, counting from 0. | Decision 20 | input data slice, output data slice, source, destination, src, dst | none |
| start | The first index of a region along one access dimension: any expression over attributes, such as `1024 - @a.addr_out`; bounds apply to its evaluated value only. | [ACT] Sec. 3.2; Decision 34 | offset, base, address, origin | none |
| size | The count of indices of a region along one access dimension. | [SHLO] `dynamic_slice` `slice_sizes` | extent, length, `len`, count, span, end | none |
| region shape | The sizes of a region followed by its buffer's base element shape. | derived from [ACT] Sec. 3.2 | data slice shape, tile shape, access shape | none |
| overlap relation (relation) | The one rule per same-buffer input-output pair: tied, apart, exact-or-apart, or may overlap; the default is exact-or-apart; pairs in different buffers need none. | Q9.1; Q9.2; Decision 31 | hazard model, aliasing mode, overlap hint | none |
| tied | The relation where the output reuses input k's storage; the expansion of `set_inplace(k)` plus a tie the backend reads. | Decision 17; Q9.1 | in-place (as a relation), aliasing, donation | none |
| apart | The relation where input k and the output share no element: some dimension has one region ending at or before the other starts. | Q9.1 | disjoint, non-aliasing | none |
| exact-or-apart | The default relation: equal starts and equal sizes in every dimension, or apart; any other overlap is a fault. | Q9.1; Q9.3; Decision 22 | no partial overlap (as a relation name), partial alias | none |
| may overlap | The relation with no overlap restriction; a permitted overlap reads the pre-issue memory state. | Q9.1; Decision 31 | unsafe overlap, hazard allowed | `set_may_overlap` is new; the name is provisional |
| marker | A shorthand that declares one pair's relation and expands to the constraints it stands for: `set_inplace` (tied), `set_apart` (apart), `set_no_partial_overlap` (exact-or-apart), `set_may_overlap` (may overlap, no constraint). | Decision 30; Q9.2; Rust `std::marker` and Java marker interfaces name a declared property | overlap hint, alias constraint, flag, in-place mark, placement mark | none; `set_inplace` and `set_apart` stay |
| shorthand | A declaration defined only by the longer declarations it expands to, with no meaning of its own. | Decision 30; CSS Cascade 4 "shorthand property"; Standard ML "derived forms" | user-productivity helper, convenience, sugar, syntactic sugar | none |
| implicit constraint | A constraint every region or pair gets without a declaration: it fits within its buffer's bounds, no address is negative, output regions are apart, and an undeclared same-buffer input-output pair is exact-or-apart; it applies to evaluated addresses. | Decision 30; Decision 34; Q9.2 | automatic check, hidden guard | none |
| instruction semantics (semantics) | The `ENTRY` block an instruction's `add_semantics` takes. | [TAIDL] Sec. 3.2; [ACT] Sec. 3.2 | body, computation, abstract computation, HLO | none |

## Semantics language

| Term | Definition | Source | Banned synonyms | Identifiers to rename in v3 |
| --- | --- | --- | --- | --- |
| operation | One `%name = type[shape] operator(...)` of an instruction semantics. | [SHLO] ops; [MLIR] operation | line, statement, instruction (in an instruction semantics), node | rule `instruction` -> `operation` in `taidl/taidl/antlr4/IDLV2.g4` and the generated parsers |
| tensor operator | The XLA-HLO operator an operation applies, named by a word such as `dot`. | [ACT] Sec. 3.2, Table 1; [TAIDL] Sec. 2.4 | operation word, opcode, op, HLO op, primitive | token `OPERATION` -> `OPERATOR` in `taidl/taidl/antlr4/IDLV2.g4` |
| reference operator | The XLA-HLO or StableHLO operator whose meaning a tensor operator takes, such as `broadcast_in_dim` for `broadcast`. | [SHLO]; [XLA] | reference operation, backing op, lowering | none |
| value | What an operation defines, named by `%` and a name. | [SHLO] value | SSA value, variable, tensor, register | none |
| region value (input value / output value) | `%In<k>` / `%Out<k>`: the value of input / output region k. | Decision 20 | tile, parameter, argument, result | none |
| operand | A value an operation reads, between its parentheses. | [SHLO]; [MLIR] | argument, input (in an instruction semantics) | none |
| operation attribute | A `name=value` after an operation's operands, such as `dimensions={1,0}`. | [MLIR] operation attribute | attribute (unqualified, in an instruction semantics), option, flag | none |
| contracting dimensions / batch dimensions | The dimensions `dot` sums over / keeps paired, given by `lhs_contracting_dims` and `lhs_batch_dims`. | [SHLO] `dot_general`; [XLA] DotGeneral | reduction dims | none; the operation attribute names follow XLA-HLO text |
| expression | An integer text over integers and references (S6.2). | current glossary | formula, index expression | none |
| condition | A true-or-false text of comparisons joined by `and` and `or` (S6.3). | current glossary | boolean expression | none |
| reference | `@a.<name>`, `@c.<name>`, or `@s.<name>` inside an expression or a condition. | Decision 21; current glossary | placeholder, symbol | none; the `@s.` prefix stays |
| presumed reshape | Reading or writing a region's value in another shape of the same element count, as `reshape` would. | current glossary | implicit reshape, auto-reshape | none |
| XLA-HLO | The IR whose text the instruction semantics use and that tiled kernels arrive in. | [ACT] Sec. 3.1; [TAIDL] Sec. 2.4 | HLO (alone), XLA IR | none |

## Tools, compilation, and errors

| Term | Definition | Source | Banned synonyms | Identifiers to rename in v3 |
| --- | --- | --- | --- | --- |
| front end | The `taidl` package: it parses and checks an ISA specification. | current glossary | parser (for the whole), frontend, compiler | none |
| generator | A program that turns an accelerator into another tool. | [ACT] Sec. 3.3 | backend (for the general case), emitter, tool | none |
| generator metadata | Optional data in an ISA specification, at accelerator, buffer, or instruction level, kept under one generator's namespace; only that generator reads it, it never changes the meaning another generator gives, and it reaches that generator's entry point, such as `generate_allo()` of act-allo; one namespace's block is a metadata entry. | Decision 39; [MLIR] discardable (dialect-prefixed) attributes; LLVM LangRef metadata; ONNX `metadata_props` on model, graph, and node | add-on, addon, generator add-on, plugin, extension, pragma, hint, annotation (for this data) | none |
| metadata entry | One generator's block of generator metadata: the plain data under that generator's namespace at one accelerator, buffer, or instruction; only that generator validates it. | Decision 39; Q10.5 | add-on entry, generator add-on, plugin block, extension block | none |
| test oracle generator | The generator of test oracles: repo `act-oracle`, distribution `taidl-to`, module `taidl_to`, function `generate_oracle`. | [TAIDL] Sec. 2.3; Q7.11 | TAIDL-TO (as a prose name), Test Oracle Generator (title case), oracle generator | none; the repo, distribution, module, and function names stay |
| compiler backend generator | The generator of compiler backends: repo `act-backend`, module `act_backend`; ACT is one. | [ACT] Sec. 3.3; Q7.11 | ACT-Backend Generator, Compiler Backend Generator (title case) | none |
| test oracle | The executable functional model of the ISA that the test oracle generator emits; gloss "functional simulator" on first use. | [TAIDL] title, Sec. 2.3 | oracle (in prose), generated oracle, simulator (unqualified), emulator | none; `oracle` stays in repo, package, and directory names |
| kernel function (kernel) | A Python function decorated with `@kernel` that writes assembly code as calls of concrete instructions and runs on a test oracle; bare "kernel" names only the decorator and its call. | [TAIDL] Sec. 6; Q7.11 | program, test program, assembly (for the function), kernel (bare, in prose) | none |
| compiler backend | The compiler a compiler backend generator emits: it takes a tiled kernel and emits assembly code. | [ACT] Sec. 3.3 | backend (unqualified), codegen | none |
| tiled kernel | An XLA-HLO sub-graph, typically 10 to 50 nodes, that a compiler backend takes. | [ACT] Sec. 1, Sec. 3.1 | tensor kernel, input kernel, kernel (unqualified, for the compiler input) | none |
| assembly code | The stream of concrete instructions and the constant tensor a compiler backend emits. | [ACT] Sec. 3.2 | machine code, compiled kernel, binary | none |
| instruction selection | The compiler backend phase that sets computational attributes and yields a pii graph, by equality saturation. | [ACT] Sec. 4.1 | lowering, isel | none |
| memory allocation | The compiler backend phase that sets addressing attributes by integer constraint programming. | [ACT] Sec. 4, Sec. 6 | bufferization, malloc (in prose) | none |
| e-graph / e-class / e-node | The equality-saturation graph, its classes of equal terms, and its nodes. | [ACT] Sec. 5 | egraph, eclass, enode | none |
| rewrite rule (IR-to-IR, IR-to-ISA) | A rule that adds an equal term to the e-graph; IR-to-ISA rules come from the ISA specification. | [ACT] Sec. 5.2 | rewrite (alone), pattern | none |
| identity instruction | A rewrite based on a no-op: an instruction that leaves the memory state unchanged. | [ACT] Sec. 5.2 | no-op instruction, nop | none |
| extraction / extractor | Choosing one pii graph from the e-graph, and the module that does it. | [ACT] Sec. 5 | selection (for this step), picker | none |
| sound / complete | A compiler backend whose assembly code always equals its tiled kernel / that compiles every tiled kernel some assembly code equals. | [ACT] Sec. 3.3 | correct, valid (for either) | none |
| unsupported feature | An ISA specification feature a generator documents it does not handle, citing a section id; the generator rejects an input that uses it with a coded error and never assumes it away. | current glossary; Decision 27 | limitation, gap, TODO | none |
| coverage status | One of `supported`, `partial`, or `unsupported`, per feature, in a generator's `docs/coverage.md`. | `act-backend/docs/coverage.md`; Q7.8 | handled, works, missing, rejected (as a status) | none |
| rejection (reject) | An error the front end or a generator raises for a problem the architect can correct, such as a typo or an invalid ISA specification, at generation or kernel compile time, before any run; the message is clear and names the stage. | Q7.8; Decision 40 | refuse, refuses, refused, refusal, deny, disallow, error out, fail (for a rejection) | every "refus" word in messages, docstrings, READMEs, and public names, as its own v3 item; `test_refusals.py` in `act-oracle/tests` and `act-backend/tests`; alias `Fault` -> `Rejection` in `taidl/taidl/operations.py` |
| fault | A defined run-time error: an issue where any violated constraint, implicit, marker, or written, fails, before any write; it writes no region and no state, and it is reported with the clause id; no case is left undefined or to each implementation; a test oracle stops with a fault only on invalid assembly code or binary input and otherwise runs correctly. | Q7.8; Q1.12; Decision 16; Decision 40 | undefined behavior, UB, implementation-defined, unspecified | none |
| dropped candidate | A candidate a compiler backend discards because it fails, with a recorded reason; the search goes on and never stops for one failing candidate. | Decision 40; [ACT] Sec. 4 | failed candidate, skipped candidate, pruned candidate (for a failure), silent fallback | none |
| invalid input kernel | A StableHLO, XLA HLO, torch-mlir, or torch.fx kernel that is invalid or outside the input specification; the only input on which a compiler backend stops. | Decision 40; [SHLO]; [XLA]; [MLIR] | bad kernel, broken kernel, unsupported kernel (for an invalid one), malformed tiled kernel | none |
| conforming | Said of a front end or generator that meets the TAIDL language specification (S1.1, S1.2). | `taidl/docs/overview.md` | compliant, compatible | none |
| ACT ecosystem | The four repos `taidl`, `act-oracle`, `act-backend`, and `act`, built around TAIDL; the `act` distribution is `act-ecosystem`. | [ACT] Sec. 1; Q7.11 | toolchain, stack, suite | none |

## Plain words

| Term | Definition | Source | Banned synonyms | Identifiers to rename in v3 |
| --- | --- | --- | --- | --- |
| uses | The plain verb for one tool applying another. | house style | leverages | none |
| the concrete claim | A named capability, such as "plugs into XLA". | house style | seamless support | none |
| from an ISA specification alone | The plain form for what a generator needs. | house style | just (as in "just from TAIDL specifications") | none |
| applies | The plain verb for a rule taking effect; or name the concrete check. | house style | honored, respected | none |
