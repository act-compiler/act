# Glossary

One canonical term per concept, following owner decisions 19-21 and then the reconciled glossary of the plans.
Docs, error messages, and unsupported-feature lists use the canonical term and none of its banned synonyms when they name a TAIDL concept.
The bans do not apply to a generator's own concepts, such as a rewrite condition or an XLA-HLO tensor, nor to internal identifiers.
This page is S10 of the TAIDL language specification.

| term | definition | banned synonyms |
| --- | --- | --- |
| ISA specification | The user's file: a Python file that builds one `Accelerator` in the surface of S2. | ISA description, spec (for the file), TAIDL specification, TAIDL definition, ISA definition, model file, config |
| TAIDL language specification | This language document, sections S1 to S12. | the spec, standard, reference manual |
| tensor accelerator (accelerator) | The `Accelerator` object an ISA specification builds: its data model and instructions. | target, device, chip |
| data model | All buffers and all state of an accelerator together. | data model (for one buffer) |
| buffer | One named storage unit of the data model, declared by `acc.buffer` (or the built-in `d0`). | tensor buffer, data model (for one buffer), scratchpad (except for the hardware it models), memory, bank |
| hbm | The built-in buffer `d0`, a `u8` buffer with one access dimension, sized by the `hbm` kernel argument. | global memory, off-chip memory, DRAM, main memory, host memory, HBM (as a name for `d0`) |
| access dimensions | The `access_dims` list of a buffer: the dimensions a region gives a start and a size for. | addressed dimensions, outer dims, rows |
| base element | One addressable unit of a buffer, always read whole. | unit, row, cell |
| base element shape | The `base_element_dims` list of a buffer: the shape of one base element. | unit shape, unit dim, inner dims, element shape, row shape |
| element type | One of the 21 type objects of S7.3, such as `bf16`; the keyword is `element_type`. | dtype (outside NumPy, JAX, or PyTorch names), data type, precision |
| instruction | One operation of the accelerator's ISA, declared by an instruction function. | op, command, intrinsic |
| issue | One run of an instruction with a value for each attribute (S5.6). | call (for an instruction), launch, dispatch |
| program | A sequence of issues run in order on one accelerator from given `hbm` contents (S5.6); a kernel function writes one, and assembly code is one; never the kernel function itself. | instruction stream, kernel (for a sequence of issues) |
| instruction attribute | A parameter of an instruction function, annotated `Comp` or `Addr`. | calling attribute, field, operand (for an attribute), immediate |
| computational attribute | An instruction attribute annotated `Comp`, which configures what the instruction computes. | computation attribute, comp attr, config attribute |
| addressing attribute | An instruction attribute annotated `Addr`, possibly negative, that only a start or a constraint reads. | address parameter, address attribute, pointer |
| state | All state variables of the accelerator together. | initial state, control register |
| state variable | A named signed 64-bit integer that persists across issues; declared by `acc.state`; v3 declares no other kind. | control register, initial state, register (unqualified), machine state variable, global, flag |
| initial value | The value a state variable holds before the first issue, from `acc.state`, or that every element of a buffer holds, from `initial_value` of `acc.buffer` (S4.3). | reset value, default |
| state update | One `update(var, expr)` an instruction records to change a state variable. | control register assignment, side effect, write-back |
| validity constraint (constraint) | A condition that must hold for an issue not to fault. | guard, predicate, assertion, precondition (for a validity constraint) |
| region | `buffer.region(starts, size=sizes)`: a box of a buffer an instruction reads or writes. | data slice, slice (alone), tile, window, access, operand (for a region), buffer (for a region) |
| region value | The value of a region in a semantics: what `read` returns, or what `write` takes. | tile, parameter, argument, result |
| input region / output region | The region of the k-th `read` / `write` of an instruction, counting from 0. | input data slice, output data slice, source, destination, src, dst |
| start | The first index of a region along one access dimension. | offset, base, address, origin |
| size | The count of indices of a region along one access dimension. | extent, length, `len`, count, span, end |
| region shape | The sizes of a region followed by its buffer's base element shape. | data slice shape, tile shape, access shape |
| instruction semantics (semantics) | The tensor values an instruction function computes from its region values with `ops` calls (S7). | computation, abstract computation, HLO, body (for the semantics alone) |
| instruction function | The function under `@acc.instruction` that the tracer runs once to record an instruction (S3.1); the decorator returns the instruction's handle (S5.1). | lambda, callback |
| value | A tensor value one `ops` call or tensor operator records in a semantics. | SSA value, variable, tensor, register |
| input value / output value | The value `read` returns / the value `write` takes for region k. | argument, result |
| operation | One `ops.<word>` call of an instruction function. | line, statement, instruction (in an instruction semantics), node |
| tensor operator | The XLA-HLO operator an operation applies, named by a word such as `dot`, called as `ops.dot`. | operation word, opcode, op, HLO op, primitive |
| reference operator | The XLA-HLO or StableHLO operator whose meaning a tensor operator takes. | reference operation, backing op, lowering |
| operand | A tensor value an `ops` call takes by position. | argument, input (in an instruction semantics) |
| operation attribute | A keyword argument of an `ops` call, such as `dimensions=[1, 0]`. | attribute (unqualified, in an instruction semantics), option, flag |
| expression | An integer value over plain integers and symbolic references, recorded as a tree (S6.2). | formula, index expression |
| condition | A comparison, or `any_of` or `all_of` over conditions (S6.3). | boolean expression |
| reference | An attribute parameter or a state variable object inside an instruction function. | placeholder, symbol |
| overlap relation (relation) | What one same-buffer input-output pair may share: tied, apart, exact-or-apart (the default), or may overlap (S5.5). | hazard model, aliasing mode, overlap hint |
| tied | The relation where the output reuses input k's storage; the `inplace=` marker of `write` declares it. | in-place (as a relation), aliasing, donation |
| apart | The relation where input k and the output share no element; the `apart=` marker declares it. | disjoint, non-aliasing |
| exact-or-apart | The default relation: equal starts and sizes in every dimension, or apart; the `no_partial_overlap=` marker declares it. | no partial overlap (as a relation name), partial alias |
| may overlap | The relation with no overlap restriction; the `may_overlap=` marker declares it. | unsafe overlap, hazard allowed |
| implicit constraint | A constraint every region carries on its evaluated address: each start at least 0 and the region inside its buffer's bounds (S5.2). | automatic check, hidden guard |
| front end | The `taidl` package: it traces an ISA specification and checks it. | parser (for the whole), frontend, compiler |
| generator | A Python package that turns a checked model into another tool, such as the test oracle generator or the compiler backend generator. | backend (for the general case), emitter, tool |
| test oracle generator | The generator of test oracles: repo `act-oracle`. | TAIDL-TO (as a prose name), Test Oracle Generator (title case), oracle generator |
| test oracle | The executable functional model of the ISA that the test oracle generator emits. | oracle (in prose), generated oracle, simulator (unqualified), emulator |
| compiler backend generator | The generator of compiler backends: repo `act-backend`. | ACT-Backend Generator, Compiler Backend Generator (title case) |
| compiler backend | The compiler a compiler backend generator emits: it takes a tiled kernel and emits assembly code. | backend (unqualified), codegen |
| kernel function (kernel) | A Python function decorated with `@kernel` that writes assembly code as calls of concrete instructions and runs on a test oracle; running it issues one program. | program (for the function), test program, assembly (for the function), kernel (bare, in prose) |
| tiled kernel | An XLA-HLO sub-graph that a compiler backend takes. | tensor kernel, input kernel, kernel (unqualified, for the compiler input) |
| assembly code | The stream of concrete instructions and the constant tensor a compiler backend emits. | machine code, compiled kernel, binary |
| unsupported feature | An ISA specification feature a generator documents it does not handle, citing a section id. | limitation, gap, TODO |
| rejection (reject) | To raise an error for an ISA specification at check, generation, or kernel compile time, as S9 lists. | refuse, refuses, refused, refusal, deny, disallow, error out, fail (for a rejection) |
| fault | The defined run-time error of S1.3: the issue writes no region and no state, and is reported with its section id. | undefined behavior, UB, implementation-defined, unspecified |
| width layer | The optional layer that gives each attribute a type `s<w>` or `u<w>`; `s64` without it (S11.2). | type system, precision, bits, add-on |
| encoding layer | The optional layer that gives an instruction an encoding block of pure bit placement (S11.3). | binary format, bit layout, bits and stuff, add-on |
| encoded attribute | An attribute a slice of the encoding block names (S11.4). | binary field, encoded field, real attribute |
| derived attribute | An attribute of an instruction with an encoding block that no slice names, tied by one affine link to an encoded attribute (S11.4). | virtual attribute, hidden field, add-on attribute |
| generator metadata | The namespace-keyed plain data on the accelerator, a buffer, or an instruction that only the named generator reads (S12). | add-on, addon, generator add-on, plugin, extension, pragma, hint, annotation (for this data) |
| metadata entry | One generator's block of generator metadata on one node (S12.1). | add-on entry, generator add-on, plugin block, extension block |
| marker | A keyword of `write` such as `inplace=` that declares one pair's relation and expands to the constraints it stands for (S5.5); `inplace` is the in-place marker, and `apart`, `no_partial_overlap`, and `may_overlap` are the placement markers. | overlap hint, alias constraint, flag, in-place mark, placement mark |
| check function (check) | `taidl.check`: runs every front-end rule on an accelerator and returns a `Model` or raises one `Rejection` (S9). | validate, validator, `Accelerator.validate()`, lint, pre-check |
| checked model (model) | The frozen typed `Model` that `check` returns, the only input a generator body reads (S4.6). | validated accelerator, typed model (after check), AST, tree, IR |
| excluded instruction (exclusion) | An instruction a generator leaves out because its model cannot hold it, with a printed and recorded reason (S9). | skipped instruction, ignored instruction, silently dropped instruction |
| shorthand | A spelling that expands to constraints a user could write by hand, as a marker or a type does. | user-productivity helper, convenience, sugar, syntactic sugar |
