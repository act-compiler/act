# Glossary

One canonical term per concept, following owner decisions 19-21 and then the reconciled glossary of the plans.
Docs, error messages, and unsupported-feature lists use the canonical term and none of its banned synonyms when they name a TAIDL concept.
The bans do not apply to a generator's own concepts, such as a rewrite condition or an XLA-HLO tensor, nor to internal identifiers.
This page is S10 of the spec.

| term | definition | banned synonyms |
| --- | --- | --- |
| ISA specification | The user's file: a Python program that builds one `Accelerator`. | ISA description, TAIDL definition, ISA definition, model file, config |
| TAIDL language specification | This language document, sections S1 to S12; "the spec" for short. | standard, reference manual |
| accelerator | The `Accelerator` object an ISA specification builds: its data model and instructions. | target, device, chip |
| data model | All buffers and all state of an accelerator together. | data model (for one buffer) |
| buffer | One named storage unit of the data model, declared by `add_buffer` (or the built-in `d0`). | tensor buffer, data model (for one unit), memory, bank |
| hbm | The built-in buffer `d0`, a `u8` buffer with one access dimension, sized by the `hbm` kernel argument. | global memory, off-chip memory, DRAM, main memory, host memory, HBM (for `d0`) |
| access dimensions | The `access_dims` list of a buffer: the dimensions a region gives a start and a size for. | addressed dimensions, outer dims, rows |
| unit shape | The `base_element_dims` list of a buffer: the shape of one addressed unit, always read whole. | unit dim, inner dims, element shape, row shape |
| element type | One of the 19 type words of S7.3, such as `bf16`. | dtype (outside NumPy or PyTorch names), data type, precision |
| instruction | One operation of the accelerator's ISA, declared by `add_instruction`. | op, command, intrinsic |
| issue | One run of an instruction with a value for each attribute (S5.6). | call (for an instruction), launch, dispatch |
| instruction attribute | A computational attribute or an addressing attribute. | parameter, field, operand, immediate |
| computational attribute | An instruction attribute that configures what it computes; read as `@c.<name>`. | computation attribute, comp attr, config attribute |
| addressing attribute | An instruction attribute, possibly negative, that a start or a constraint reads; read as `@a.<name>`. | address parameter, address attribute, pointer |
| state | All state variables of the accelerator together. | initial state, control registers |
| state variable | A named value of a declared kind, an integer by default, that persists across issues; read as `@s.<name>`. | register, control register, global, machine state variable, flag |
| initial value | The integer `add_state` gives a state variable before the first issue. | reset value, default |
| state update | A text `@s.<name> = <value>` an instruction runs to change a state variable. | assignment, side effect, write-back |
| constraint | A condition that must hold for an issue not to fault. | guard, predicate, assertion, precondition |
| region | `[buffer, starts, sizes]`: a box of a buffer an instruction reads or writes. | data slice, tile, slice, access, window, operand |
| region value | The value of a region in a semantics: `%In<k>` or `%Out<k>`. | tile, tensor |
| input region / output region | The k-th region given to `set_inputs` / `set_outputs`, counting from 0. | source / destination, src / dst |
| start | The first index of a region along one access dimension. | offset, base, address, origin |
| size | The count of indices of a region along one access dimension. | extent, length, count, span |
| region shape | The sizes of a region followed by its buffer's unit shape. | tile shape, access shape |
| semantics | The `ENTRY` block an instruction's `add_semantics` takes. | body, computation, HLO, kernel |
| line | One `%name = type[shape] operation(...)` of a semantics. | statement, instruction (in a semantics), op |
| value | What a line defines, named by `%` and a name. | SSA value, variable, tensor, register |
| input value / output value | `%In<k>` / `%Out<k>`: the value of input / output region k. | parameter, argument, result |
| operation word | The word naming a line's operation, such as `dot`. | opcode, op, HLO op, primitive |
| reference operation | The XLA-HLO or StableHLO operation whose meaning an operation word takes. | backing op, lowering |
| operand | A value a line reads, between its parentheses. | argument, input (in a semantics) |
| operation attribute | A `name=value` after a line's operands, such as `dimensions={1,0}`. | attribute (unqualified, in a semantics), option, flag |
| expression | An integer value text over integers and references (S6.2). | formula, index expression |
| condition | A true-or-false text of comparisons joined by `and` and `or` (S6.3). | predicate, boolean expression |
| reference | `@a.<name>`, `@c.<name>`, or `@s.<name>` inside an expression. | variable, placeholder, symbol |
| in-place mark | The output reuses the marked input's storage; `set_inplace`, shorthand for the tied relation (S5.5). | donation |
| placement mark | `set_apart`, `set_no_partial_overlap`, or `set_may_overlap`, shorthand for the apart, exact-or-apart, or may-overlap relation (S5.5). | overlap hint, alias constraint |
| relation | What one same-buffer input-output pair may share: tied, apart, exact-or-apart (the default), or may overlap (S5.5). | hazard mode |
| implicit constraint | A constraint every region carries on its evaluated address: each start at least 0 and the region inside its buffer's bounds (S5.2). | bounds check, address check |
| front end | The `taidl` package: it parses and checks an ISA specification. | parser (for the whole), frontend, compiler |
| object model | The Python objects and attributes a generator reads from an accelerator (S4.6). | IR (for the whole), AST (for the whole) |
| generator | A program that turns an accelerator into another artifact, such as TAIDL-TO or ACT-Backend. | backend (for the general case), emitter, tool |
| test oracle generator | TAIDL-TO, in the repo `act-oracle`: the generator of test oracles. | oracle generator, simulator generator |
| test oracle | TAIDL-TO's output: an executable functional model of the ISA. | simulator (unqualified), emulator |
| compiler backend generator | ACT-Backend: the generator of compiler backends. | backend generator, compiler generator |
| compiler backend | ACT-Backend's output: a compiler from XLA-HLO kernels to instructions. | backend (unqualified), codegen |
| kernel | A program a compiler backend takes (input kernel, in XLA-HLO) or emits (compiled kernel, of instructions); always qualified. | machine code, assembly, program (for either) |
| unsupported feature | An ISA specification feature a generator documents it does not handle, citing a section id. | limitation, gap, TODO |
| reject | To raise an error for an ISA specification at generation or kernel compile time, as S9 lists. | refuse, deny, disallow, error out, honor (for its opposite) |
| fault | The defined run-time error of S1.3: the issue writes no region and no state, and is reported with its section id. | undefined behavior, UB, implementation-defined, trap, crash |
| width layer | The optional layer that gives each attribute a type `s<w>` or `u<w>`; `s64` without it (S11.2). | add-on, width annotation |
| encoding layer | The optional layer that gives an instruction an encoding block of pure bit placement (S11.3). | add-on, encoding annotation |
| encoded attribute | An attribute a slice of the encoding block names (S11.4). | field attribute |
| derived attribute | An attribute of an instruction with an encoding block that no slice names, tied by one affine link to an encoded attribute (S11.4). | hidden attribute, virtual attribute |
| generator metadata | The namespace-keyed plain data on the accelerator, a buffer, or an instruction that only the named generator reads (S12). | add-on, hint, annotation (for metadata) |
| metadata entry | One generator's block of generator metadata on one node (S12.1). | metadata block |
| marker | A mark such as `set_inplace` that expands to the constraints it stands for (S5.5). | flag (for a mark) |
| shorthand | A spelling that expands to constraints a user could write by hand, as a marker or a type does. | sugar |
