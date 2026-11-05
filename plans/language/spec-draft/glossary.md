# Glossary

One canonical term per concept.
Docs, error messages, and unsupported-feature lists use the canonical term and none of its banned synonyms when they name a TAIDL concept.
The bans do not apply to a generator's own concepts, such as a rewrite condition or an XLA-HLO tensor, nor to internal identifiers.
This page is S10 of the spec.

| term | definition | banned synonyms |
| --- | --- | --- |
| ISA description | A Python program that builds one `Accelerator`. | spec (for this file), ISA definition, model file, config |
| the spec | This language document, sections S1 to S10. | standard, reference manual |
| accelerator | The `Accelerator` object an ISA description builds: its data models, state variables, and instructions. | target, device, chip |
| data model | A named memory of the accelerator, declared by `add_data_model` (or the built-in `d0`). | buffer, scratchpad, memory, tensor, bank |
| off-chip memory | The built-in data model `d0`: a flat sequence of `u8` bytes. | DRAM, HBM, main memory, host memory |
| addressed dimensions | The `dimensions` list of a data model: the dimensions a region gives a start and a size for. | access dims, outer dims, rows, `access_dim` |
| unit shape | The `unit_dim` list of a data model: the shape of one addressed unit, always read whole. | unit dim, inner dims, element shape, row shape |
| element type | One of the 19 type words of S7.3, such as `bf16`. | dtype (outside NumPy or PyTorch names), data type, `var_type` (except as the parameter name), precision |
| instruction | One operation of the accelerator's ISA, declared by `add_instruction`. | op, command, intrinsic |
| instruction attribute | A computational attribute or an addressing attribute. | parameter, field, operand, immediate |
| computational attribute | An instruction attribute that configures what it computes; read as `@c.<name>`. | computation attribute, comp attr, config attribute |
| addressing attribute | An instruction attribute that says where a region starts; read as `@a.<name>`. | address parameter, address attribute, pointer |
| state variable | A named integer of the accelerator that persists across instructions; read as `@s.<name>`. | register, global, machine state variable, flag |
| initial value | The integer a state variable holds before the first instruction. | reset value, default |
| state update | A text `@s.<name> = <value>` an instruction runs to change a state variable. | assignment, side effect, write-back |
| constraint | A condition that must hold for an instruction to be valid. | guard, predicate, assertion, precondition |
| region | `[data model, starts, sizes]`: a box of a data model an instruction reads or writes. | tile, slice, access, window, operand, buffer |
| region value | The value of a region in a semantics: `%In<k>` or `%Out<k>`. | tile, tensor |
| input region / output region | The k-th region given to `set_inputs` / `set_outputs`, counting from 0. | source / destination, src / dst |
| start | The first index of a region along one addressed dimension. | offset, base, address, origin |
| size | The count of indices of a region along one addressed dimension. | extent, length, count, span |
| region shape | The sizes of a region followed by its data model's unit shape. | tile shape, access shape |
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
| in-place mark | The `set_inplace` flag saying an instruction overwrites an input region. | aliasing, donation |
| placement mark | The `set_apart` or `set_no_partial_overlap` flags, each of which appends constraints. | overlap hint, alias constraint |
| presumed reshape | Reading or writing a region's value in another shape of the same element count, as `reshape` would. | implicit reshape, auto-reshape |
| front end | The `taidl` package: it parses and checks an ISA description. | parser (for the whole), frontend, compiler |
| generator | A program that turns an accelerator into another artifact, such as TAIDL-TO or ACT-Backend. | backend (for the general case), emitter, tool |
| test oracle | TAIDL-TO's output: an executable functional model of the ISA. | simulator (unqualified), emulator |
| compiler backend | ACT-Backend's output: a compiler from XLA-HLO kernels to instructions. | backend (unqualified), codegen |
| kernel | A program a compiler backend takes (input kernel, in XLA-HLO) or emits (compiled kernel, of instructions); always qualified. | machine code, assembly, program (for either) |
| unsupported feature | An ISA description feature a generator documents it does not handle, citing a section id. | limitation, gap, TODO |
| reject | To raise an error for an ISA description, as S9 lists. | refuse, deny, disallow, error out, honor (for its opposite) |
| undefined behavior | A case whose result the spec does not give; an ISA description that reaches it is in error. | unspecified, UB |
| implementation-defined | A case each generator decides and documents. | platform-specific, vendor-specific |
