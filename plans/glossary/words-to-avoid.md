# Words to avoid

Words to replace when they name a TAIDL concept in docs, error messages, or user-visible names.
Each row copies the banned column of `terms.md`; a CI grep checks docs against it.

| Use | Instead of |
| --- | --- |
| ISA specification | ISA description, spec (for the file), TAIDL specification, TAIDL definition, ISA definition, model file, config |
| TAIDL language specification (language specification) | the spec, standard, reference manual |
| tensor accelerator (accelerator) | target, device, chip |
| buffer | tensor buffer, data model (for one buffer), scratchpad (except for the hardware it models), memory, bank |
| hbm | global memory, off-chip memory, DRAM, main memory, host memory, HBM (as a name for `d0`) |
| access dimensions | addressed dimensions, outer dims, rows |
| base element | unit, row, cell |
| base element shape | unit shape, unit dim, inner dims, element shape, row shape |
| element type | dtype (outside NumPy, JAX, or PyTorch names), data type, precision |
| state variable (state) | control register, initial state, register (unqualified), machine state variable, global, flag |
| initial value | reset value, default |
| memory state | machine state, memory contents |
| instruction (abstract instruction where the contrast matters) | op, command, intrinsic |
| concrete instruction | call (for the instruction itself), instance, invocation |
| partially instantiated instruction (pii) | partial instruction, plan node |
| instruction attribute | calling attribute, parameter, field, operand (for an attribute), immediate |
| computational attribute | computation attribute, comp attr, config attribute |
| addressing attribute | address parameter, address attribute, pointer |
| derived attribute | virtual attribute, hidden field, add-on attribute |
| width layer | type system, precision, bits, add-on |
| encoding layer | binary format, bit layout, bits and stuff, add-on |
| encoding block | encoding annotation, bit layout, format block, add-on |
| encoded attribute | binary field, encoded field, real attribute |
| validity constraint (constraint) | guard, predicate, assertion, precondition (for a validity constraint) |
| state update | control register assignment, side effect, write-back |
| region | data slice, slice (alone), tile, window, access, operand (for a region), buffer (for a region) |
| input region / output region | input data slice, output data slice, source, destination, src, dst |
| start | offset, base, address, origin |
| size | extent, length, `len`, count, span, end |
| region shape | data slice shape, tile shape, access shape |
| overlap relation (relation) | hazard model, aliasing mode, overlap hint |
| tied | in-place (as a relation), aliasing, donation |
| apart | disjoint, non-aliasing |
| exact-or-apart | no partial overlap (as a relation name), partial alias |
| may overlap | unsafe overlap, hazard allowed |
| marker | overlap hint, alias constraint, flag, in-place mark, placement mark |
| shorthand | user-productivity helper, convenience, sugar, syntactic sugar |
| implicit constraint | automatic check, hidden guard |
| instruction semantics (semantics) | computation, abstract computation, HLO, body (for the semantics alone) |
| operation | line, statement, instruction (in an instruction semantics), node |
| tensor operator | operation word, opcode, op, HLO op, primitive |
| reference operator | reference operation, backing op, lowering |
| value | SSA value, variable, tensor, register |
| region value (input value / output value) | tile, parameter, argument, result |
| operand | argument, input (in an instruction semantics) |
| operation attribute | attribute (unqualified, in an instruction semantics), option, flag |
| contracting dimensions / batch dimensions | reduction dims |
| expression | formula, index expression |
| condition | boolean expression |
| reference | placeholder, symbol |
| presumed reshape | implicit reshape, auto-reshape |
| XLA-HLO | HLO (alone), XLA IR |
| front end | parser (for the whole), frontend, compiler |
| generator | backend (for the general case), emitter, tool |
| generator metadata | add-on, addon, generator add-on, plugin, extension, pragma, hint, annotation (for this data) |
| metadata entry | add-on entry, generator add-on, plugin block, extension block |
| test oracle generator | TAIDL-TO (as a prose name), Test Oracle Generator (title case), oracle generator |
| compiler backend generator | ACT-Backend Generator, Compiler Backend Generator (title case) |
| test oracle | oracle (in prose), generated oracle, simulator (unqualified), emulator |
| kernel function (kernel) | program (for the function), test program, assembly (for the function), kernel (bare, in prose) |
| compiler backend | backend (unqualified), codegen |
| tiled kernel | tensor kernel, input kernel, kernel (unqualified, for the compiler input) |
| assembly code | machine code, compiled kernel, binary |
| issue | call (for an instruction), launch, dispatch |
| program | instruction stream, kernel (for a sequence of issues) |
| instruction selection | lowering, isel |
| memory allocation | bufferization, malloc (in prose) |
| e-graph / e-class / e-node | egraph, eclass, enode |
| rewrite rule (IR-to-IR, IR-to-ISA) | rewrite (alone), pattern |
| identity instruction | no-op instruction, nop |
| extraction / extractor | selection (for this step), picker |
| sound / complete | correct, valid (for either) |
| unsupported feature | limitation, gap, TODO |
| coverage status | handled, works, missing, unsupported (as a status) |
| rejection (reject) | refuse, refuses, refused, refusal, deny, disallow, error out, fail (for a rejection) |
| fault | undefined behavior, UB, implementation-defined, unspecified |
| dropped candidate | failed candidate, skipped candidate, pruned candidate (for a failure), silent fallback |
| invalid input kernel | bad kernel, broken kernel, unsupported kernel (for an invalid one), malformed tiled kernel |
| conforming | compliant, compatible |
| ACT ecosystem | toolchain, stack, suite |
| authoring surface (surface) | DSL, meta-DSL, semantics text, string form (for v3), TAIDL syntax |
| tracer | source parser, interpreter, compiler (for the tracer) |
| call site | location, origin, trace line |
| check function (check) | validate, validator, `Accelerator.validate()`, lint, pre-check |
| checked model (model) | validated accelerator, typed model (after check), AST, tree, IR |
| generator check | backend validation, checker subclass, check registry, plugin check |
| excluded instruction (exclusion) | skipped instruction, ignored instruction, silently dropped instruction |
| converter | migrator, `taidl migrate`, upgrade tool, compatibility shim |
| uses | leverages |
| the concrete claim | seamless support |
| from an ISA specification alone | just (as in "just from TAIDL specifications") |
| applies | honored, respected |
