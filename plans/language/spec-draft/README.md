# TAIDL language spec

The spec of TAIDL, in sections S1 to S10; each page holds the sections listed beside it.

- [overview.md](overview.md): S1, scope, conformance, and the faults and defined behavior table S1.3
- [grammar.md](grammar.md): S2 lexical structure and S3 grammar, naming the normative `.g4` files
- [declarations.md](declarations.md): S4.1 to S4.5, names, the accelerator, buffers, `d0`, state
- [object_model.md](object_model.md): S4.6, the objects and attributes a generator reads
- [instructions.md](instructions.md): S5.1 to S5.5, attributes, regions, constraints, state updates, marks
- [execution.md](execution.md): S5.6, programs, initial storage, and the steps of one issue
- [expressions.md](expressions.md): S6, references, integer arithmetic, conditions
- [semantics.md](semantics.md): S7, semantics blocks, element types, shapes, literals
- [operations.md](operations.md): S8.1 to S8.3, operation words, their operands, attributes, and element types
- [operation_meaning.md](operation_meaning.md): S8.4 and S8.5, reference pin, edge cases, and the operations outside the language
- [conformance.md](conformance.md): S9, what a conforming front end rejects
- [glossary.md](glossary.md): S10, the canonical terms
