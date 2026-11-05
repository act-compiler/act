# TAIDL language specification

The TAIDL language specification, in sections S1 to S12; each page holds the sections listed beside it.

- [overview.md](overview.md): S1, scope, conformance, and the faults and defined behavior table S1.3
- [grammar.md](grammar.md): S2 lexical structure and S3 grammar, naming the normative `.g4` files
- [declarations.md](declarations.md): S4.1 to S4.5, names, the accelerator, buffers, `d0`, state, and the pointer to names TAIDL-TO reserves
- [object_model.md](object_model.md): S4.6, the five tree names a generator reads that the language fixes
- [instructions.md](instructions.md): S5.1 to S5.5, attributes, regions, the one constraint mechanism, state updates as a parallel assignment, overlap relations and their marks
- [execution.md](execution.md): S5.6, programs, storage before the first issue, the steps of one issue, and fault guarantees
- [expressions.md](expressions.md): S6, references, 64-bit integer arithmetic per operator, conditions
- [semantics.md](semantics.md): S7, semantics blocks, element types, shapes, literals
- [operations.md](operations.md): S8.1 to S8.3, operation words, their operands, attributes, and element types
- [operation_meaning.md](operation_meaning.md): S8.4 and S8.5, reference pin, edge cases, conversions, float rounding and accumulation bounds, and the operations outside the language
- [conformance.md](conformance.md): S9, the failure policy and what a conforming front end rejects
- [glossary.md](glossary.md): S10, the canonical terms
- [layers.md](layers.md): S11, the optional width layer, encoding layer, and derived-attribute rule over the 64-bit default
- [generator_metadata.md](generator_metadata.md): S12, generator metadata, metadata entries, and the steps for adding a generator
