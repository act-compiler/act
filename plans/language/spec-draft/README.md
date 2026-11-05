# TAIDL language specification

The TAIDL language specification, in sections S1 to S12; each page holds the sections listed beside it.

- [overview.md](overview.md): S1, scope, conformance, and the faults and defined behavior table S1.3
- [grammar.md](grammar.md): S2 the Python surface and S3 tracing: imported names, operators on symbolic values, instruction families, and what tracing rejects
- [declarations.md](declarations.md): S4.1 to S4.5, names, `Accelerator`, `acc.buffer`, `d0`, `acc.state`, and the pointer to names the test oracle generator reserves
- [object_model.md](object_model.md): S4.6, the frozen `Model` that `check` returns and its printer
- [instructions.md](instructions.md): S5.1 to S5.5, the running example, typed parameters, `read` and `write`, `require`, `update`, overlap relations and their `write` markers
- [execution.md](execution.md): S5.6, programs, storage before the first issue, the steps of one issue, and fault guarantees
- [expressions.md](expressions.md): S6, references, 64-bit integer arithmetic with `//` and `%`, conditions with `any_of` and `all_of`
- [semantics.md](semantics.md): S7, values and `ops` calls, inferred types and shapes, element types, constants, operation attributes as keywords
- [operations.md](operations.md): S8.1 to S8.3, tensor operators, their operands, attributes, and element types
- [operation_meaning.md](operation_meaning.md): S8.4 and S8.5, reference pin, edge cases, conversions, float rounding and accumulation bounds, and the operations outside the language
- [conformance.md](conformance.md): S9, the failure policy, the check function, what it rejects, and generator checks with exclusion
- [glossary.md](glossary.md): S10, the canonical terms
- [layers.md](layers.md): S11, the optional width layer, encoding layer, and derived-attribute rule over the 64-bit default
- [generator_metadata.md](generator_metadata.md): S12, generator metadata, metadata entries, and the steps for adding a generator
