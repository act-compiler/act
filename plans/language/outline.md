# TAIDL spec outline

Section ids are stable; other tracks cite them as `S<n>` or `S<n>.<m>`, and S1.3 rows as `S1.3.<n>`.
Each section lives in the `taidl/docs/` page named beside it.

| id | title | file |
| --- | --- | --- |
| S1 | Scope, conformance, and notation | `overview.md` |
| S1.1 | Conforming front end | `overview.md` |
| S1.2 | Conforming generator and unsupported features | `overview.md` |
| S1.3 | Faults and defined behavior | `overview.md` |
| S2 | The Python surface | `grammar.md` |
| S2.1 | Names a specification imports | `grammar.md` |
| S2.2 | Operators on symbolic values | `grammar.md` |
| S3 | Tracing | `grammar.md` |
| S3.1 | Instruction functions and families | `grammar.md` |
| S3.2 | What tracing rejects | `grammar.md` |
| S4 | Declarations | `declarations.md` |
| S4.1 | Names | `declarations.md` |
| S4.2 | Accelerator | `declarations.md` |
| S4.3 | Buffers and the hbm buffer `d0` | `declarations.md` |
| S4.4 | State | `declarations.md` |
| S4.5 | Declaration order | `declarations.md` |
| S4.6 | The `Model` | `object_model.md` |
| S5 | Instructions | `instructions.md` |
| S5.1 | Instruction attributes | `instructions.md` |
| S5.2 | Regions: inputs and outputs | `instructions.md` |
| S5.3 | Constraints | `instructions.md` |
| S5.4 | State updates | `instructions.md` |
| S5.5 | Overlap between inputs and outputs | `instructions.md` |
| S5.6 | Programs and issues | `execution.md` |
| S6 | Expressions | `expressions.md` |
| S6.1 | References and where each may appear | `expressions.md` |
| S6.2 | Integer arithmetic | `expressions.md` |
| S6.3 | Conditions | `expressions.md` |
| S7 | Semantics | `semantics.md` |
| S7.1 | Values and operations | `semantics.md` |
| S7.2 | Input and output values | `semantics.md` |
| S7.3 | Element types | `semantics.md` |
| S7.4 | Shapes | `semantics.md` |
| S7.5 | Operands and operation attributes | `semantics.md` |
| S8 | Operations | `operations.md` |
| S8.1 | Tensor operators and their reference operators | `operations.md` |
| S8.2 | Operands and operation attributes per operation | `operations.md` |
| S8.3 | Element types per operation | `operations.md` |
| S8.4 | Meaning and edge cases | `operation_meaning.md` |
| S8.5 | Operations outside the language | `operation_meaning.md` |
| S9 | The check function, rejections, and generator checks | `conformance.md` |
| S10 | Glossary | `glossary.md` |
| S11 | Optional layers | `layers.md` |
| S11.1 | Layers and the default | `layers.md` |
| S11.2 | The width layer | `layers.md` |
| S11.3 | The encoding layer | `layers.md` |
| S11.4 | The derived-attribute rule | `layers.md` |
| S12 | Generator metadata | `generator_metadata.md` |
| S12.1 | Metadata entries | `generator_metadata.md` |
| S12.2 | What the front end does | `generator_metadata.md` |
| S12.3 | What a generator does | `generator_metadata.md` |
| S12.4 | Adding a generator | `generator_metadata.md` |
