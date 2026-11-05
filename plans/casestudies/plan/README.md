# Case-study plan items

One file per item; sizes assume 3-4 engineers and one coordinated major release (TAIDL v3).
An item that duplicates an L item points to it and keeps the evidence and the verification.
Dependency prefixes: L language, F semantics, B backend, O oracle, T tests, U shared code, P prior art.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| C1 | [Typed-view idiom and byte order](c01_typed_views.md) | F6, C10 | S | Q1 |
| C2 | [Strided regions: evidence that admits L19](c02_strided_regions.md) | C11 | S | Q1 |
| C3 | [Instruction families and indexed state: idiom](c03_instruction_families.md) | C10 | S | Q1 |
| C4 | [Mode-dependent semantics: idiom](c04_mode_semantics.md) | C11 | S | Q1 |
| C5 | [Body operations: evidence that admits L20](c05_body_operations.md) | C12 | S | Q2 |
| C6 | [Added element types: evidence for L17](c06_element_types.md) | L17 | S | Q1 |
| C7 | [Numeric annotations: evidence for L21](c07_numeric_annotations.md) | C10 | S | Q2 |
| C8 | [Data-dependent addressing: study that admits L23](c08_data_dependent_addressing.md) | C11 | M | Q2 |
| C9 | [Float attributes and bit operators: idiom](c09_scalar_extensions.md) | C10 | S | Q1 |
| C10 | [Modeling cookbook and scope notes](c10_modeling_cookbook.md) | L, T | S | Q1 |
| C11 | [Case studies as an executable corpus](c11_executable_corpus.md) | T, O | M | Q1 |
| C12 | [Re-run the case studies on v3, add NVDLA](c12_rerun_on_v3.md) | C1, C8, C10, C11, L17 | S | Q2 |
| C13 | [Defined-behavior audit of the case studies](c13_defined_behavior_audit.md) | L, F, O, T, C11 | S | Q1 |
