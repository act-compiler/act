# Case-study plan items

One file per item; sizes assume 3-4 engineers and one coordinated major release (TAIDL v3).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| C1 | [Typed views of storage](c01_typed_views.md) | L, F, B, O, T | M | Q1 |
| C2 | [Strided and multi-dimensional regions](c02_strided_regions.md) | C1, L, B, O, T | L | Q1 |
| C3 | [Instruction families and indexed state](c03_instruction_families.md) | L, S, B, O | M | Q1 |
| C4 | [Mode-dependent semantics](c04_mode_semantics.md) | L, F, B, O, T | M | Q1 |
| C5 | [Body-taking operations and convolution](c05_body_operations.md) | P, L, O, B, T | L | Q2 |
| C6 | [Block-scaled and added element types](c06_element_types.md) | P, L, F, O, B | M | Q2 |
| C7 | [Numeric annotations](c07_numeric_annotations.md) | L, O, D | M | Q2 |
| C8 | [Data-dependent addressing](c08_data_dependent_addressing.md) | L, O, B | L | Q2 |
| C9 | [Float attributes and bit operators](c09_scalar_extensions.md) | L, S, B, O | S | Q1 |
| C10 | [Modeling cookbook and scope notes](c10_modeling_cookbook.md) | L, T | S | Q1 |
| C11 | [Case studies as an executable corpus](c11_executable_corpus.md) | T, O | M | Q1 |
| C12 | [Re-run the case studies on v3](c12_rerun_on_v3.md) | C1, C2, C3, C4, C5, C6, C7, C8, C9, C11 | S | Q2 |
| C13 | [Defined-behavior audit of the case studies](c13_defined_behavior_audit.md) | L, F, O, T, C11 | S | Q1 |
