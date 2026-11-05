# Corpus coverage of the language specification outline

Each outline section gets one row: what kind of case covers it, where the first cases come from, and which item lands them.
Sections with no case say so, and why.

| section | case kind | first cases from | item |
| --- | --- | --- | --- |
| S1.1 conforming front end | none; defines what T6 asserts | - | T6 |
| S1.2 conforming generator, unsupported features, exclusion | exclusion rows beside the `supported` and `unsupported` rows | - | T11, T38 |
| S1.3 run-time norm | `program.toml` cases per norm row, expecting a result or a `fault`; the catalogue is `norms.md` | `act-oracle/tests/test_kernel.py:590-749`, :1229, `act-backend/tests/test_refusals.py:351-358` | T26, T29, T8, T30 |
| S2.1, S2.2 tokens | none in v3 for the string form; names and literals are Python's, with `trace` rows for the spellings it rejects | `taidl/tests/test_grammar.py:52-144`, :351-388 | T38 |
| S3.1, S3.2 grammar | traced expression cases and `trace` rejection rows | `test_grammar.py:145-316`, `test_expressions.py:23-108` | T7, T38 |
| S4.1 names | surface cases | `test_instruction.py:48-100` | T7 |
| S4.2 accelerator | surface cases | `test_instruction.py:101-115` | T7 |
| S4.3 buffers and `d0` | surface cases | `test_data_model.py:24-63`, `act-backend/tests/test_constructs.py:644` | T7 |
| S4.4 state variables | surface cases | `test_instruction.py:441-469` | T7 |
| S4.5 declaration order | surface cases | `test_instruction.py:500-585` | T7 |
| S5.1 instruction attributes | surface cases | `test_instruction.py:116-134` | T7 |
| S5.1 attribute types and encodings | width, encoding, derived-attribute, and metadata cases, plus the assembler round trip | `act-oracle/taidl_to/generic/utils.py:268-275`, `isel_ir2isa_rewrites_rs_generator.py:166-181` | T32 to T35 |
| S5.2 regions | surface cases with kernels and `program.toml` | `test_instruction.py:195-440`, `act-oracle/tests/test_kernel.py:1287-1728` | T7, T29 |
| S5.3 constraints | surface cases with kernels | `test_expressions.py:139-187`, `act-oracle/tests/test_constraints.py:67-149` | T7 |
| S5.4 state updates and order | surface cases with kernels; state updates read the pre-state (Q1.1) | `test_expressions.py:188-222`, `act-oracle/tests/test_state.py:81-130` | T7 |
| S5.5 overlap relations and markers | the T30 matrix of five declarations by three placements, plus rejection, size-defect, negative-attribute, written-constraint, and cross-check rows | `test_instruction.py:269-293`, :294-440, `act-backend/tests/test_refusals.py:383-396`, `test_extraction.py:441-563` | T30, T31 |
| S6.1 to S6.3 expressions | expression cases | `test_expressions.py:23-276` | T7 |
| S7.1 to S7.5 semantics blocks | instruction-function cases | `test_grammar.py:389-612`, `act-backend/tests/test_refusals.py:224-304` | T7 |
| S8.1, S8.2 words, operands, attributes | `ops.<word>` cases per word | `test_grammar.py:451-612`, `act-oracle/tests/test_templates.py` | T7, T8 |
| S8.3 element types per operation | stored-value cases per word and type | the six act-oracle reference files | T8 |
| S8.4 meaning and edge cases | stored-value cases with closed forms | `test_reduce.py:117-129`, `test_constant_iota.py:115-143` | T8 |
| S8.5 operations outside the language | `rejected` cases | `test_grammar.py:284` | T8 |
| S1.3.10 value outside its type | boundary and fault rows per width | `utils.py:268-275` | T32 |
| S9 what a front end rejects | every `rejected` case cites one S9 clause | all rejected cases | T6 |
| S10 glossary | none | - | - |

- A section whose row names no existing test is a gap the corpus fills with new cases in the T7 or T8 PR for its outline file.
- The `coverage.md` report of T26 counts accepted and rejected cases per S id, S9 clause, and norm row, so a row at zero is visible.
- CI fails on a zero row outside the allowlist: S1.1, S1.2, S10, and pending rows.
