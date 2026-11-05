# Corpus coverage of the spec outline

Each outline section gets one row: what kind of case covers it, where the first cases come from, and which item lands them.
Sections with no case say so, and why.

| section | case kind | first cases from | item |
| --- | --- | --- | --- |
| S1.1 conforming front end | none; defines what T6 asserts | - | T6 |
| S1.2 conforming generator, unsupported features | none; defines the `supported` and `rejected` rows | - | T11 |
| S1.3 implementation-defined behavior and the run-time norm | `implementation-defined` cases with no generator expectation; `error` cases for every run-time violation the norm defines | `act-oracle/tests/test_kernel.py:590-749`, :1229, `act-backend/tests/test_refusals.py:351-358` | T5, T8 |
| S2.1, S2.2 tokens | `semantics.txt` cases through the lexer | `taidl/tests/test_grammar.py:52-144`, :351-388 | T7 |
| S3.1, S3.2 grammar | `semantics.txt` and expression cases through the parser | `test_grammar.py:145-316`, `test_expressions.py:23-108` | T7 |
| S4.1 names | API cases | `test_instruction.py:48-100` | T7 |
| S4.2 accelerator | API cases | `test_instruction.py:101-115` | T7 |
| S4.3 data models and `d0` | API cases | `test_data_model.py:24-63`, `act-backend/tests/test_constructs.py:644` | T7 |
| S4.4 state variables | API cases | `test_instruction.py:441-469` | T7 |
| S4.5 declaration order | API cases | `test_instruction.py:500-585` | T7 |
| S5.1 instruction attributes | API cases | `test_instruction.py:116-134` | T7 |
| S5.2 regions | API cases with kernels | `test_instruction.py:195-440`, `act-oracle/tests/test_kernel.py:1287-1728` | T7 |
| S5.3 constraints | API cases with kernels | `test_expressions.py:139-187`, `act-oracle/tests/test_constraints.py:67-149` | T7 |
| S5.4 state updates and order | API cases with kernels | `test_expressions.py:188-222`, `act-oracle/tests/test_state.py:81-130` | T7 |
| S5.5 in-place and placement marks | API cases, plus kernel cases that read the marked input after the instruction and see the output | `test_instruction.py:269-293`, `act-backend/tests/test_refusals.py:383-396`, `test_extraction.py:441-563` | T7, T13 |
| S6.1 to S6.3 expressions | expression cases | `test_expressions.py:23-276` | T7 |
| S7.1 to S7.5 semantics blocks | `semantics.txt` cases | `test_grammar.py:389-612`, `act-backend/tests/test_refusals.py:224-304` | T7 |
| S8.1, S8.2 words, operands, attributes | `semantics.txt` cases per word | `test_grammar.py:451-612`, `act-oracle/tests/test_templates.py` | T7, T8 |
| S8.3 element types per operation | reference-value cases per word and type | the six act-oracle reference files | T8 |
| S8.4 meaning and edge cases | reference-value cases with closed forms | `test_reduce.py:117-129`, `test_constant_iota.py:115-143`, `test_linalg.py` | T8 |
| S8.5 operations outside the language | `rejected` cases | `test_grammar.py:284` | T8 |
| S9 what a front end rejects | every `rejected` case cites one S9 clause | all rejected cases | T6 |
| S10 glossary | none | - | - |

- A section whose row names no existing test is a gap the corpus fills with new cases in the T7 or T8 PR for its outline file.
- The runner reports, per section, the count of accepted, rejected, and implementation-defined cases, so a section at zero is visible.
