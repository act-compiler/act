# Response to the implementer review

Rows answer `reviews/spec/implementer.md` questions Q1-Q38 in order.
A panel decision such as Q1.5 or Q9.2 is cited where it settles a row.

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| Q1 `Accelerator` attributes | accepted | `accelerator.py:19-30` | S4.6; L29 |
| Q2 buffer attributes | accepted | `data_model.py:43-50` | S4.6 |
| Q3 `d0` in the list | accepted | `accelerator.py:31`; `d0` is entry 0 | S4.6, S4.3; L30 |
| Q4 instruction name and attribute lists | accepted | `instruction.py:69-71`; `parameters` is the addressing list | S4.6; L15 |
| Q5 raw and parsed regions | accepted | `instruction.py:131-140`, `459-461` | S4.6 |
| Q6 tree node types | accepted | `parse_text` returns the entry rule's first child (`instruction.py:364`) | S4.6 Trees |
| Q7 backtick lookup | accepted | `instruction.py:249-252` maps token nodes | S4.6 Trees |
| Q8 attribute names by text | accepted | `name : IDENTIFIER or OPERATION` (`IDLV2.g4:28`) | S3.1, S4.6 |
| Q9 `parameter(k)` and literal access | partly | Accessor path stated; a worked tree waits for the examples item | S4.6; L9 |
| Q10 program form | accepted | No driver in `taidl/`; generators need one model | S5.6 |
| Q11 initial contents | accepted | Owner decision 16 | S5.6; D31 |
| Q12 `d0` size | accepted | Size is the `hbm` byte count, as O's `hbm` argument sizes it; past it faults | S4.3, S5.2, S1.3.3 |
| Q13 sub-byte layout | accepted | O packs sub-byte types (`utils.py:880-908`) | S8.4 |
| Q14 float types and libraries | partly | Bit layout named as `ml_dtypes`, which XLA uses; no library is mandated, since that adds a dependency | S7.3 |
| Q15 `d0` byte order | accepted | Two generators must share binaries | S8.4 |
| Q16 state width and wrap | accepted | Exact arithmetic, fault past 64 bits | S6.2, S4.4 |
| Q17 failed constraint | accepted | Owner decision 16 | S1.3.5, S5.6 |
| Q18 placement constraints location | accepted | Appended to `constraints` (`instruction.py:228-231`) | S4.6 |
| Q19 `ROOT` at run time | accepted | Front-end check only (`instruction.py:276-318`) | S7.1, S5.6 |
| Q20 line order | accepted | Operands must be defined earlier (`operations.py:185-188`, `215-216`) | S5.6 |
| Q21 regions without semantics | accepted | Not rejected today (`instruction.py:99`, `254-257`) | S5.2, S9; L11 |
| Q22 scalar region shape | partly | Rule stated; a worked example waits for the examples item | S5.6; L9 |
| Q23 line dimension 0 or less | accepted | Owner decision 16 | S7.4, S1.3.7 |
| Q24 attribute expressions read post-update state | accepted | Same evaluation point as shapes | S5.4, S5.6 |
| Q25 negative axes | accepted | StableHLO requires axes in `[0, rank)` | S8.4 |
| Q26 reference revision | partly | Pinned to the oracle's jax, as ambiguity T10 | S8.4; L2 |
| Q27 StableHLO vs XLA-HLO edge cases | accepted | XLA CPU values are what O computes today | S8.4 |
| Q28 `compare` default | accepted | `oracle_visitor.py:114-118` | S8.4 |
| Q29 `dot` and bf16 accumulation | accepted | Two Higham bounds with u of the result type (Q1.5) | S1.3.15; L31 |
| Q30 reduce initial values | accepted | `utils.py:439-452` | S8.4 |
| Q31 `constant` fit | accepted | As ambiguity T1 | S7.5 |
| Q32 operand order | accepted | The reference operation fixes it | S8.4 |
| Q33 `pred` and broadcast shapes | partly | Shapes follow the reference until the per-word rules land | L3 |
| Q34 trust the line's type and shape | partly | A generator rejects a mismatch until the front end checks it; then it may trust the line | S8.3, S8.4; L12 |
| Q35 one exception type for unsupported features | rejected | Generators are separate programs; S1.2 fixes the message content (feature and section id), and a Python class name binds no Rust output | none |
| Q36 example per S5 feature | partly | Valid; `tests/test_examples.py` runs two examples, so new examples ride on the examples item | L9 |
| Q37 `copy` is identity | accepted | Reference operation | S8.4 |
| Q38 float tolerance | accepted | Same norm as Q29 | L31 |
