# Checks done in both the front end and a generator

Each row is one condition checked in more than one place; a cell is `file:line` or `none`.
S-ids are those of the spec draft in `taidl/docs` (S1.3 lists undefined and implementation-defined cases; S9 lists rejections).

| Condition | Front end | act-backend | act-oracle | Spec |
| --- | --- | --- | --- | --- |
| Literal divisor `0` | `taidl/taidl/instruction.py:413-415` (only the literal `0` right after `/` or `%`) | `act_backend/spec_checks.py:210, :292, :255` (after folding integers) | `taidl_to/generic/utils.py:80` (each loop iteration) | S6.2, S9 Texts |
| Integer outside signed 64 bits | `taidl/taidl/instruction.py:404-436` (literals only) | `spec_checks.py:218, :294, :252, :302, :364` (computed values) | `generic/utils.py:268-271` (warning) | S6.2, S1.3 |
| `%In<k>` or `%Out<k>` against the regions | `instruction.py:153, :158` (only after both `set_inputs` and `add_semantics`) | `spec_checks.py:195` | `semantic_generator.py:40` | S7.2, S9 Semantics |
| Operand count of a word | `taidl/taidl/operations.py:207-210` | `spec_checks.py:66` (the backend's own arity, for example `concatenate` of 2) | `oracle_visitor.py:289` through `generic/shapes.py` | S8.2, S9 Semantics |
| ROOT position | `instruction.py:279, :315` | `spec_checks.py:192` (ROOT on the last line) | none | S7.1 |
| Region value larger than its region | none | `spec_checks.py:415` | `semantic_generator.py:53`, `generic/utils.py:743` | S7.2, S1.3 |
| Region value typed unlike its data model | none | none | `semantic_generator.py:45` | S7.2, S1.3 |
| `constant` literal fits its element type | none | `spec_checks.py:75` | `oracle_visitor.py:387`, `generic/utils.py:397-425` | S8.4 |
| Element type known | `accelerator.py:43` (19 types) | `spec_checks.py:48` (13 types) | `decorator.py:96` (call time), `oracle_visitor.py:255` | S7.3 |
| Operation word known | grammar plus `operations.py:OPERATIONS` | `spec_checks.py:58` (18 words) | `oracle_visitor.py:239` (dead, O5) | S8.1 |
| Dimension of a data model fits 64 bits | none | `spec_checks.py:302` | none | S4.3, S6.2 |

## Reading

- Same condition, up to three wordings and three exception types (`SyntaxError`, `ValueError`, `ZeroDivisionError` then `ValueError`).
- Where a generator re-checks a condition the front end already checks, the generator copy is dead or order-dependent; see `late.md`.
- A condition that S1.3 calls undefined is checked by one generator and not the other, so the same ISA description is accepted by one and rejected by the other.
