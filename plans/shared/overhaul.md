# Target split between the front end and the generators

Derived from the current code; paths are relative to /home/user.
Assumes one coordinated major release, TAIDL v3, and three to four engineers.

## Per module

| Module | Verdict | Target |
| --- | --- | --- |
| taidl/taidl/instruction.py | keep | Parsing, reference and literal checks stay; gains no generator logic. |
| taidl/taidl/operations.py | reshape | Each row gains its shape rule and element-type set; attribute order becomes an exported, documented list. |
| taidl/taidl/data_model.py | reshape | `TYPES` becomes a table with width, signedness, float bits and range; `array_dim_str` and `num_dim_str` go. |
| taidl/taidl/accelerator.py | reshape | Gains one `validate()` that runs every whole-description check once. |
| new taidl/taidl/shapes.py | merge | From act-oracle/taidl_to/generic/shapes.py:31-543 and act-oracle/taidl_to/oracle_visitor.py:19-97. |
| new taidl/taidl/integers.py | merge | From act-backend/act_backend/spec_checks.py:120-176 and the literal checks at taidl/taidl/instruction.py:400-437. |
| act-oracle/taidl_to/generic/shapes.py | replace | Imports the front end's rules; keeps only the call-time entry for expression shapes (act-oracle/taidl_to/generic/utils.py:771-786). |
| act-oracle/taidl_to/semantic_generator.py:36-56 | replace | Region checks move to the front end. |
| act-oracle/taidl_to/template.py | reshape | Adopts the strict key check of act-backend/act_backend/template_loader.py:69-81. |
| act-oracle/taidl_to/expression_writer.py | keep | Python emitter. |
| act-backend/act_backend/spec_checks.py | reshape | Loses folding, region and type checks; keeps backend-only refusals (one output, supported words, state). |
| act-backend/act_backend/expression_writer.py | reshape | Unchecked Rust floors like the checked path (X2). |
| act-backend/act_backend/generic/src/ir/dtype.rs | keep | Its variant list is checked against the front end's table by a test. |
| act-backend/act_backend/generic/src/isel/initializer/expression.rs:173-178 | replace | Generated from the front end's attribute order. |
| */docker/*.sh, */.github/workflows/test.yml | merge | One copy in `act`; each repo calls it with its prefix. |
| examples/mini_trn*.py in three repos | merge | One corpus in `taidl`, with expected verdicts. |

## Ordered steps

1. Fix the two divergences that need no new module: X2 in the backend, X9 in the oracle.
2. Document the front-end object contract and pin it with a contract test in `taidl`.
3. Export the attribute order and generate the Rust copy from it.
4. Add the element-type table and `literal_fits`; switch the backend's `reads_as` and the oracle's `mlir_literal` range check to it.
5. Move the oracle's shape and per-word type rules into the front end, behind `Accelerator.validate()`.
6. Add region-value checks to `validate()`; delete the generator copies.
7. Move integer folding into the front end; delete `integer` and `past_64_bits` from the backend.
8. Land the conformance corpus; both generators run it in CI.
9. Cut v3: front end tag, then both generators, then the umbrella, gated by act/tests/check_versions.py.
10. Fold the docker and CI copies into `act`.
