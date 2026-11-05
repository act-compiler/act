# Target split between the front end and the generators

Derived from the current code; paths are relative to /home/user.
Assumes one coordinated major release, TAIDL v3, and three to four engineers.

## Per module

| Module | Verdict | Target |
| --- | --- | --- |
| taidl/taidl/instruction.py | keep | Parsing, reference and literal checks stay until U14 deletes the string surface; gains no generator logic. |
| taidl/taidl/operations.py | reshape | Each row gains its shape rule and element-type set; attribute order becomes an exported, documented list. |
| taidl/taidl/data_model.py | reshape | `TYPES` becomes a table with width, signedness, float bits and range; `array_dim_str` and `num_dim_str` go. |
| taidl/taidl/accelerator.py | reshape | Authoring object; freezes when `check` runs. |
| new taidl/taidl/check.py | merge | `check(acc)` runs every whole-description rule once, collects every problem into one `Rejection`, and returns the frozen `Model` with the derived facts as fields; `python -m taidl check` calls it. |
| new taidl/taidl/trace.py | merge | The tracer: runs a decorated function once on symbolic attributes and records the node model with call sites (U12). |
| new taidl/taidl/shapes.py | merge | From act-oracle/taidl_to/generic/shapes.py:31-543 and act-oracle/taidl_to/oracle_visitor.py:19-97; imports nothing; entry takes typed nodes or `(type, dims)` tuples. |
| new taidl/taidl/integers.py | merge | From act-backend/act_backend/spec_checks.py:120-176 and the literal checks at taidl/taidl/instruction.py:400-437. |
| act-oracle/taidl_to/generic/shapes.py | replace | The generator copies the front end's shape module into each generated oracle, stamped with the taidl version; the oracle keeps only the call-time entry for expression shapes (act-oracle/taidl_to/generic/utils.py:771-786). |
| act-oracle/taidl_to/semantic_generator.py:36-56 | replace | Region checks move to the front end (U3); the oracle entry takes a `Model` or calls `check` once, adds no check, and reads the `Model` (U10). |
| act-oracle/taidl_to/template.py | reshape | Adopts the strict key check of act-backend/act_backend/template_loader.py:69-81. |
| act-oracle/taidl_to/expression_writer.py | keep | Python emitter. |
| act-backend/act_backend/spec_checks.py | reshape | Loses folding, region and type checks (:394-418 goes with U10); keeps the backend's own functions after `check`: an unknown `meta` key rejects, and an instruction with state, other than one output, a size over an addressing attribute, a constraint the allocator cannot hold, or an unsupported word is excluded with a coded reason (B14, B46). |
| act-backend/act_backend/expression_writer.py | reshape | Unchecked Rust floors like the checked path (X2). |
| act-backend/act_backend/generic/src/ir/dtype.rs | keep | Its variant list is checked against the front end's table by a test. |
| act-backend/act_backend/generic/src/isel/initializer/expression.rs:173-178 | replace | Generated from the front end's attribute order. |
| act-backend/act_backend/generic/src/ir/egraph.rs:403-409 | reshape | The `merge` panic becomes the shared coded fault `VF` at selection. |
| */docker/*.sh, */.github/workflows/test.yml | merge | One copy in `act`; each repo calls it with its prefix. |
| examples/mini_trn*.py in three repos | merge | One corpus in `taidl`, with expected verdicts. |
| taidl/taidl/antlr4/, string-taking calls | delete | Removed in v3 after the converter has run on the corpus (U13, U14). |

## Ordered steps

1. Fix the two divergences that need no new module: X2 in the backend, X9 in the oracle.
2. Publish the typed front-end model with `CONTRACT_VERSION` and pin it with a contract test in `taidl` (U2).
3. Export the attribute order and generate the Rust copy from it.
4. Add the element-type table and `literal_fits`; switch the backend's `reads_as` and the oracle's `mlir_literal` range check to it.
5. Build `check` over the string front end, returning the `Model`; port every generator to read only the `Model` (U3, U10).
6. Move the oracle's shape and per-word type rules into the front end, behind `check` (U4 PR 1), then add the new checks (U4 PR 2).
7. Add region-value checks to `check`; generators delete their copies (U3, U10).
8. Move integer folding into the front end; delete `integer` and `past_64_bits` from the backend.
9. Build the tracer to produce the same `Model` (U12).
10. Run the converter on the 2.x line over the corpus, commit the converted files, and pass one equivalence test per file (U13).
11. Land the conformance corpus; both generators run it in CI.
12. Delete the string surface, `taidl/antlr4/`, and the parser dependency (U14).
13. Cut v3 with released pins only: front end tag, then both generators, then the umbrella, gated by act/tests/check_versions.py.
14. Fold the docker and CI copies into `act`.
