# Checks done late in a generator that belong in the front end

A check belongs in the front end when it needs only the ISA description and its answer does not depend on one generator's limits.
The `Move` column says what happens in v3: the front end rejects it (a row in S9), or it stays a run-time violation (class `V`) that the oracle raises and a generator either checks or rejects as unsupported.

| Check | Where it runs today | Why it is not generator-specific | Move |
| --- | --- | --- | --- |
| Outputs declared and no semantics | `act-oracle/taidl_to/semantic_generator.py:16`; the backend skips the instruction silently at `act_backend/generator.py:119-121` | Two generators disagree on the same description | Reject in `add_instruction` end or at generation entry (B24 in `work/backend/plan`) |
| Semantics read without `set_inputs` | `act_backend/spec_checks.py:195`, `semantic_generator.py:40` | `taidl/taidl/instruction.py:90, :210-215` skips `check_values` until both calls ran | Check at the end of the description, S9 Semantics row |
| Region value element count differs from its region | `spec_checks.py:415`, `semantic_generator.py:53` | S7.2 and S1.3 call it undefined and unchecked | Reject for literal sizes (L12), keep call-time check in the oracle |
| Region value element type differs from its data model | `semantic_generator.py:45` only | Same | Reject (B22, L12) |
| `constant` literal outside its element type | `spec_checks.py:75`, `oracle_visitor.py:387` | Depends on S7.3 only | Reject (L16, F12) |
| Computed divisor `0` over literals | `spec_checks.py:210, :292` | A pure integer fold (S6.2) | Reject; the front end folds integers (L14) |
| Computed value outside signed 64 bits | `spec_checks.py:218, :294` | v3 has no implementation-defined width: overflow is a defined run-time error | Front end rejects literal overflow (`R`); a value computed from attributes is a `V` row the oracle raises and the backend rejects as `U` where it cannot check |
| Data model dimension above 2**63-1 | `spec_checks.py:302` | A property of the declaration | Reject in `add_data_model` (`taidl/taidl/accelerator.py:41`) |
| Initial state value outside signed 64 bits | none; S4.4 calls it undefined | A property of the declaration | Reject in `add_initial_state` (`accelerator.py:56`, F11) |
| `@a.` inside a semantics or a size | `spec_checks.py:203, :287` | S6.1 calls it undefined and unrejected | Reject outside a start or a constraint (L10) |
| Size or dimension of 0 or less | `generic/utils.py:662` at a call | S5.2 calls it undefined | Reject literals in the front end; keep the call-time check for expressions |
| Shape and element-type agreement per word | `generic/shapes.py` (oracle only) | S8.4 calls it undefined and unchecked | Move (L12, B23, O7) |
| Word at an element type outside its S8.3 row | `oracle_visitor.py:255` | S8.3 is language data | Move with the S8.3 table (L3, O6) |
| Reserved names | `generator.py:41`, `api_generator.py:59`, `instruction_codegen.py:202`, `semantic_generator.py:29` | Generator-specific | Keep as `U` rows; list them in the oracle's unsupported-feature table |
| State updates and state reads | `spec_checks.py:375, :382` | Generator-specific | Keep as `U` rows |

## Order problem

- The backend runs its checks in the order `outputs, semantics, operands, data models, constraints, state, regions` (`generator.py:125-131`).
- A description with several faults reports whichever check runs first, so the message depends on the generator, not on the description.
- One shared pass in the front end fixes the order for R rows (D4); generator rows follow in a fixed order (D6).
