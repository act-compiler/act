# Overhaul: target structure and steps

The target keeps the front end's checks and tables, replaces the string surface and its grammars with the Python surface, and adds the check function.
Steps run in order; each names the plan items it carries.

## Per piece

| piece | action | target | preserved |
| --- | --- | --- | --- |
| `IDLV2.g4`, `IDLV2Expr.g4`, `IDLV2*.py` | remove in v3 (U14) | none; the tracer records the surface (U12) | the language they accept, through the U13 converter |
| tracer | new (U12) | `@acc.instruction`, `read`, `write`, `require`, `update`, `ops` | |
| `taidl.check` and `Model` | new (L43) | one `Rejection` or a frozen `Model` | every S9 rule |
| `taidl/operations.py` | keep, extend | adds per-word shape and type rules beside `OPERATIONS` | the table and its checks |
| `taidl/instruction.py` | reshape | place check for references; pair relations, `set_may_overlap`, and the `same` size term (L35, L36); shape checking in the shape module of U4 | every public method and its checks |
| `taidl/accelerator.py` | keep | `add_buffer` and `add_state` per owner decision 19; order and range checks; `d0` with one access dimension | the rest of the API |
| `taidl/data_model.py` | keep | `num_dim_str` removed; other renames in the L30 table | `TYPES`, `State`, the constructor's arguments |
| `taidl/names.py` | keep | | all |
| `docs/declarations.md` | reshaped | S4.1 to S4.5 | all facts |
| `docs/instructions.md` | reshaped | S5.1 to S5.5 | all facts |
| `docs/expressions.md` | reshaped | S6 | all facts |
| `docs/semantics.md` | reshape | S7, one sentence per line | all facts |
| `docs/operations.md` | reshape, split | S8 split into words, per-word table, types, meaning, outside | all tables |
| `docs/overview.md`, `grammar.md` (the surface), `object_model.md`, `execution.md`, `conformance.md`, `glossary.md` | new | S1, S2 and S3, S4.6, S5.6, S9, S10 | |
| `README.md` | keep | docs list updated | all else |

## Steps

1. Land the reshaped and new docs with section ids (L1, L29).
2. Pin the reference version and write the per-word rules (L2, L3).
3. Make the fault rows and stated results hold in B and O (L4, L5, L6).
4. Add the front-end checks (L10, L11, L16).
5. Build `check` and the `Model` from the string front end, and port every generator to it (L43).
6. Add the shape checks (U4, which absorbs L12) on the L3 rules.
7. Trace the Python surface into the same `Model`, with the API renames (U12, L14, L15, L30).
8. Convert the corpus and test `Model` equivalence per file (U13); then remove the string surface and ANTLR (U14).
9. Settle the float accuracy norm (L31).
10. Add extensions in rank order (L17, then L23).
