# Overhaul: target structure and steps

The target keeps every existing piece and reshapes four of them; nothing is replaced from scratch.
Steps run in order; each names the plan items it carries.

## Per piece

| piece | action | target | preserved |
| --- | --- | --- | --- |
| `IDLV2.g4` | keep, small edits in v3 | one `INT` rule | every token and parser rule's language |
| `IDLV2Expr.g4` | keep | one `INT` rule | everything else |
| `IDLV2*.py` | regenerate only | from the edited grammars | generation script `generate.sh` |
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
| `docs/overview.md`, `grammar.md`, `object_model.md`, `execution.md`, `conformance.md`, `glossary.md` | new | S1, S2 and S3, S4.6, S5.6, S9, S10 | |
| `README.md` | keep | docs list updated | all else |

## Steps

1. Land the reshaped and new docs with section ids (L1, L29).
2. Pin the reference version and write the per-word rules (L2, L3).
3. Make the fault rows and stated results hold in B and O (L4, L5, L6).
4. Add the front-end checks that need no grammar change (L10, L11, L14, L16).
5. Batch grammar, API, and rename changes into v3 (L13, L15, L30) with B and O.
6. Add the shape checks (U4, which absorbs L12) on the L3 rules.
7. Settle the float accuracy norm after the panel (L31).
8. Add extensions in rank order (L17, then L23).
