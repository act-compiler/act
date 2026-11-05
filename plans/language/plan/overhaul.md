# Overhaul: target structure and steps

The target keeps every existing piece and reshapes four of them; nothing is replaced from scratch.
Steps run in order; each names the plan items it carries.

## Per piece

| piece | action | target | preserved |
| --- | --- | --- | --- |
| `IDLV2.g4` | keep, small edits in v3 | rule `instruction` renamed `line`; one `INT` rule | every token and parser rule's language |
| `IDLV2Expr.g4` | keep | one `INT` rule | everything else |
| `IDLV2*.py` | regenerate only | from the edited grammars | generation script `generate.sh` |
| `taidl/operations.py` | keep, extend | adds per-word shape and type rules beside `OPERATIONS` | the table and its checks |
| `taidl/instruction.py` | reshape | place check for references; shape checking moved to a new `shapes.py` | every public method and its checks |
| `taidl/accelerator.py` | keep | order check on `add_data_model`, range check on initial values | the API |
| `taidl/data_model.py` | reshape | glossary field names; generator-only strings removed | `TYPES`, `State`, the constructor's arguments |
| `taidl/names.py` | keep | | all |
| `docs/declarations.md` | reshaped | S4 with subsections | all facts |
| `docs/instructions.md` | reshaped | S5 with subsections | all facts |
| `docs/expressions.md` | reshaped | S6 with subsections | all facts |
| `docs/semantics.md` | reshape | S7, one sentence per line | all facts |
| `docs/operations.md` | reshape, split | S8 split into words, per-word table, types, meaning, outside | all tables |
| `docs/overview.md`, `grammar.md`, `conformance.md`, `glossary.md` | new | S1, S2 and S3, S9, S10 | |
| `README.md` | keep | docs list updated | all else |

## Steps

1. Land the reshaped and new docs with section ids (L1).
2. Pin the reference version and write the per-word table (L2, L3).
3. Fix the open meanings (L4, L5, L6) with B and O.
4. Add the front-end checks that need no grammar change (L10, L11, L14, L16).
5. Batch grammar and rename changes into v3 (L13, L15) with B and O.
6. Add the shape checker (L12) on the L3 table.
7. Add extensions in rank order (L17 onward).
