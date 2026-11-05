# act-oracle naming cleanup

The oracle's code and docs use several names for one concept (buffer and data model, tile and input value, refuse and reject) and some vague helper names (`lhs_util`, `*_info`).
The canonical terms come from the language glossary (L track); the table below is the todo, none of it applied.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O15 | Apply the naming table: identifiers, comments, docs | O1, L25, L27 | M | Q1 |
| O16 | Generation log lines that state what happens | none | S | Q1 |

## O15 Apply the naming table

- Breaking: yes.
- Changes: `taidl_to/`, `README.md`, `docs/`; preserves public names `hbm`, `kernel`, `fsim`, `fsim-compile`, `comp_attrs`, `attrs`.
- What: rename internal identifiers in the table below (`lhs_util`, `parameter_util`, `*_info`, `sh_*`, `buffer`, `tile`, `window`, `fsim_compile` slot, `SEMANTIC_INIT`).
- What: replace "refuse" by "reject" and "buffer" by "data model" in README, docs and error messages.
- What: keep public names: `hbm`, `kernel`, the `fsim` and `fsim-compile` modes, and the generated `comp_attrs` and `attrs`, which are reserved names in S4.1.
- Benefits: one term per concept across spec, oracle and backend; grep finds every use.
- Caveats: error messages are matched by tests (`tests/test_refusals.py`); messages change with their tests in one commit.
- Caveats: renaming `sh_*` touches the generated `api.py` text and so the reserved-name set computed at `taidl_to/api_generator.py:61-86`.
- Fail points: a rename that collides with a reserved name rejects a previously valid spec; diff the reserved set before and after.
- Order and dependencies: after O1 so files are renamed once; after L25 settles the glossary; L27 applies it across tracks.
- Verification: `grep -rniE 'refus|buffer|_util\b|_info\b' taidl_to README.md docs` empty but for public names; suite green.

## O16 Generation log lines

- Breaking: no.
- Changes: `taidl_to/generator.py`; preserves the generation steps.
- What: `generator.py:63` prints "Copied generic oracle structure to" the final path while copying into a staging directory.
- What: print one line per phase that is true when printed, or none, leaving progress to the caller.
- Benefits: logs match what happened when a build fails mid-way.
- Caveats: users may scrape these lines; they are not documented as API.
- Fail points: none beyond the message text.
- Order and dependencies: independent.
- Verification: read the output of one failing and one passing generation.

## Naming table

| Old | New | Where | Applied? |
| --- | --- | --- | --- |
| `lhs_util` | `result_name` | `taidl_to/generic/utils.py:304` | no |
| `parameter_util` | `numbered_name` | `taidl_to/generic/utils.py:315` | no |
| `result_type_info`, `operand_info`, `attribute_info`, `attr_info`, `shape_info` | the unpacked names directly, such as `lhs_type, lhs_shape = self.visit(...)` | `taidl_to/oracle_visitor.py:225,247,268,403,441,453` | no |
| `fsim_compile` (template slot) | `body` | `taidl_to/instruction_codegen.py:183,189`, `taidl_to/templates/templates.txt:318` | no |
| `fsim`, `fsim-compile` (kernel modes) | keep: public API; define once in README | `taidl_to/generic/decorator.py:149-200` | no, public |
| `SEMANTIC_INIT`, `SEMANTIC_COUNTER`, `PROLOGUE_INIT`, `generate_semantic_init` | `MODULE_STATE`, `DATA_MODEL_COUNTER`, `DATA_MODEL_PROLOGUE`, `generate_module_state` | `taidl_to/templates/templates.txt:451,466,469`, `taidl_to/api_generator.py:12` | no |
| `semantic_generator.py`, `generate_semantic_code` | `semantics_codegen.py`, `generate_semantics` | `taidl_to/semantic_generator.py` | no |
| `sh_*` prefix (16 emitters) | `emit_*` | `taidl_to/generic/utils.py:459-850` | no |
| `op_name`, `op_to_template`, `op` | `word`, `word_template` (glossary: operation word) | `taidl_to/oracle_visitor.py:208-449` | no |
| `buffer`, `buffer_name`, `rhs_name` (a data model) | `data_model` | `taidl_to/generic/utils.py:698-713`, `taidl_to/instruction_codegen.py:63-123` | no |
| `tile` (a region's value) | `region value` (terminology reference) | `taidl_to/generic/utils.py:733-748`, `taidl_to/instruction_codegen.py:22-30,69-76` | no |
| `window` (the cells a region spans) | `region` (terminology reference) | `taidl_to/generic/utils.py:698,807-850` | no |
| `extent` (a data model's dimension) | `size` of the data model's dimension | `taidl_to/generic/utils.py:700-712` | no |
| refuse, refusal, refused | reject, rejection (glossary) | `README.md:54-142`, `docs/operations.md:125-145`, comments across `taidl_to/` | no |
| HBM (prose) | off-chip memory, keeping the `hbm` argument | `README.md:94-103`, `docs/operations.md:15` | no |
| `comp_attrs`, `attrs` (generated dicts) | keep: reserved names in the generated `api.py`; renaming changes the reserved list in S4.1 | `taidl_to/expression_writer.py:6` | no, cross-track |
| `generic/` (directory) | `runtime/`: it holds what the generated oracle runs, not generic code | `taidl_to/generic/`, `taidl_to/generator.py:52` | no |
| `utils.py` | split by role, see `plan/checks.md` | `taidl_to/generic/utils.py` | no |
| "presumed reshape" (glossary) vs "of its element count, read or written row-major" | presumed reshape | `README.md:122`, `taidl_to/generic/utils.py:734-735` | no |
| `Unroll` (exception used for control flow) | `ChangesOverLoop` | `taidl_to/generic/utils.py:750` | no |
| dtype (prose, outside NumPy names) | element type | `README.md:96,105` | no |
| assembly (prose for a kernel file) | compiled kernel | `README.md:46`, `examples/*/assembly/` (directory name kept unless A renames it) | no |
| just (filler) | delete | `README.md:3` | no |
