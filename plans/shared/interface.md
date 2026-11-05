# Interface: what the generators consume from taidl

Both generators take an `Accelerator` and read the parsed `Instruction` objects directly.
There is no serialized intermediate form: the contract is the Python attribute set of three classes plus the ANTLR parse trees.

## Accelerator and data model

| Member | Form | Backend reads at | Oracle reads at |
| --- | --- | --- | --- |
| `Accelerator.name` | str | act-backend/act_backend/generator.py:91 | act-oracle/taidl_to/generator.py:31 |
| `Accelerator.base_dir` | str | act-backend/act_backend/generator.py:88 | act-oracle/taidl_to/generator.py:28 |
| `Accelerator.instructions` | list[Instruction] | act-backend/act_backend/generator.py:121 | act-oracle/taidl_to/generator.py:47 |
| `Accelerator.data_model` | list[DataModel], `d0` first | act-backend/act_backend/generator.py:128 | act-oracle/taidl_to/generator.py:38 |
| `Accelerator.state` | list[State] | not read (refused per instruction) | act-oracle/taidl_to/api_generator.py:43 |
| `DataModel.var_name` | str; `d0` is a magic name | act-backend/act_backend/instruction_info.py:71 | act-oracle/taidl_to/api_generator.py:23 |
| `DataModel.access_dim` | list[int] | act-backend/act_backend/malloc_globals_cc_generator.py:14 | not read |
| `DataModel.unit_dim` | list[int] | act-backend/act_backend/malloc_instructions_h_generator.py:532 | act-oracle/taidl_to/instruction_codegen.py:68 |
| `DataModel.var_dim` | list[int] | act-backend/act_backend/spec_checks.py:300 | not read |
| `DataModel.var_type` | str of `TYPES` | not read | act-oracle/taidl_to/semantic_generator.py:44 |
| `DataModel.array_dim_str` | display string `['4','8']` | not read | act-oracle/taidl_to/api_generator.py:25, quotes stripped by `replace` |
| `State.name`, `State.value` | str, int | not read | act-oracle/taidl_to/api_generator.py:44 |

## Instruction

| Member | Form | Backend reads at | Oracle reads at |
| --- | --- | --- | --- |
| `instruction` | str name | act-backend/act_backend/generator.py:54 | act-oracle/taidl_to/instruction_codegen.py:150 |
| `comp_attr` | list[str] | act-backend/act_backend/generator.py:57 | act-oracle/taidl_to/instruction_codegen.py:158 |
| `parameters` | list[str], the addressing attributes | act-backend/act_backend/malloc_instructions_h_generator.py:444 | act-oracle/taidl_to/instruction_codegen.py:153 |
| `instr_inputs`, `instr_outputs` | raw regions, text or int | act-backend/act_backend/spec_checks.py:313 | act-oracle/taidl_to/instruction_codegen.py:122 |
| `instr_input_asts`, `instr_output_asts` | `[buffer, [start trees], [size trees]]` | act-backend/act_backend/generator.py:60 | act-oracle/taidl_to/instruction_codegen.py:64 |
| `instr_semantics_ast` | `IDLV2Parser.ModuleContext` or None | act-backend/act_backend/generator.py:121 | act-oracle/taidl_to/semantic_generator.py:13 |
| `instr_expression_asts` | dict TerminalNode -> expression tree | act-backend/act_backend/spec_checks.py:200 | act-oracle/taidl_to/instruction_codegen.py:28 |
| `constraints`, `constraint_asts` | parallel lists, text and tree | act-backend/act_backend/spec_checks.py:354 | act-oracle/taidl_to/instruction_codegen.py:36 |
| `update`, `update_asts` | parallel lists | act-backend/act_backend/spec_checks.py:374 (refusal only) | act-oracle/taidl_to/instruction_codegen.py:53 |
| `inplace` | list[bool] | act-backend/act_backend/instruction_info.py:112 | not read |
| `placed` | dict method -> trees | act-backend/act_backend/spec_checks.py:360 (message only) | not read |
| `data_models` | the accelerator's list | via `find_data_model` | act-oracle/taidl_to/semantic_generator.py:26 |
| `find_data_model()` | method | act-backend/act_backend/spec_checks.py:406 | act-oracle/taidl_to/semantic_generator.py:43 |

## Module-level contract

| What | Backend reads at | Oracle reads at |
| --- | --- | --- |
| `taidl.antlr4` visitors and token types | act-backend/act_backend/semantics_visitor.py:3, act-backend/act_backend/expression_writer.py:4 | act-oracle/taidl_to/expression_writer.py:3 |
| `taidl.operations.OPERATIONS` attribute order as payload order | act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:253 | not read |
| Same order, hand-copied in Rust | act-backend/act_backend/generic/src/isel/initializer/expression.rs:173-178 | not read |
| Lazy umbrella entry points | act/act/generators/__init__.py:4-15 | act/act/generators/__init__.py:4-15 |
| Re-exported classes | act/act/taidl/__init__.py:3 | act/act/taidl/__init__.py:3 |

## Documentation of the contract

- Only the tree attributes are documented: taidl/docs/expressions.md:42-47 names `constraint_asts`, `update_asts`, `instr_input_asts`, `instr_output_asts`, `instr_expression_asts`.
- `comp_attr`, `parameters`, `inplace`, `placed`, `array_dim_str`, the magic `d0`, and the attribute order of `OPERATIONS` are undocumented.
- The class annotations at taidl/taidl/instruction.py:34-57 are the nearest thing to a schema, without stability marks.
- `taidl/__init__.py:9` exports only the four classes; `operations` and `antlr4` are reached as internals.

## Where generators re-parse or re-derive

- The oracle parses `array_dim_str` back into dims (act-oracle/taidl_to/api_generator.py:25) though `var_dim` holds them.
- The backend regex-scans a shape's raw text for `@c.` names (act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:91, :134-137) though the trees hold `COMPUTATIONAL` tokens.
- Both re-match `%In<k>` and `%Out<k>` names by regex (act-oracle/taidl_to/semantic_generator.py:39; act-backend/act_backend/spec_checks.py:193-198) that taidl already checks at taidl/taidl/instruction.py:142-161.
- Both strip backticks with `getText()[1:-1]` (act-backend/act_backend/spec_checks.py:211, :219).
- Both special-case `d0` by name (act-backend/act_backend/instruction_info.py:71; act-oracle/taidl_to/api_generator.py:23).
- Each generator derives shapes, element types, and folded integers itself; taidl checks none of them (taidl/taidl/operations.py:4).
