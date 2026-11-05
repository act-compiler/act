# Object model

This page is S4.6 of the spec: the Python objects a generator reads from an accepted accelerator.
A generator reads only the attributes below; any other attribute is internal to the front end.
Names are those of the reference front end; identifier renames beyond owner decision 19 are pending panel.

## S4.6 Objects and attributes

### Accelerator

| attribute | type | holds |
| --- | --- | --- |
| `name` | `str` | the accelerator's name |
| `base_dir` | `str` | where generators write (S4.2) |
| `data_model` | `list` of buffers | every buffer in declaration order; entry 0 is `d0` |
| `state` | `list` of state variables | every state variable in declaration order |
| `instructions` | `list` of instructions | every instruction in declaration order |

### Buffer

| attribute | type | holds |
| --- | --- | --- |
| `var_name` | `str` | the buffer's name |
| `access_dim` | `list[int]` | the access dimensions; `[]` for `d0` until plan item L30 lands |
| `unit_dim` | `list[int]` | the unit shape |
| `var_dim` | `list[int]` | the full shape: `access_dim` followed by `unit_dim` |
| `var_type` | `str` | the element type word of S7.3 |

### Instruction

| attribute | type | holds |
| --- | --- | --- |
| `instruction` | `str` | the instruction's name |
| `comp_attr` | `list[str]` | the computational attributes, in order |
| `parameters` | `list[str]` | the addressing attributes, in order |
| `constraints`, `constraint_asts` | `list[str]`, `list` of trees | every constraint, including those placement marks append, and its tree |
| `update`, `update_asts` | `list[str]`, `list` of trees | every state update in order, and its tree |
| `instr_inputs`, `instr_outputs` | `list` of regions | each region as given: `[name, starts, sizes]`, a start or size an `int` or `str` |
| `instr_input_asts`, `instr_output_asts` | `list` | each region as `[name, start trees, size trees]`; an `int` is parsed from its `str` |
| `instr_semantics_ast` | tree or `None` | the `module` tree of the semantics, `None` for a state-only instruction |
| `instr_expression_asts` | `dict` | each `EXPRESSION` token node of the semantics tree, by identity, to its tree |
| `inplace` | `list[bool]` | one in-place flag per input region (S5.5) |

## Trees

- The front end parses each text when the call that takes it runs, and keeps its tree beside the text.
- A tree is the ANTLR context of the entry rule's first child: `expression` for a start, a size, or a backtick expression; `condition` for a constraint; `update` for a state update.
- Backtick expressions cover shape dimensions, `constant` operands, and operation attribute values; each is looked up by its token node.
- An operation attribute's name is read by its text, never by its token type, since it may lex as `OPERATION` (S3.1 rule `name`).
- `parameter(k)` reads k from its operand's text; a `constant` literal is its operand's text.
