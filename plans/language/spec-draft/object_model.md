# Object model

This page is S4.6 of the spec: the Python names a generator reads from an accepted accelerator that the language fixes.
Every other attribute, with its type, is documented in the `taidl` README, not here.
The typed contract model, its `CONTRACT_VERSION`, and the derived facts `validate()` attaches are shared plan items U2 and U3.

## S4.6 Fixed names

| attribute | type | holds |
| --- | --- | --- |
| `constraint_asts` | `list` of trees | the tree of every constraint, including those a mark expands to |
| `update_asts` | `list` of trees | the tree of every state update, in order |
| `instr_input_asts` | `list` | each input region as `[name, start trees, size trees]`; an `int` is parsed from its `str` |
| `instr_output_asts` | `list` | each output region in the same form |
| `instr_expression_asts` | `dict` | each `EXPRESSION` token node of the semantics tree, by identity, to its tree |

- Plan item L29 tests that each name exists on an accepted accelerator with the stated type.
- The relation of each same-buffer input-output pair and the in-place tie are recorded for the backend (S5.5); plan item L35 fixes their names.

## Trees

- The front end parses each text when the call that takes it runs, and keeps its tree beside the text.
- A tree is the ANTLR context of the entry rule's first child: `expression` for a start, a size, or a backtick expression; `condition` for a constraint; `update` for a state update.
- ANTLR trees are reachable but unstable: their shape may change in any release.
- Backtick expressions cover shape dimensions, `constant` operands, and operation attribute values; each is looked up by its token node.
- An operation attribute's name is read by its text, never by its token type, since it may lex as `OPERATOR` (S3.1 rule `name`).
- `parameter(k)` reads k from its operand's text; a `constant` literal is its operand's text.
