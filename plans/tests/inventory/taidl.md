# Inventory: taidl

The five test files of the taidl repo, 131 tests, with a verdict each.
Style values are those of the inventory key.

| file | what it tests | style | verdict | reason |
| --- | --- | --- | --- | --- |
| `tests/test_grammar.py` (42) | lexer and parser acceptance and rejection of semantics text, literals, shapes, attributes, operation words | spec, impl | rewrite into the corpus (T7) | `TestWords` at :274-297 iterates grammar words and `TYPES`, which is S2 and S7.3 material; `TestDocs` at :317-350 parses `docs/operations.md` tables by regex and pins their layout |
| `tests/test_instruction.py` (60) | the Python front end: names, regions, in-place flags, overlap, state, semantics attachment | spec | keep, tag (T1) | maps onto S4.1, S5.2, S5.4, S5.5 (in-place and overlap tests seed T30); `test_identifier_accepted` :50 and `test_one_start_and_size_per_addressed_dimension` :222 hold no assertion and pass by not raising |
| `tests/test_expressions.py` (18) | expression grammar, references, constraints, updates, declared names | spec | keep, tag (T1) | S6.1 to S6.3; well parametrized at :24-250 |
| `tests/test_data_model.py` (7) | buffer dimensions and element type names | spec | keep, tag (T1) | S4.3; parametrized over `TYPES` at :53 |
| `tests/test_examples.py` (4) | the two examples load and parse | impl | merge into T10 | pins `IDLV2Parser.ModuleContext` at :92; example lists are hard-coded in `EXAMPLES` |

## Counts

| keep | rewrite | merge | delete |
| --- | --- | --- | --- |
| 85 | 42 | 4 | 0 |
