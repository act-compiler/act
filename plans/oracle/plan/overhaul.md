# act-oracle target structure

The target is derived from today's `taidl_to/` package: no module is rewritten from scratch, and every step leaves the generated `api.py` byte-identical unless it is a stated behavioral fix.
Lowering stays StableHLO text; the change is fewer layers and no copy of language rules.

## Per module

| Today | Target | Action | What is preserved |
| --- | --- | --- | --- |
| `generator.py` | `generator.py` | keep | All of it; log lines reworded (O16). |
| `api_generator.py` | `api_generator.py` | keep | All of it, including `reserved_names`. |
| `semantic_generator.py` | merged into `instruction_codegen.py` | merge | Its region checks and the visitor call, moved unchanged; `TRACE_NAMES` moves with them. |
| `instruction_codegen.py` | `instruction_codegen.py` | reshape | Function skeleton, constraint, update, size, input and output generation. |
| `oracle_visitor.py` | `oracle_visitor.py` plus `words.py` | reshape | The visitor's traversal and attribute forms; six tables become one in `words.py` (O4); line record (O2). |
| `expression_writer.py` | `expression_writer.py` | keep | All of it. |
| `template.py` | `template.py` | reshape | `generate_code`, `indent_code`, `dict_text`; `layout` removed (O18). |
| `templates/templates.txt` | `templates/templates.txt` | reshape | `API_FILE`, `API_FUNC`, module state and check templates; per-word templates replaced by emitter calls (O3). |
| `generic/` | `runtime/` | rename | Directory contents, copied into each oracle as today (O1). |
| `generic/utils.py` | `runtime/loops.py`, `runtime/checks.py`, `runtime/emit.py`, `runtime/kernel_io.py` | split | Every function, moved by role (O1). |
| `generic/shapes.py` | `runtime/shapes.py`, rules imported from the front end | reshape | Call-time entry point `disagreement`; literal-shape rules move to the front end (O7). |
| `generic/decorator.py` | `runtime/decorator.py` | keep | Kernel API, declaration checks, platform choice; jax configuration scoped (O12). |
| `xla-debug/` | `xla-debug/` | keep | All of it. |
| `docs/operations.md` | folded into `docs/coverage.md` | merge | Oracle-only rows: CHLO words, `copy`, `parameter`, `reduce_*`, 2-bit `dot`, test bounds (O14). |

## Ordered steps

Structural prep first, then behavioral deltas.

1. O5: delete the unreachable branch and the double attribute visit in `oracle_visitor.py`; output identical.
2. O1: split `generic/utils.py` and rename `generic/` to `runtime/`; output identical but import lines.
3. O18: drop `layout` from `template.py`; output differs only in line breaks.
4. O4: one word table in `words.py`; output identical.
5. O8, O11: literal rounding and wrapped starts; output changes only for affected specs.
6. O6: spec errors raise `ValueError`; S8.3 from the front end once L3 lands.
7. O14, O17, O15, O16: docs, coverage test, names, log lines.
8. O2, O3: one line record, per-word templates replaced by emitter calls.
9. O7, O13, O9, O10, O12: front-end shape rules, reference interpreter, remaining fidelity items.
