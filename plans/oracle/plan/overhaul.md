# act-oracle target structure

The target is derived from today's `taidl_to/` package: no module is rewritten from scratch.
Every structural step leaves the generated `api.py` byte-identical unless it states a behavioral fix.
Lowering stays StableHLO text; the change is no loop algebra, no runtime code as template text, and no copy of language rules.

## Per module

| Today | Target | Action | What is preserved |
| --- | --- | --- | --- |
| `generator.py` | `generator.py` | keep | All of it; log lines reworded (O16). |
| `api_generator.py` | `api_generator.py` | keep | All of it, including `reserved_names`. |
| `semantic_generator.py` | visitor call only | reshape | Region checks move to the front end (O7). |
| `instruction_codegen.py` | `instruction_codegen.py` | reshape | One function per instruction passing its record to `runtime` (O2, O3); in-place storage reuse (O10). |
| `oracle_visitor.py` | `oracle_visitor.py` | reshape | Traversal and attribute forms; one instruction record (O2); S8.3 and static checks from the front end (O6, O7). |
| `expression_writer.py` | `expression_writer.py` | keep | All but `held` and `fixed` (O20). |
| `template.py` | `template.py` | reshape | `generate_code`, `indent_code`, `dict_text`; `layout` removed (O18). |
| `templates/templates.txt` | API function skeleton | reshape | Runtime code to `runtime/trace.py` (O1); per-word templates to emitter calls (O3); `while` emission removed (O20). |
| `generic/` | `runtime/` | rename | Copied into each oracle as today (O1). |
| `generic/utils.py` | `runtime/checks.py`, `runtime/emit.py`, `runtime/kernel_io.py` | split | Loop algebra deleted (O20); the rest moved by role (O1). |
| `generic/shapes.py` | front-end rules, call-time entry `disagreement` | reshape | Each rule, moved (O7). |
| `generic/decorator.py` | `runtime/decorator.py` | keep | Kernel API, declaration checks, platform choice; jax configuration dropped (O12); excess precision off (O21). |
| `xla-debug/` | `xla-debug/` | keep | All of it. |
| `docs/operations.md` | folded into `docs/coverage.md` | merge | Oracle-only rows and the test bounds (O14). |

## Ordered steps

Fidelity fixes first, then the spike, structure, and the remaining behavioral deltas.

1. O18, O5: drop the splitter and the dead branches; output differs only in line breaks.
2. O8, O21: literals rounded once, no excess precision; the new behavioral baseline.
3. O19: unroll spike; the owner fixes the compile-time bound first.
4. O20 if the spike passes, else O11: one of the two lands.
5. O1: runtime code into `runtime/`, `utils.py` split; output identical but import lines.
6. O24, O10, O9: call-time errors, in-place storage reuse, `f16` solve.
7. O2: instruction record; output identical.
8. O6, O16: error classes, log lines.
9. O3: emitter calls in generic form with the lowering table; module text differs only in form.
10. O7, O12, O22, O23, O13: front-end checks, jax configuration, norms, in-module errors, differential reference.
11. O14, O17, O15: docs, coverage test, names.
