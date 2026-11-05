# act-oracle target structure

The target is derived from today's `taidl_to/` package: no module is rewritten from scratch.
Every structural step leaves the generated `api.py` byte-identical unless it states a behavioral fix.
Lowering stays StableHLO text; the change is no loop algebra, no runtime code as template text, and no copy of language rules.

## Per module

| Today | Target | Action | What is preserved |
| --- | --- | --- | --- |
| `generator.py` | `generator.py` | keep | All of it; log lines reworded (O16); shape module copied and stamped (O7); debug handler reused (O28). |
| `api_generator.py` | `api_generator.py` | keep | All of it, including `reserved_names`. |
| `semantic_generator.py` | visitor call only | reshape | Region checks run in the front end (O7). |
| `instruction_codegen.py` | `instruction_codegen.py` | reshape | One function per instruction passing its record to `runtime` (O2, O3); updates after outputs (O25); constraint list checked before any load (O10). |
| `oracle_visitor.py` | `oracle_visitor.py` | reshape | Traversal and attribute forms; one instruction record (O2); S8.3 and static checks from the front end (O6, O7). |
| `expression_writer.py` | `expression_writer.py` | keep | All but `held` and `fixed` (O20). |
| `template.py` | `template.py` | reshape | `generate_code`, `indent_code`, `dict_text`; `layout` removed (O18). |
| `templates/templates.txt` | API function skeleton | reshape | Runtime code to `runtime/trace.py` (O1); per-word templates to emitter calls (O3); `while` emission removed (O20); zero fill removed (O26). |
| `generic/` | `runtime/` | rename | Copied into each oracle as today (O1). |
| `generic/utils.py` | `runtime/checks.py`, `runtime/emit.py`, `runtime/kernel_io.py` | split | Loop algebra deleted (O20); the rest moved by role (O1). |
| `generic/shapes.py` | the front end's shape module, copied at generation | replace | Each rule, moved by U4; call-time entry `disagreement` (O7). |
| `generic/decorator.py` | `runtime/decorator.py` | keep | Kernel API, declaration checks, platform choice; XLA flags per compile (O21); jax configuration dropped (O12). |
| `xla-debug/` | `xla-debug/` | keep | All of it; built once per install and jax version (O28). |
| `docs/operations.md` | folded into `docs/coverage.md` | merge | Oracle-only rows and the test bounds (O14). |

## Ordered steps

Fidelity fixes first, then loop deletion before any split, then structure, then the remaining deltas (Q5.9).

1. O18, O5, O6 class change: splitter and dead branches go; output differs only in line breaks.
2. O8, O21, O27, O22, O23, O25, O10: literals, XLA flags, `f32` widening, norm rows, fixed data values, pre-state reads, overlap constraints; the new behavioral baseline.
3. O19: unroll spike against the bound the owner sets first.
4. O20 if the spike passes, else O11: one of the two lands.
5. O1: runtime code into `runtime/`, `utils.py` split; output identical but import lines.
6. O24, O32, O31, O26, O9: call-time errors, width layer as constraints, failure-policy audit, unwritten reads, `f16` solve.
7. O2: instruction record; output identical; compile time measured.
8. O16: log lines.
9. O3, only if O2's compile time needs it: emitter calls in generic form; module text differs only in form.
10. O33: neutrality test for the encoding layer and generator metadata; the binary decoder joins when it exists.
11. O7, O12, O13, O30, O28: front-end checks after U4, jax configuration, differential reference, overlap test matrix, debug handler.
12. O14, O17, O15: docs, coverage test, names.
13. O29: fsim-lib migration study; no step above waits on it.

- O10 follows O25: it checks the front end's constraint list before any load and computes from the pre-state.
- O23 follows O22: it adds the fixed-value guards for divide, remainder and float-to-integer convert.
