# act-oracle review

The oracle as it stands on branch `claude/clever-fermat-1442fd`.
Most of it is sound: generation is atomic, every rejection names the instruction, and integer arithmetic is exact.
The weak parts are duplicated language rules, a string-template layer, a hand-written formatter, and silent numeric deviations (D1, D11, D21, D24), and run-time violations it does not detect (D22, D23).

## What works and should be kept

- Atomic generation: the previous oracle is removed first, the new one assembled in a staging directory and moved into place only when built (`taidl_to/generator.py:32-34`, `generator.py:57-87`).
- Reserved names derived from the generated text, not hand-listed: `reserved_names` reads `api.py`'s symbol tables and the decorator's `api.` attributes (`taidl_to/api_generator.py:61-86`).
- Every rejection names the instruction, and a call-time one names the failing loop iteration (`taidl_to/generic/utils.py:704-712`, `utils.py:796-804`).
- Exact integer expressions: Python integers, floor `/` mapped to `//` (`taidl_to/expression_writer.py:20-21`), Python `%` taking the divisor's sign as S6.2 requires, and a `UserWarning` past 64 bits (`utils.py:268-271`).
- Every call-time check covers every loop iteration a value depends on (`utils.py:68-72`).
- Numerics chosen for fidelity: every dot at `HIGHEST` precision (`utils.py:604`), non-finite literals emitted by their bits (`utils.py:411-420`), reduce identities per type including `f8e4m3fn`'s missing infinity (`utils.py:439-451`).
- StableHLO text as the target, naming each word's reference operation directly, with CHLO where StableHLO lacks one (`taidl_to/oracle_visitor.py:124`, `oracle_visitor.py:330-333`).
- Kernel declaration checks before tracing (`taidl_to/generic/decorator.py:64-140`).
- Tests that hold attention kernels bit-exact to goldens on the CPU (`README.md:163`).

## What is weak

- Language rules copied into the generator: `TYPES` equals S8.3 (`oracle_visitor.py:19-98`) and `shapes.py` checks S7.4, which the front end leaves to generators (S7 and S9).
- Facts of one word spread over six tables (`oracle_visitor.py:19-124`, `oracle_visitor.py:208-221`).
- Three layers to emit one line: visitor, `templates.txt` (470 lines, 24 templates), and `sh_*` emitters in `utils.py`.
- A hand-written ruff-style line splitter for generated code (`taidl_to/template.py:61-156`).
- Hidden visitor state: attributes visited twice under a toggled `holding` flag (`oracle_visitor.py:266-274`).
- Process-wide jax configuration (`decorator.py:22-23`), and no control of XLA's excess precision.
- Fixed runtime code kept as template text in `API_FILE` (`templates/templates.txt:26-293`), unseen by ruff and tests.
- Symbolic loop algebra and `while` emission (`utils.py:15-260`), the source of D11, D17, D19, D20.
- Float literals rounded twice (`oracle_visitor.py:383`, `utils.py:408-409`).

## Verdict per component

| Component | Lines | Verdict | Reason |
| --- | --- | --- | --- |
| `generator.py` | 89 | keep | Atomic, short; only its log lines change (O16). |
| `api_generator.py` | 86 | keep | Derived reserved names are the right design; `d0` skip at line 22 stays. |
| `semantic_generator.py` | 63 | reshape | Its region type and element-count checks move to the front end (O7); the visitor call stays. |
| `instruction_codegen.py` | 224 | keep, reshape | Per-instruction function, constraint and size checks are sound; takes the checks from `semantic_generator.py`. |
| `oracle_visitor.py` | 459 | reshape | One instruction record (O2), lowering table (O3), S8.3 and static checks from the front end (O6, O7). |
| `expression_writer.py` | 111 | keep | Small and faithful to S6. |
| `template.py` | 156 | reshape | Keep `generate_code`, `indent_code`, `dict_text`; drop `layout` and its splitter (O18). |
| `templates/templates.txt` | 470 | reshape | Runtime code moves to `runtime/` (O1); per-word templates become emitter calls (O3); `while` emission goes (O20). |
| `generic/utils.py` | 989 | reshape | Loop algebra deleted (O20); the rest split by role (O1). |
| `generic/shapes.py` | 543 | reshape | Rules are correct and needed; they move to the front end through U4 and come back as a copy at generation (O7); call-time entry stays. |
| `generic/decorator.py` | 217 | keep, reshape | Kernel API and checks stay; jax configuration dropped (O12); XLA flags per compile (O21). |
| `xla-debug/` | 170 | keep | Small FFI print handler, built per generation today; built once per install and jax version (O28). |
| `docs/` and `README.md` | 325 | reshape | `docs/coverage.md` added; `docs/operations.md` folded into it (O14). |
