# act-oracle review

The oracle as it stands on branch `claude/clever-fermat-1442fd`, read this session without running it.
Most of it is sound: generation is atomic, every rejection names the instruction, and integer arithmetic is exact.
The weak parts are duplicated language rules, a string-template layer, a hand-written formatter, and a few silent numeric deviations (D1, D11).

## What works and should be kept

- Atomic generation: the previous oracle is removed first, the new one assembled in a staging directory and moved into place only when built (`taidl_to/generator.py:32-34`, `generator.py:57-87`).
- Reserved names derived from the generated text, not hand-listed: `reserved_names` reads `api.py`'s symbol tables and the decorator's `api.` attributes (`taidl_to/api_generator.py:61-86`).
- Every rejection names the instruction, and a call-time one names the failing loop iteration (`taidl_to/generic/utils.py:704-712`, `utils.py:796-804`).
- Exact integer expressions: Python integers, floor `/` mapped to `//` (`taidl_to/expression_writer.py:20-21`), Python `%` taking the divisor's sign as S6.2 requires, and a `UserWarning` past 64 bits (`utils.py:268-271`).
- Symbolic loop values with exact quotients, so a start over a loop variable is checked at every iteration (`utils.py:18-150`).
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
- Process-wide jax configuration (`decorator.py:22-23`).
- Float literals rounded twice (`oracle_visitor.py:383`, `utils.py:408-409`).

## Verdict per component

| Component | Lines | Verdict | Reason |
| --- | --- | --- | --- |
| `generator.py` | 89 | keep | Atomic, short; only its log lines change (O16). |
| `api_generator.py` | 86 | keep | Derived reserved names are the right design; `d0` skip at line 22 stays. |
| `semantic_generator.py` | 63 | merge | Its region type and element-count checks move beside the region code in `instruction_codegen.py`; the visitor call stays. |
| `instruction_codegen.py` | 224 | keep, reshape | Per-instruction function, constraint and size checks are sound; takes the checks from `semantic_generator.py`. |
| `oracle_visitor.py` | 459 | reshape | One word table (O4), one line record (O2), no double visit (O5), S8.3 from the front end (O6). |
| `expression_writer.py` | 111 | keep | Small and faithful to S6. |
| `template.py` | 156 | reshape | Keep `generate_code`, `indent_code`, `dict_text`; drop `layout` and its splitter (O18). |
| `templates/templates.txt` | 470 | reshape | Keep the API function and module skeletons; per-word templates become emitter calls (O3). |
| `generic/utils.py` | 989 | reshape | Split by role into four files (O1); contents kept. |
| `generic/shapes.py` | 543 | reshape | Rules are correct and needed; literal-shape rules move to the front end (O7), call-time entry stays. |
| `generic/decorator.py` | 217 | keep, reshape | Kernel API and checks stay; jax configuration scoped (O12). |
| `xla-debug/` | 170 | keep | Small FFI print handler, built per generation. |
| `docs/` and `README.md` | 325 | reshape | `docs/coverage.md` added; `docs/operations.md` folded into it (O14). |
