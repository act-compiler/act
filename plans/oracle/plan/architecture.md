# act-oracle architecture and module layout

TAIDL-TO turns each instruction's semantics into a Python function that, at kernel compile time, emits StableHLO text for one jax-compiled module.
Today that path runs through four layers: the ANTLR visitor, string templates in `taidl_to/templates/templates.txt`, generated `api.py`, and the 989-line `taidl_to/generic/utils.py`.
This plan keeps StableHLO text as the lowering target, unrolls every loop if the spike allows, and moves fixed runtime code out of templates into Python.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O19 | Spike: unroll every loop at trace time, measure compile time | O8, O21 | S | Q1 |
| O20 | Unroll every loop; delete loop algebra, `Unroll`, `while` emission | O19 | M | Q1 |
| O1 | Runtime code out of `API_FILE` into `runtime/`; split what remains of `utils.py` | O20 | M | Q1 |
| O2 | One instruction record from the visitor, consumed by checks and emission | O1, O5 | M | Q1 |
| O3 | Emitter calls in generic form instead of string templates; one word table | O2 | M | Q2 |
| O18 | Drop the hand-written line splitter for generated code | none | S | Q1 |

## O19 Unroll spike

- Breaking: no.
- Changes: a branch only; nothing lands.
- What: force `loop['unrolled'] = True` (`taidl_to/templates/templates.txt:82`) and measure kernel compile time with `k('fsim-compile')` on the examples and the attention tests.
- Benefits: decides O20 and O11 on a measurement; unrolling is already the path for any loop around a state update or a varying shape (`templates.txt:101-104`, `templates.txt:137-140`).
- Caveats: module size grows with the trip count; that is the one real cost.
- Fail points: a bound agreed after the measurement is no bound; the owner fixes it before the spike runs.
- Order and dependencies: after O8 and O21, so the baseline outputs are the fixed ones.
- Verification: a table of compile times, forced against today, per example and attention test.

## O20 Unroll every loop

- Breaking: no.
- Changes: `taidl_to/generic/utils.py`, `taidl_to/templates/templates.txt`, `taidl_to/expression_writer.py`; preserves every check and its message.
- What: a loop variable is a plain Python `int` at each call, so every start, size and expression is a concrete integer.
- What: delete the symbolic loop algebra (`generic/utils.py:15-260`), the `Unroll` retrace (`templates.txt:107-125`, `utils.py:750-761`), `stablehlo.while` emission (`templates.txt:177-237`) and `held` and `fixed` (`expression_writer.py:336-339`).
- Benefits: about 400 lines fewer; D11, D17, D19 and D20 go by construction; every addressing value is checkable exactly (O24).
- Caveats: only if O19 is within the bound; otherwise O11 stays the plan.
- Fail points: an expression held where it must vary; the loop cases of `tests/test_kernel.py` fail loudly.
- Order and dependencies: after O19.
- Verification: suite green; attention goldens bit-exact on CPU.

## O1 Runtime code into runtime/

- Breaking: no.
- Changes: `taidl_to/templates/templates.txt`, `taidl_to/generic/`; preserves every function and the oracle directory layout.
- What: move the fixed runtime code of `API_FILE` (`templates.txt:26-293`, about 260 lines) into `runtime/trace.py` as plain Python that ruff and tests see.
- What: rename `taidl_to/generic/` to `taidl_to/runtime/`, since `generator.py:52-63` copies it into every oracle as the code the oracle runs.
- What: split what remains of `utils.py` by role: call checks, StableHLO emitters, kernel I/O.
- Benefits: generated `api.py` holds per-ISA data only; each file under about 300 lines.
- Caveats: pure moves, no renames of functions, so the diff reviews as moves.
- Fail points: `reserved_names` (`taidl_to/api_generator.py:61-86`) reads names from the generated head; diff the reserved set before and after.
- Order and dependencies: after O20, so no deleted code is moved first.
- Verification: suite green; `api.py` equal but its head; reserved set unchanged.

## O2 One instruction record

- Breaking: no.
- Changes: `taidl_to/oracle_visitor.py`, `taidl_to/semantic_generator.py`, `taidl_to/instruction_codegen.py`; preserves the traversal and generated text.
- What: the visitor builds one record per instruction: regions, constraints, updates, and per line the word, result, type, dims, operands and attributes.
- What: type checks, shape checks and emission read the record, so attributes are visited once (`oracle_visitor.py:266-274`) and the `holding` flag goes.
- Benefits: removes hidden visitor state; the record is what O3's runtime `call` and O13's reference consume.
- Caveats: without O20, the record keeps both the held and the per-call text of every expression.
- Fail points: a held value where a per-call one is needed; loop tests in `tests/test_kernel.py`.
- Order and dependencies: after O1 and O5; absorbs O5's double-visit part.
- Verification: generated `api.py` byte-identical for both examples.

## O3 Emitter calls, generic form, one word table

- Breaking: no.
- Changes: `taidl_to/templates/templates.txt`, `taidl_to/template.py`, `taidl_to/oracle_visitor.py`, `runtime/`; preserves one function per instruction in `api.py`.
- What: each generated function passes its arguments and its record to one runtime `call`; per-word templates go.
- What: one emitter writes MLIR generic form for every word; `sh_pretty`, `sh_cast`, `sh_transpose`, `sh_broadcast` and `sh_iota` restate one pattern (`generic/utils.py:477-571`); only `reduce`, `dot` and `constant` keep their own function.
- What: O4's word table lands here as the lowering columns (op name, dialect, attribute mapper, reducer, guard of O22); the language columns go to the front end with O7.
- Benefits: removes the template layer and `PRETTY` and `op_to_template` (`oracle_visitor.py:100`, `oracle_visitor.py:208-221`); fixes D16 by deletion.
- Caveats: a traceback must still name the instruction, so the per-instruction function stays.
- Fail points: a row with the wrong dialect fails at MLIR parse naming no instruction; `tests/test_elementwise.py` and `tests/test_linalg.py` run every word.
- Order and dependencies: after O2; conditional on O2's measured compile time.
- Verification: module text differs only in form; suite green; compile time within 10 percent.

## O18 Drop the line splitter

- Breaking: no.
- Changes: `taidl_to/template.py`, `tests/conftest.py`, `tests/test_templates.py`; preserves `generate_code`, `indent_code`, `dict_text`.
- What: delete `layout` and its helpers (`taidl_to/template.py:61-156`) and its two calls (`api_generator.py:35`, `instruction_codegen.py:197`).
- What: drop the ruff format check on every generated oracle (`tests/conftest.py:278-282`) and the layout cases of `tests/test_templates.py`.
- Benefits: about 95 lines fewer; no second formatter to keep in step with ruff.
- Caveats: generated lines get long; one function per instruction keeps them readable.
- Fail points: a test silently loosened instead of deleted; review the test diff.
- Order and dependencies: independent; first.
- Verification: suite green; `api.py` differs only in line breaks.
