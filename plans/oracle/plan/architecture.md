# act-oracle architecture and module layout

TAIDL-TO turns each instruction's semantics into a Python function that, at kernel compile time, emits StableHLO text for one jax-compiled module.
Today that path runs through four layers: the ANTLR visitor, string templates in `taidl_to/templates/templates.txt`, generated `api.py`, and the 989-line `taidl_to/generic/utils.py`.
This plan keeps StableHLO text as the lowering target, unrolls every loop (Q5.1), and moves fixed runtime code out of templates into Python.
The oracle stays on its Python and jax path; the later move to fsim-lib (owner decision 23) is a study item, O29.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O19 | Spike: unroll every loop on a synthetic stress kernel against the Q5.1 bound | O21 | S | Q1 |
| O20 | Unroll every loop; delete loop algebra, `Unroll`, `while` emission, `held`, `fixed` | O19, O8 | M | Q1 |
| O1 | Runtime code out of `API_FILE` into `runtime/`; split what remains of `utils.py` | O20 | M | Q1 |
| O2 | One instruction record from the visitor, consumed by checks and emission | O1, O5 | M | Q1 |
| O3 | Emitter calls in generic form instead of string templates; one word table | O2 | M | Q2 |
| O18 | Drop the hand-written line splitter for generated code | none | S | Q1 |
| O28 | Build the C++ debug handler once per install and jax version | none | S | Q2 |
| O29 | Study: migration plan of the oracle onto fsim-lib | none | M | Q2 |

## O19 Unroll spike

- Breaking: no.
- Changes: a branch only; nothing lands.
- What: force `loop['unrolled'] = True` (`taidl_to/templates/templates.txt:82`) and measure kernel compile time with `k('fsim-compile')`.
- What: the fail point is a synthetic stress kernel with a stated trip count and instruction count, a compile-time limit on the CI image, and a per-instruction op-count cap.
- What: the Q5.1 resolution rows set the four numbers:
  - stress kernel: two nested loops of 64 by 64 trips around 4 instructions;
  - trace plus XLA compile of it within 60 s on the CI CPU image;
  - every existing loop test within 2x its current `while` time;
  - an op-count cap of 2x the largest current per-instruction count, above which the oracle rejects with a coded coverage limit.
- What: also run the examples and the attention tests, though no file under `examples/` calls `start_loop`.
- Benefits: decides O20 on a measurement; unrolling is already the path for any loop around a state update or a varying shape (`templates.txt:101-104`, `templates.txt:137-140`).
- Caveats: module size grows with the trip count; that is the one real cost.
- Fail points: a bound agreed after the measurement is no bound.
- Order and dependencies: after O21, so excess precision is fixed; the 2x baselines are measured before the spike.
- Verification: compile time and op count of the stress kernel against the bound; a table per example and attention test.

## O20 Unroll every loop

- Breaking: no.
- Changes: `taidl_to/generic/utils.py`, `taidl_to/templates/templates.txt`, `taidl_to/expression_writer.py`; preserves every check and its message.
- What: a loop variable is a plain Python `int` at each call, so every start, size and expression is a concrete integer.
- What: delete the symbolic loop algebra (`generic/utils.py:15-260`), the `Unroll` retrace (`templates.txt:107-125`, `utils.py:750-761`), `stablehlo.while` emission (`templates.txt:177-237`) and `held` and `fixed` (`expression_writer.py:336-339`).
- What: a kernel past the op-count cap raises `NotImplementedError`, a coverage limit listed in `docs/coverage.md`, not a spec error.
- Benefits: about 400 lines fewer; D11, D17, D19 and D20 go by construction; every addressing value is checkable exactly (O24).
- Caveats: only if O19 is within the bound; otherwise O11, the fallback, lands.
- Fail points: an expression held where it must vary; the loop cases of `tests/test_kernel.py` fail loudly.
- Order and dependencies: after O19, and after O8 so the output baselines are fixed (Q8.10).
- Verification: suite green; attention goldens bit-exact on CPU.

## O1 Runtime code into runtime/

- Breaking: no.
- Changes: `taidl_to/templates/templates.txt`, `taidl_to/generic/`; preserves every function and the oracle directory layout.
- What: move the fixed runtime code of `API_FILE` (`templates.txt:26-293`, about 260 lines) into `runtime/trace.py` as plain Python that ruff and tests see.
- What: rename `taidl_to/generic/` to `taidl_to/runtime/`, since `generator.py:59-64` copies it into every oracle as the code the oracle runs.
- What: split what remains of `utils.py` by role: call checks, StableHLO emitters, kernel I/O.
- Benefits: generated `api.py` holds per-ISA data only; each file under about 300 lines.
- Caveats: pure moves, no renames of functions, so the diff reviews as moves.
- Fail points: `reserved_names` (`taidl_to/api_generator.py:61-86`) reads names from the generated head; diff the reserved set before and after.
- Order and dependencies: after O20, so no deleted code is moved (Q5.9).
- Verification: suite green; `api.py` equal but its head; reserved set unchanged.

## O2 One instruction record

- Breaking: no.
- Changes: `taidl_to/oracle_visitor.py`, `taidl_to/semantic_generator.py`, `taidl_to/instruction_codegen.py`; preserves the traversal and generated text.
- What: the visitor builds one record per instruction from the `Model` (U10): regions, constraints, updates, and per line the word, result, type, dims, operands and attributes.
- What: type checks, shape checks and emission read the record, so attributes are visited once (`oracle_visitor.py:266-274`) and the `holding` flag goes.
- What: measure kernel compile time before and after; the figure decides O3.
- Benefits: removes hidden visitor state; the record is what O3's runtime `call` and O13's reference consume.
- Fail points: a value fixed at generation where a per-call one is needed; loop tests in `tests/test_kernel.py`.
- Order and dependencies: after O1 and O5; absorbs O5's double-visit part; the tree walk it replaces is deleted with the string surface (U14).
- Verification: generated `api.py` byte-identical for both examples.

## O3 Emitter calls, generic form, one word table

- Breaking: no.
- Changes: `taidl_to/templates/templates.txt`, `taidl_to/template.py`, `taidl_to/oracle_visitor.py`, `runtime/`; preserves one function per instruction in `api.py`.
- What: each generated function passes its arguments and its record to one runtime `call`; per-word templates go.
- What: one emitter writes MLIR generic form for every word; `sh_pretty`, `sh_cast`, `sh_transpose`, `sh_broadcast` and `sh_iota` restate one pattern (`generic/utils.py:477-571`); only `reduce`, `dot` and `constant` keep their own function.
- What: O4's word table lands here as the lowering columns (op name, dialect, attribute mapper, reducer, the guards of O22); the language columns go to the front end with O7.
- What: lowering stays StableHLO text; a `jax.lax` path is rejected (Q5.9).
- What: this generic emitter is the one source of per-instruction StableHLO; O34 and the backend dialect (B50) both read it (Q12.4).
- Benefits: removes the template layer and `PRETTY` and `op_to_template` (`oracle_visitor.py:100`, `oracle_visitor.py:208-221`); fixes D16 by deletion.
- Caveats: a traceback must still name the instruction, so the per-instruction function stays.
- Fail points: a row with the wrong dialect fails at MLIR parse naming no instruction; `tests/test_elementwise.py` and `tests/test_linalg.py` run every word.
- Order and dependencies: after O2, and only if O2's measured compile time needs it; `PRETTY` stays until then (Q5.9).
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

## O28 Debug handler built once

- Breaking: no.
- Changes: `taidl_to/generator.py`, `xla-debug/`; preserves the handler and its output.
- What: the `cmake` call (`generator.py:72-84`) takes no ISA input, so build the handler once per install and jax version and reuse it in every oracle.
- Benefits: generation no longer runs a C++ build; no fidelity is lost (Q5.9).
- Fail points: a stale handler after a jax bump; key the build by the jax version and fail loudly on a mismatch.
- Order and dependencies: independent; after O1 if `generic/` has moved.
- Verification: two generations reuse one build; a jax version change rebuilds it.

## O29 fsim-lib migration study

- Breaking: no.
- Changes: a planning doc only; nothing in act-oracle changes.
- What: owner decision 23 moves the oracle's execution to an all-C/C++ simulator through fsim-lib later; this item writes that migration plan.
- What: start from the follow-up analysis in this track's `fsim/` folder: interface fit, what each O item carries over, the native path, and the fidelity risks per owner decision.
- What: the plan is O34 to O36 (`fsim.md`), native-path steps 1 to 4 on U15's pin set; step 5 stays out under owner decision 28 (Q12.4).
- What: name the gates a migration must pass: every golden and norm row of this plan holds on fsim, and each fsim pass keeps decisions 16, 17, 22 and 24.
- Benefits: the next plan starts from measured fit, not a fresh survey.
- Caveats: only O34 waits on it, and it reorders nothing on the current path.
- Order and dependencies: none; it runs after the current path's Q2 items and reads their results.
- Verification: a migration plan with items, dependencies and a fail point per fidelity risk.
