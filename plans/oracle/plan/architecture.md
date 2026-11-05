# act-oracle architecture and module layout

TAIDL-TO turns each instruction's semantics into a Python function that, at kernel compile time, emits StableHLO text for one jax-compiled module.
Today that path runs through four layers: the ANTLR visitor, string templates in `taidl_to/templates/templates.txt`, generated `api.py`, and the 989-line `taidl_to/generic/utils.py` that the generated code calls.
This plan keeps StableHLO text as the lowering target and removes one layer and one duplicate of the language, in structural steps before behavioral ones.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O1 | Split `generic/utils.py` by role and rename `generic/` to `runtime/` | T | M | Q1 |
| O2 | One line record from the visitor, consumed by checks and emission | O1, O5 | M | Q1 |
| O3 | Replace string templates by emitter calls | O2, O4 | L | Q2 |
| O18 | Drop the hand-written line splitter for generated code | O1 | S | Q1 |

## O1 Split utils.py and rename generic/

- Breaking: no.
- Changes: `taidl_to/generic/utils.py`, `taidl_to/generic/`; preserves every function and the oracle directory layout.
- What: move `taidl_to/generic/utils.py` into four files by role: loop values (`LoopValue`, `iterations`, `evaluate`, lines 15-300), call checks (`check_*`, lines 651-806), StableHLO emitters (`sh_*`, `mlir_*`, lines 304-650 and 807-850), kernel I/O (`kernel_prologue`, `kernel_epilogue`, packing, lines 851-989).
- What: rename `taidl_to/generic/` to `taidl_to/runtime/`, since `generator.py:52-63` copies it into every oracle as the code the oracle runs.
- Benefits: each file under about 300 lines; a reader finds the emitter for a word without scrolling past loop algebra.
- Caveats: the generated `api.py` imports names from `utils`; every template import line changes in the same commit.
- Caveats: pure moves only, no renames of functions, so the diff is reviewable as moves.
- Fail points: a missed import fails at kernel compile time, not at generation; `tests/test_examples.py` and `tests/test_kernel.py` compile kernels and would fail loudly.
- Fail points: `reserved_names` at `taidl_to/api_generator.py:61-86` reads names from the generated head; a new import name can reserve a new instruction name silently, so diff the reserved set before and after.
- Order and dependencies: first item of the quarter; needs a green suite from T.
- Verification: full suite passes; `git diff -M --stat` shows moves; reserved name set unchanged.

## O2 One line record

- Breaking: no.
- Changes: `taidl_to/oracle_visitor.py`, `taidl_to/semantic_generator.py`; preserves the traversal, attribute forms and generated text.
- What: `OracleVisitor.visitInstruction` (`taidl_to/oracle_visitor.py:223-302`) builds a record per line: word, result name, type, dims as integers or expressions, operands, attributes in both held and per-call forms.
- What: the type check, the generation-time shape check and the call-time shape check all read the record, so attributes are visited once, not twice (`oracle_visitor.py:266-274`), and the `holding` flag goes.
- Benefits: removes hidden visitor state; one place states what a line is; the record is what a NumPy reference interpreter (O13) would also consume.
- Caveats: the record must keep both the held and the per-call Python text of every expression, which is the subtle part of loop handling.
- Fail points: a loop expression held where it must vary gives a silently constant value; `tests/test_kernel.py` loop cases and the unroll tests must cover each attribute form (value, brace list, slice list).
- Order and dependencies: after O1 and O5.
- Verification: the generated `api.py` for both examples is byte-identical before and after.

## O3 Emitter calls instead of string templates

- Breaking: no.
- Changes: `taidl_to/templates/templates.txt`, `taidl_to/template.py`, `taidl_to/generic/utils.py`; preserves the `API_FILE` and `API_FUNC` skeletons and every `sh_*` emitter.
- What: the generated `api.py` today interleaves template text from `taidl_to/templates/templates.txt` with Python that calls `sh_*`; replace each template with one generated call `emit(word, record, ...)` into the runtime.
- What: `taidl_to/template.py` (156 lines) and most of `templates.txt` go; the API function skeleton stays as one template.
- Benefits: the code that emits StableHLO lives in Python files that ruff and tests see, not in a text file of fragments; one table maps a word to its emitter (O4).
- Caveats: generated `api.py` gets shorter but more opaque to a reader of one instruction; keep the per-instruction function so a traceback still names the instruction.
- Fail points: compile time grows if per-call dispatch is slower; `k('fsim-compile')` already returns its compile time, so compare it on `examples/mini_trn_double`.
- Fail points: the reserved-name logic at `api_generator.py:61-86` depends on the generated text; recompute and diff the reserved set.
- Order and dependencies: after O2 and O4; B track may reuse the record if it adopts the same layout.
- Verification: suite green; attention goldens bit-exact on CPU; compile time within 10 percent.

## O18 Drop the line splitter

- Breaking: no.
- Changes: `taidl_to/template.py`; preserves `generate_code`, `indent_code`, `dict_text`.
- What: `layout` and its helpers (`taidl_to/template.py:61-156`) re-implement ruff's line splitting so the generated `api.py` reads as formatted code.
- What: delete them and the two calls (`taidl_to/api_generator.py:35`, `taidl_to/instruction_codegen.py:197`); generated code keeps its template line breaks.
- Benefits: about 95 lines fewer; no second formatter to keep in step with ruff.
- Caveats: a reader of a generated `api.py` sees long lines; the owner may value formatted output for debugging.
- Fail points: `tests/test_templates.py` holds expected layouts (line 509 names ruff's layout); those cases are deleted with the feature, not silently loosened.
- Order and dependencies: after O1; independent of O3, which removes most generated text anyway.
- Verification: suite green; generated `api.py` differs from before only in line breaks (`diff -w` after joining continuation lines).

