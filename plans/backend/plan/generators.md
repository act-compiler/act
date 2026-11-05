# Python generators and templates

The Python side is 12 generator modules filling 79 text templates (`act_backend/templates/`).
Many templates exist in several layouts (`TWINS`, `template_loader.py:14`), chosen by line width so the output already passes rustfmt and clang-format.
No gate checks the generated crate's layout: `tests/lint.py:19,27` runs clippy and `-Werror`, and the CI format check covers `git ls-files` only (`README.md:110`).
Each emitter re-walks the TAIDL ANTLR tree (`isel_ir2isa_rewrites_txt_generator.py:56-65`), and `spec_checks.py` imports emitter internals (`spec_checks.py:9-18`).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B12 | Emit plain code without layout twins | B6 | S | Q2 |
| B13 | One selectable-instruction model built from the front end's objects | B1, L | M | Q1 |
| B14 | Checks as one pass producing that model | B13 | M | Q1 |

## B12 Emit plain code without layout twins

- What: one template per fragment for the emitters that remain after B6; no width logic and no formatter run at generation.
- Preserves: each fragment's text; `tests/lint.sh` as the gate.
- Benefits: removes the twins and the width logic (`fits`, `packed`, `CALL_WIDTH`, `isel_ir2isa_rewrites_rs_generator.py:369-399`).
- Caveats: generated code is read only under `--log` or in debugging; a formatter run stays optional.
- Fail points: clippy lints that depend on layout; noticed by `tests/lint.sh`.
- Order and dependencies: after B6, which deletes much of what the templates emit.
- Verification: `tests/lint.sh` passes; template count drops; B3 gate identical.

## B13 One selectable-instruction model

- What: one dataclass per selectable instruction holding inputs, output, attributes, pattern, payload values and conditions, built from the front end's semantics objects, not ANTLR contexts.
- What: the model also emits the B2 struct for Rust and C++.
- Preserves: each emitter's output; `expression_writer.py` and `operand_sizes.py`.
- Benefits: emitters stop re-visiting ANTLR trees; one place says what the backend knows of an instruction.
- Caveats: the front end's object model is a language-track item; until it lands, B13 builds from today's `instruction.py` objects.
- Fail points: a field computed differently than before; noticed by generated-crate diffs.
- Order and dependencies: after B1; joint with the language track.
- Verification: generated crates for both examples identical.

## B14 Checks as one pass producing the model

- What: `spec_checks.py` builds the B13 model and rejects while building; no check imports an emitter.
- Preserves: every rejection, its order and its message.
- Benefits: a rejection and the code it guards cannot drift; each coverage row maps to one function.
- Caveats: rejection order and messages are pinned by `tests/test_refusals.py`.
- Fail points: a rejection lost in the move; noticed by `tests/test_refusals.py`.
- Order and dependencies: after B13; before B15.
- Verification: `tests/test_refusals.py` passes; no check-emitter import cycle.
