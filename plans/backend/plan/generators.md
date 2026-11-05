# Python generators and templates

The Python side is 12 generator modules filling 79 text templates (`act_backend/templates/`).
Many templates exist in several layouts (`TWINS`, `template_loader.py:14`), chosen by line width so the output already passes rustfmt and clang-format.
No gate checks the generated crate's layout: `tests/lint.py:19,27` runs clippy and `-Werror`, and the CI format check covers `git ls-files` only (`README.md:110`).
Each emitter re-walks the TAIDL ANTLR tree (`isel_ir2isa_rewrites_txt_generator.py:56-65`), and `spec_checks.py` imports emitter internals (`spec_checks.py:9-18`).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B12 | Emit plain code without layout twins; rustfmt at generation | B6 | S | Q2 |
| B13 | One selectable-instruction model built from the front end's objects | B1, U2 | M | Q1 |
| B14 | Checks as one pass producing that model | B13, B21 | M | Q1 |

## B12 Emit plain code without layout twins

- What: one template per fragment for the emitters that remain after B6; no width logic (Q4.12).
- What: rustfmt runs on the generated crate at generation time (owner decision 29).
- Preserves: each fragment's text; `tests/lint.sh` as the gate.
- Benefits: removes the twins and the width logic (`fits`, `packed`, `CALL_WIDTH`, `isel_ir2isa_rewrites_rs_generator.py:369-399`); generated crates read the same as hand-written Rust.
- Caveats: rustfmt becomes a requirement of the generation path; the rustfmt test of the twins (`tests/test_generators.py:166-185`) goes with them.
- Fail points: a missing rustfmt on a user's machine; noticed by a generation test that fails with a named error.
- Order and dependencies: after B6, which deletes much of what the templates emit.
- Verification: `tests/lint.sh` passes; template count drops; B3 gate identical.

## B13 One selectable-instruction model

- What: one dataclass per selectable instruction holding inputs, output, attributes, pattern, payload values and conditions, built from the front end's semantics objects, not ANTLR contexts.
- What: B13 builds from today's `instruction.py` objects under U2's documented contract; it reads `validate()`'s derived facts once U3 publishes them (Q3.5).
- Preserves: each emitter's output; `expression_writer.py` and `operand_sizes.py`.
- Benefits: emitters stop re-visiting ANTLR trees; one place says what the backend knows of an instruction.
- Caveats: the model holds no allocator struct; B2's Rust struct is a separate schema (Q4.11).
- Fail points: a field computed differently than before; noticed by generated-crate diffs.
- Order and dependencies: after B1 and U2 (Q3.3); re-pointing at a later front-end model is small.
- Verification: generated crates for both examples identical.

## B14 Checks as one pass producing the model

- What: `spec_checks.py` builds the B13 model and rejects while building, after `validate()` has run (B21); no check imports an emitter.
- What: a check the backend keeps is a backend limit, listed in the coverage doc; a language rule is `validate()`'s (Q3.2, Q3.11).
- Preserves: every backend-limit rejection, its order and its message.
- Benefits: a rejection and the code it guards cannot drift; each coverage row maps to one function.
- Caveats: rejection order and messages are pinned by `tests/test_refusals.py`.
- Fail points: a rejection lost in the move; noticed by `tests/test_refusals.py`.
- Order and dependencies: after B13; before B15.
- Verification: `tests/test_refusals.py` passes; no check-emitter import cycle.
