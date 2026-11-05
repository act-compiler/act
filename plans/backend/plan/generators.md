# Python generators and templates

The Python side is 12 generator modules filling 79 text templates (`act_backend/templates/`).
Many templates exist in several layouts (`TWINS` in `act_backend/template_loader.py:14`: wrapped, aligned, stacked, vertical, stacked_vertical), chosen by line width so the output already passes rustfmt and clang-format (`template_loader.py:8-19`).
`spec_checks.py` re-runs the rewrite generator's visitor to check attributes (`spec_checks.py:213-215`), so checking and emitting share code paths in both directions.
No item here breaks compatibility; sizes assume two engineers.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B12 | Format generated code with rustfmt and clang-format instead of layout twins | B3 | M | Q1 |
| B13 | One selectable-instruction model consumed by checks and emitters | B1 | M | Q1 |
| B14 | Checks as one pass producing that model | B13 | M | Q2 |

## B12 Format generated code with the formatters

- What: emit plain code from one template per fragment, then run `rustfmt` over the crate and `clang-format` over the allocator during generation.
- Preserves: each fragment's text; `tests/lint.sh` as the gate.
- Benefits: removes the twin templates and the width logic (`fits`, `packed`, `CALL_WIDTH`, and branches such as `isel_ir2isa_rewrites_rs_generator.py:369-399`); most of the 79 templates collapse.
- Caveats: `rustfmt` ships with cargo, which generation already needs; `clang-format` does not and would be a new build requirement: owner decision (Q-open).
- Fail points: formatter version drift changes generated text; pin versions in the env image; noticed by `tests/lint.sh`.
- Order and dependencies: after B3; ideally after B6, which deletes much of what the templates emit.
- Verification: `tests/lint.sh` passes; template count drops; goldens byte-identical (the compiled assembly, not the crate text).

## B13 One selectable-instruction model

- What: one dataclass per selectable instruction holding inputs, output, attributes, pattern, payload values and conditions, built once; every emitter reads it (`instruction_info.py`, `generator.py:49`).
- Preserves: each emitter's output; `expression_writer.py` and `operand_sizes.py` as is.
- Benefits: emitters stop re-visiting ANTLR trees; one place to read what the backend knows of an instruction.
- Caveats: the model must hold expression trees for C++ and Rust writers (`expression_writer.py`).
- Fail points: a field computed differently than before; noticed by generated-crate diffs on the examples.
- Order and dependencies: after B1.
- Verification: generated crates for both examples byte-identical before and after.

## B14 Checks as one pass producing the model

- What: `spec_checks.py` builds the B13 model and rejects while building; no generator imports a check helper and no check imports an emitter (today `spec_checks.py:9-18`).
- Preserves: every rejection, its order and its message.
- Benefits: a rejection and the code it guards cannot drift; `docs/coverage.md` rows map to one function each.
- Caveats: rejection order and messages are pinned by `tests/test_refusals.py`.
- Fail points: a rejection lost in the move; noticed by `tests/test_refusals.py`.
- Order and dependencies: after B13; before B15.
- Verification: `tests/test_refusals.py` passes; import graph has no check-emitter cycle.
