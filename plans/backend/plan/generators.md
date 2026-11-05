# Python generators and templates

The Python side is 12 generator modules filling 79 text templates (`act_backend/templates/`).
Many templates exist in several layouts (`TWINS`, `template_loader.py:14`), chosen by line width so the output already passes rustfmt and clang-format.
No gate checks the generated crate's layout: `tests/lint.py:19,27` runs clippy and `-Werror`, and the CI format check covers `git ls-files` only (`README.md:110`).
Each emitter re-walks the TAIDL ANTLR tree (`isel_ir2isa_rewrites_txt_generator.py:56-65`), and `spec_checks.py` imports emitter internals (`spec_checks.py:9-18`).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B12 | Emit plain code without layout twins; rustfmt at generation | B6 | S | Q2 |
| B13 | One selectable-instruction model built from the front end's `Model` | B1, U2 | M | Q1 |
| B14 | Backend checks after `check`: reject or exclude, producing that model | B13, B21 | M | Q1 |
| B46 | Exclusion report: printed, stored in the crate, named at kernel compile | B14 | S | Q1 |

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

- What: one dataclass per selectable instruction holding inputs, output, attributes, pattern, payload values and conditions, built from the front end's `Model`, not ANTLR contexts.
- What: B13 builds from today's `instruction.py` objects under U2's documented contract until `check` returns the `Model`; it then reads the `Model` and its derived facts only (Q3.5, Q11.3).
- Preserves: each emitter's output; `expression_writer.py` and `operand_sizes.py`.
- Benefits: emitters stop re-visiting ANTLR trees; one place says what the backend knows of an instruction.
- Caveats: the model holds no allocator struct; B2's Rust struct is a separate schema (Q4.11).
- Fail points: a field computed differently than before; noticed by generated-crate diffs.
- Order and dependencies: after B1 and U2 (Q3.3); re-pointing at a later front-end model is small.
- Verification: generated crates for both examples identical.

## B14 Backend checks after `check`: reject or exclude

- What: the backend's checks are plain functions the entry calls after `check` (B21); they build the B13 model; no check imports an emitter; there is no registry and no checker subclass (Q11.3).
- What: a language rule is `check`'s (Q3.2, Q3.11); a check the backend keeps is a backend limit, listed in the coverage doc.
- What: an unknown key in the `backend` metadata namespace rejects, naming node and key, since the architect can correct it.
- What: an instruction is excluded from the backend's instruction set, with a reason, when it reads or updates state, has other than one output, has a size over an addressing attribute or state, has a constraint the allocator cannot hold, or uses a tensor operator outside the supported table.
- What: the other model limits of `coverage.md` (an element type outside the supported 13, a `slice` stride, a `constant` expression, a `concatenate` arity) follow the same rule.
- What: each exclusion carries the code, the section id, and the author's line (B46); an empty usable set rejects.
- Preserves: every backend-limit condition and its message text, which becomes the reason; rejection order for the cases that still reject.
- Benefits: a valid ISA with one semaphore instruction needs one specification, not two (casestudies/trainium2.md:115); the backend and the oracle agree on what is valid; each coverage row maps to one function.
- Caveats: `tests/test_refusals.py` pins order and messages; the cases that became exclusions assert the reason in the report instead.
- Caveats: no strict mode turns exclusions into errors; the printed report and the expectations table give the signal.
- Fail points: a limit that excludes silently; noticed by a test per exclusion row that reads the printed and the stored report.
- Order and dependencies: after B13; before B15.
- Verification: `tests/test_refusals.py` passes for the rejecting cases; one test per exclusion row; no check-emitter import cycle.

## B46 Exclusion report

- What: each exclusion is one line with code, section id, instruction, and the author's file and line in the Q3.8 form (D6, D18); it is printed at generation and stored in the generated crate.
- What: the generation summary prints the count of excluded instructions.
- What: the crate records, for each excluded instruction, the tensor operators of its semantics.
- What: a kernel with no covering is rejected at kernel compile time (B42); the message names the lowest uncovered operation and lists, as a hint, each excluded instruction whose tensor operators include it, with its stored reason.
- What: `generate_backend` returns the frozen report (U2); the backend binary at `<base_dir>/backends/<accelerator>` is a path the report lists.
- What: S1.2 gains the exclusion outcome beside rejection; both name the feature and neither gives it another meaning (Q11.3).
- Benefits: no silent drop; the architect sees what the target left out and why.
- Caveats: the report is part of the crate and does not change the emitted compiler; the neutrality test compares output with metadata stripped, not the report.
- Fail points: an exclusion printed but not stored; noticed by a test that reads the crate.
- Order and dependencies: after B14; before B42's kernel check.
- Verification: an ISA specification with a semaphore instruction and one usable instruction prints and stores one line; a kernel whose only cover is the semaphore is rejected with it in the hint; an ISA specification of only excluded instructions rejects.
