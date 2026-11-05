# Owner decisions and tech-lead proposals: response

Owner decisions and panel decisions bind; each proposal row names the panel question that settled it.

## Owner decisions

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| Decision 11: 3-4 engineers over two quarters | accepted | Binding | overhaul.md lanes; plan/README.md |
| Decision 16: no undefined behavior | accepted | Binding; each coverage row states rejected, not selected, or supported; the `merge` panic becomes a coded fault | coverage-table.md; B26 |
| Decision 17: in-place output reuses the marked input's storage | accepted | Binding; `constraint.cc:214-230` only permits overlap; panel 9 Q9.4 makes tied an equality in the allocator | coverage-table.md; B30; D14 |
| Decision 19: buffers, state, `hbm` | accepted | Binding | coverage-table.md; naming.md; glossary_objections.md; B27, B28 |
| Decision 20: region | accepted | Binding; already the plan's word | none |
| Decision 21: `@s.` stays | accepted | Binding; already the plan's spelling | none |
| Decision 26: act-backend 2.0 | accepted | Binding; breaking items ride it | defects.md; overhaul.md; decisions.md |
| Decision 27: wall-clock user default, deterministic tests | accepted | Binding | B29, B34 |
| Decision 28: cbindgen allowed | accepted | Binding; derives the C header | B2 |
| Decision 29: rustfmt at generation | accepted | Binding; replaces "no formatter run" | B12 |
| Decision 31: no hazard model | accepted | Binding; panel 9 Q9.1 and Q9.3: the author declares each pair's relation, the default is exact-or-apart | B11, B30, B39; D16 |
| Decision 32: "ISA specification" | accepted | Binding | coverage-table.md; plan items |
| Decision 34: negative addressing attributes, no name-keyed meaning | accepted | Attribute solver variables start at 0 (`malloc_instructions_h_generator.py:456`); no name-keyed behavior found in `act_backend/*.py` | D15; B38 |

## Backend proposals R17-R27

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| R17 deterministic gate first | accepted | Q4.2; owner decision 27 keeps the wall-clock mode as the user default | B29, B3, B34 |
| R18 typed payloads, B7, B9 before B6 | accepted | Q4.3 | isel.md |
| R19 conditions as data with one applier | partly | Q4.4: generated Rust bound by name; a two-rule data-form spike may overturn it | B6 |
| R20 one cost-guided extractor | accepted | Q4.5: file merge first, k-best after B29 and B9, enumerator test-only | B37, B10 |
| R21 in-place as a solver equality | accepted | Q4.6 sequenced B30 before B11; Q9.4 narrows the touch exception to each pair's relation | B30, B11, B39 |
| R22 reject overlapping rewrite rules | rejected | Q4.1: overlap allowed, ties by declaration index, exact duplicates warned | B35 |
| R23 keep egg; borrow rule sets, explanations, stop reasons | partly | Q4.10: keep egg, guards on bare-variable rules, phases only if node-limit stops remain | B32, B33 |
| R24 no `clang-format` at generation | partly | Owner decision 29: rustfmt runs on the generated crate at generation | B12 |
| R25 region count checks to the front end | accepted | Q3.2: the backend entry calls `check` once and deletes `check_regions` | B21, B22, B23 |
| R26 rename `TensorInfo`, keep `Buffer` | accepted | `Buffer` kept under owner decision 19 | naming.md |
| R27 state in v3 Q2 after the typed model | partly | Q4.7: read-only state folds in v3; updates after v3 | B16, B24 |

## Other proposals touching the backend

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| R5, R58 remove `set_inplace` and placement marks | rejected | The marks stay (owner decisions 22, 30); Q9.2 expands each to its relation | coverage-table.md; B39 |
| R11 no-semantics instructions | accepted | Q4.7: the silent drop is fixed first | B24 |
| R62 shape module | accepted | Q3.1: shape and per-word type rules are U4's; B23 cut | B23, B7 |
| R9, R44, R57 (`ROOT` placement, clause ids, constant literals) | accepted | Q2.9 leaves `ROOT` unchanged; Q3.8 puts a clause id on line 1 of a message; Q1.10 and Q5.5 parse a literal exactly and range-check the exact value; B20's float range check matches the existing integer check (`spec_checks.py:97-105`) | none |
