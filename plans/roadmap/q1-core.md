# Quarter 1, core tracks

Plan items scheduled for quarter 1 in the core tracks, by track.

## language

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| L1 | Ship the spec skeleton and section ids | -- | S | -- |
| L2 | Pin the reference operation version | L1 | S | -- |
| L4 | Check generators against the stated reduce identities and `constant` fill | L2 | S | -- |
| L5 | Make the region and in-place fault rows hold | L1 | S | R25 |
| L6 | Exact integer arithmetic with a 64-bit fault | L1 | S | -- |
| L7 | ACT-Backend unsupported-feature table against section ids | L1 | S | -- |
| L10 | Reject `@a.` outside starts and constraints | L1 | S | -- |
| L11 | Reject late buffers and partial instructions | L1 | S | -- |
| L13 | Align integer literal lexing between the two grammars | L1 | S | -- |
| L14 | Test the exact-token zero divisor rule | L1 | S | -- |
| L24 | Multi-core, async DMA, mask, stride, and saturation idioms, documented without syntax | L1 | S | -- |
| L25 | Reconcile the glossary with the terminology reference | L1 | S | -- |
| L26 | Apply the glossary to the taidl docs and error messages | L25 | M | -- |
| L29 | Object model contract test | L1 | S | -- |
| L30 | Owner vocabulary in the public API and `d0` as a one-dimension buffer | L1 | S | -- |
| L31 | Float accuracy norm for `dot`, `reduce_*`, and transcendental words | L2 | S | -- |

## backend

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| B1 | Target module layout and stage names | B29 | S | -- |
| B2 | Typed candidate and kernel-fact handoff to the C++ allocator | B1, B13 | M | -- |
| B3 | End-to-end gate: goldens, chosen assembly, candidate logs, oracle run | B29 | M | -- |
| B4 | Remove the C++ candidate parser | B2 | M | -- |
| B6 | One rewrite value per rule, variables bound by name | B7, B9 | M | R18, R19 |
| B7 | Shape and element type inferred in `Analysis::make` | B31 | L | R18 |
| B8 | Name and test each IR-to-IR rewrite rule | B6 | S | -- |
| B9 | Extraction without cloning or mutating the e-graph | B7 | S | R18 |
| B13 | One selectable-instruction model built from the front end's objects | B1 | M | -- |
| B14 | Checks as one pass producing that model | B13 | M | -- |
| B15 | Keep `docs/coverage.md` in sync by a test; document S1.3 cases | B14, L7 | S | -- |
| B21 | Reject a region value whose element count differs from its region (D1) | B3, L5 | S | -- |
| B22 | Reject a region value whose element type differs from its buffer (D2) | B3, L5 | S | -- |
| B24 | Check, or reject, instructions with regions and no semantics (D4) | L5 | S | -- |
| B25 | C++ errors returned, never `assert` (D5) | B4 | S | -- |
| B26 | Extraction errors as dropped candidates, unset shape as `Option` (D7, D8) | B7 | S | -- |
| B27 | Glossary and owner vocabulary in messages, docs and Python names | L25, L27 | S | -- |
| B28 | Rust and C++ identifier and log renames | B1 | S | -- |
| B29 | Deterministic search mode and `search` as a library function | -- | M | -- |
| B30 | In-place storage reuse as a solver equality | B2 | M | -- |
| B31 | Typed payloads in `TensorOp` | B1 | M | -- |

## oracle

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| O5 | Delete the dead branch and the dead `sh_copy` check | -- | S | -- |
| O10 | In-place mark: the output reuses the marked input's storage | L5 | S | -- |
| O14 | Fold `docs/operations.md` into `docs/coverage.md` and README | L1, L7 | S | R33 |
| O16 | Generation log lines that state what happens | -- | S | -- |
| O17 | Keep `docs/coverage.md` true by test | O14 | S | -- |
| O18 | Drop the hand-written line splitter for generated code | -- | S | R35 |
| O21 | Set `xla_allow_excess_precision` false; test per-op rounding | -- | S | -- |

## tests

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| T2 | Remove hidden fixture coupling in the two conftest.py files | -- | M | -- |
| T3 | One shared test kit for helpers duplicated across repos | T2, T22 | M | -- |
| T22 | Open questions for the owner | -- | S | -- |

## act

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| A1 | Apply glossary terms and plain words in README.md | A8 | S | R51, R52 |
| A2 | Fix README structure, punctuation, and CI description | -- | S | R51 |
| A3 | Remove news, roadmap, and release-history lines from README | -- | S | -- |
| A4 | Rename vague identifiers and comments in tests/ | A1 | S | -- |
| A5 | Fix act/ docstrings and the distribution name | A8 | S | -- |
| A7 | ASCII and one-sentence-per-line gate for text files | A12 | S | -- |
| A8 | One name per generator across the four repos | -- | M | -- |
| A9 | Name the current tutorial in README.md or retire the label | -- | S | -- |
| A10 | Check the current tutorial against the pinned image and API | -- | M | R50, R63 |
| A12 | Terminology and ASCII pass on the current tutorial | A1 | M | -- |
| A14 | Align paper titles and setup text | A1 | S | -- |
| A15 | Close gaps in check_versions.py | -- | S | -- |
| A16 | Write the release order and a checklist | -- | M | -- |
| A20 | Test the README usage snippet | A1 | S | -- |
