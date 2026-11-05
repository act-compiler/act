# Quarter 1, core tracks

Plan items scheduled for quarter 1 in the core tracks, by track.

## language

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| L1 | Ship the spec skeleton and section ids | -- | S | -- |
| L2 | Pin the reference operation version | L1 | S | -- |
| L4 | Fix the `reduce_<reducer>` initial values and `constant` fill | L2 | S | -- |
| L5 | Close region edge cases: bounds, zero sizes, output overlap | L1 | M | R25 |
| L6 | Fix integer arithmetic: `/`, `%`, overflow | L1 | S | -- |
| L7 | Per-generator unsupported-feature tables against section ids | L1 | M | -- |
| L10 | Reject `@a.` outside starts and constraints | L1 | S | -- |
| L11 | Reject a data model declared after an instruction | L1 | S | -- |
| L13 | Align integer literal lexing between the two grammars | L1 | S | -- |
| L14 | Catch every literal-zero divisor | L1 | S | -- |
| L24 | Multi-core and async DMA idioms, documented without syntax | L1 | S | -- |
| L25 | Reconcile the glossary with the terminology reference | L1 | S | -- |
| L26 | Apply the glossary to the taidl docs and error messages | L25 | M | -- |
| L27 | Apply the glossary to B, O, and A docs | L25 | M | -- |

## backend

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| B1 | Target module layout and stage names | -- | S | -- |
| B2 | Typed candidate handoff from Rust to the C++ allocator | B1 | M | -- |
| B3 | End-to-end golden harness that runs without docker | -- | M | -- |
| B4 | Remove the C++ candidate parser | B2 | M | -- |
| B6 | One Rust value per rewrite: pattern, condition, result data | B1, B3 | M | R18, R19 |
| B7 | Shape and element type inferred in `Analysis::make` | B6 | L | R18 |
| B8 | Name and test each IR-to-IR rewrite rule | B6 | S | -- |
| B9 | Extraction without cloning or mutating the e-graph | B7 | M | R18 |
| B10 | One extractor module with two strategies over shared code | B1, B3 | M | -- |
| B12 | Format generated code with rustfmt and clang-format instead of layout twins | B3 | M | R19 |
| B13 | One selectable-instruction model consumed by checks and emitters | B1 | M | -- |
| B21 | Reject a region value whose element count differs from its region (D1) | B3, L5 | S | -- |
| B22 | Reject a region value whose element type differs from its data model (D2) | B3, L5 | S | -- |
| B24 | Check, or reject, instructions with regions and no semantics (D4) | L5 | S | -- |
| B25 | C++ errors returned, never `assert` (D5) | B4 | S | -- |
| B27 | Glossary terms in messages, docs and Python names | L25, L27 | S | -- |
| B28 | Rust and C++ identifier and log renames | B1 | S | -- |

## oracle

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| O1 | Split `generic/utils.py` by role and rename `generic/` to `runtime/` | -- | M | R28 |
| O2 | One line record from the visitor, consumed by checks and emission | O1, O5 | M | R30 |
| O4 | One word table: types, dialect, emitter, reducer | O5 | S | R30 |
| O5 | Delete dead branch and the double attribute visit | -- | S | -- |
| O10 | Define and implement `set_inplace` | L5 | S | -- |
| O11 | Reject a loop-expression start whose 64-bit steps overflow | L6 | S | R28 |
| O14 | Fold `docs/operations.md` into `docs/coverage.md` and README | L1, L7 | S | R33 |
| O15 | Apply the naming table: identifiers, comments, docs | L25, L27, O1 | M | -- |
| O16 | Generation log lines that state what happens | -- | S | -- |
| O17 | Keep `docs/coverage.md` true by test | O14 | S | -- |
| O18 | Drop the hand-written line splitter for generated code | O1 | S | R35 |

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
