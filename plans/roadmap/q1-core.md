# Quarter 1, core tracks

Plan items scheduled for quarter 1 in the core tracks, by track.

## language

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| L1 | Ship the spec skeleton and section ids | -- | S | -- |
| L2 | Pin the reference operation version | L1 | S | -- |
| L3 | Normative shape and element type rules per operation | L2 | L | R62 |
| L4 | Check generators against the stated reduce identities and `constant` fill | L2 | S | -- |
| L5 | Make the region, in-place, and overlap fault rows hold | L1, L36 | S | R25 |
| L6 | 64-bit integer arithmetic per operator with a fault | L1 | S | -- |
| L7 | ACT-Backend unsupported-feature table against section ids | L1 | S | -- |
| L10 | Reject `@a.` outside starts and constraints | L1 | S | -- |
| L11 | Reject late buffers and partial instructions | L1 | S | -- |
| L13 | Align integer literal lexing between the two grammars | L1 | S | -- |
| L14 | Test the exact-token zero divisor rule | L1 | S | -- |
| L15 | Rename confusing internal names | L25 | S | -- |
| L16 | Range and type checks at the declaring call | L3 | S | -- |
| L17 | Element types `f4e2m1fn` and `f8e8m0fnu` | L2, L3 | S | -- |
| L24 | Multi-core, async DMA, mask, stride, saturation, and `scaled_dot` idioms, documented without syntax | L1 | S | -- |
| L25 | Reconcile the glossary with the terminology reference | L1 | S | -- |
| L26 | Apply the glossary to the taidl docs and error messages | L25 | M | -- |
| L29 | Object model contract test for the five fixed names | L1 | S | -- |
| L30 | v3 rename table: public API, token, grammar rule, and `d0` as a one-dimension buffer | L1 | M | -- |
| L31 | Float accuracy bounds for `dot`, `reduce_add`, `reduce_multiply` | L2 | S | -- |
| L32 | Defined-behavior norms: pre-state issue, unwritten reads, numerics, fault guarantees | L1, L2 | M | -- |
| L33 | Remove rare words and redundant spellings, with one coded rejection | L1 | S | -- |
| L34 | Rename "refus" to "reject" in messages, docstrings, READMEs, and identifiers | L30 | S | -- |
| L35 | Add `set_may_overlap` and the pair-relation record | L36 | S | -- |
| L36 | Expand each pair's relation with the size term and the exact-or-apart default | L1 | S | -- |
| L37 | Width layer: a type per attribute, `s64` by default | L1, U2 | M | -- |
| L40 | Generator metadata in the typed contract | U2, U3 | S | -- |

## backend

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| B1 | Target module layout and stage names | B29 | S | -- |
| B2 | Typed candidate and kernel-fact handoff to the C++ allocator | B1, B25, B31 | M | -- |
| B3 | End-to-end gate: goldens, chosen assembly, candidate logs, oracle run | B29 | M | -- |
| B4 | Remove the C++ candidate parser | B2 | M | -- |
| B6 | One rewrite value per rule, variables bound by name | B7, B9 | M | R18, R19 |
| B7 | Shape and element type inferred in `Analysis::make` | B31 | L | R18 |
| B8 | Name and test each IR-to-IR rewrite rule | B6 | S | -- |
| B9 | Extraction without cloning or mutating the e-graph | B7 | S | R18 |
| B13 | One selectable-instruction model built from the front end's objects | B1, U2 | M | -- |
| B14 | Checks as one pass producing that model | B13, B21 | M | -- |
| B15 | Keep `docs/coverage.md` in sync by a test; document S1.3 cases | B14, L7 | S | -- |
| B21 | Call `Accelerator.validate()` first and delete the backend's region count check (D1, D2) | U3 | S | -- |
| B24 | Reject, or report, instructions with no semantics (D4) | L5 | S | -- |
| B25 | C++ errors returned, never `assert` (D5) | -- | S | -- |
| B26 | Extraction errors as dropped candidates, unset shape as `Option`, shape disagreement as a coded fault (D7, D8) | B7 | S | -- |
| B27 | Glossary and owner vocabulary in messages, docs and Python names | L25, L27 | S | -- |
| B28 | Rust and C++ identifier and log renames | B1 | S | -- |
| B29 | Deterministic search mode and `search` as a library function | -- | M | -- |
| B30 | Pair relations in the allocator: tie equality and a narrowed touch exception | B2, B39 | M | -- |
| B31 | Typed payloads in `TensorOp` | B1 | M | -- |
| B35 | Overlapping IR-to-ISA rules: declaration-index tie-break, duplicate warning | B29, B6 | S | -- |
| B37 | One extraction module and no re-extraction after saturation | B1, B29 | S | -- |
| B38 | Allocator attribute domains follow the width layer (D15); default `s64` | -- | S | -- |
| B39 | Pair-relation record in the typed handoff | B2, L35 | S | -- |
| B40 | Allocation rows per relation | B30 | S | -- |
| B41 | Every candidate-level failure is a dropped candidate with a recorded reason (D5, D7, D8, D17 to D21) | B25, B26 | M | -- |

## oracle

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| O1 | Runtime code out of `API_FILE` into `runtime/`; split what remains of `utils.py` | O20 | M | R28 |
| O2 | One instruction record from the visitor, consumed by checks and emission | O1, O5 | M | R30 |
| O5 | Delete the dead branch and the dead `sh_copy` check | -- | S | -- |
| O6 | Spec errors raise `ValueError`; S8.3 table from the front end | L3 | S | R62 |
| O8 | Round float literals once, from the exact decimal | L16 | S | -- |
| O9 | `triangular_solve` at `f16`: measure, then widen only where needed | L3 | S | -- |
| O10 | Overlap constraints and in-place marks: check the list before any load, value from the pre-state | L5, O25 | S | -- |
| O11 | Fallback only if O19 fails: keep `while` with an equivalence test; reject wrapping starts | O19 | S | R28 |
| O14 | Fold `docs/operations.md` into `docs/coverage.md` and README | L1, L7 | S | R33 |
| O15 | Apply the naming table: identifiers, comments, docs | L25, L27, O1 | M | -- |
| O16 | Generation log lines that state what happens | -- | S | -- |
| O17 | Keep `docs/coverage.md` true by test | O14 | S | -- |
| O18 | Drop the hand-written line splitter for generated code | -- | S | R35 |
| O19 | Spike: unroll every loop on a synthetic stress kernel against an owner-set bound | O21 | S | -- |
| O20 | Unroll every loop; delete loop algebra, `Unroll`, `while` emission, `held`, `fixed` | O19, O8 | M | -- |
| O21 | XLA flags per compile: no excess precision, no flush to zero, IEEE min and max | -- | S | -- |
| O22 | Enforce the numerics norm table: a guard per row and a golden per row | L6 | M | -- |
| O23 | Fixed values for integer divide, remainder and float-to-integer convert; no data faults | O22 | S | -- |
| O24 | Call-time errors: checked 64-bit values, constraint overflow, initial state | L5, L6, O20 | S | -- |
| O25 | Expressions read the pre-state; updates commit with the outputs | L5 | S | -- |
| O27 | Widen narrow float `dot` and `reduce` to `f32` in the module | L6 | S | -- |
| O30 | Overlap test matrix: declarations by placements | O10 | S | -- |
| O31 | Failure policy: errors only on invalid input, naming instruction and clause; audit every raise site | O20, O6 | M | -- |
| O32 | Width layer: the oracle checks attribute types as implicit range constraints | O10, O24 | S | -- |

## tests

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| T1 | Tag every surviving test with its language specification section | L1 | M | -- |
| T2 | Remove hidden fixture coupling in the two conftest.py files | -- | M | -- |
| T3 | One shared test kit for helpers duplicated across repos | T2 | M | -- |
| T5 | Corpus format, location, and naming | L1 | S | -- |
| T6 | Front-end verdict runner in taidl | D3, T5 | M | -- |
| T7 | Seed the corpus from existing front-end tests (S2 to S7) | T1, T5, T6 | L | -- |
| T8 | Operations corpus with reference values (S8) | L3, T5 | L | -- |
| T10 | Real ISA specifications as the end-to-end list | T5 | S | -- |
| T11 | Expectations table per generator, coverage.md generated from it | D1, T5 | M | -- |
| T12 | Expectation runner: supported, or unsupported with the documented code | D6, T11, T6 | M | -- |
| T14 | One shared end-to-end suite in act | B29, T10, T3 | M | -- |
| T16 | CI wiring: corpus travels with the pinned taidl release | T12, T5 | S | -- |
| T17 | Split `act-oracle/tests/test_kernel.py` by theme | T1 | M | -- |
| T19 | Decision: corpus lives in taidl, consumed through the pin | T5 | S | -- |
| T20 | Decision: impl tests survive with a header, PR-defense tests do not | -- | S | -- |
| T21 | Decision: expectations table is the source, coverage.md is rendered | T11 | S | -- |
| T22 | Answers in force for the test-suite questions | -- | S | -- |
| T26 | Norm catalogue and coverage report | D14, T5, T6 | M | -- |
| T27 | CI tiers, budgets, and deterministic mode | B29, T12 | M | -- |
| T28 | Mapped-deletion check | -- | S | -- |
| T29 | Program runner over an oracle interface | D8, T5, T6 | M | -- |
| T30 | Overlap matrix, rejection, and cross-check cases | D8, T29, T5, T6 | M | -- |
| T35 | Generator metadata cases (neutrality, namespaces) | L40, T12, T5, T6, U2 | M | -- |

## act

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| A1 | Apply glossary terms and plain words in README.md | A8, L26 | S | R51, R52 |
| A2 | Fix README structure, punctuation, and CI description | -- | S | R51 |
| A4 | Rename vague identifiers and comments in tests/ | A1 | S | -- |
| A5 | Fix act/ docstrings and the distribution name | A8 | S | -- |
| A7 | ASCII and one-sentence-per-line gate for text files | A11 | S | -- |
| A8 | One name per generator across the four repos | -- | M | -- |
| A9 | Retire the "current tutorial" label | -- | S | -- |
| A10 | List the tutorial syntax that fails on the v3 API | -- | M | R50, R63 |
| A11 | Archive all three tutorials on their tags and remove them from master | A10 | M | R50, R54 |
| A15 | Close gaps in check_versions.py | -- | S | -- |
| A16 | Write the release order and a checklist | -- | M | -- |
| A20 | Run the quickstart in the image | A25 | S | -- |
| A25 | Write the quickstart | A1 | S | -- |
