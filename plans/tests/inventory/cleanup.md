# Clean-up items

The four items the inventory calls for before the conformance suite lands.
T1 tags, T2 and T3 fix the fixtures, T4 deletes the text pins last.

## Index

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| T1 | Tag every surviving test with its language specification section | L1 | M | Q1 |
| T2 | Remove hidden fixture coupling in the two conftest.py files | - | M | Q1 |
| T3 | One shared test kit for helpers duplicated across repos | T2 | M | Q1 |
| T4 | Delete tests that pin generated text or internal names | T11, T13, T14, T20 | S | Q2 |

## T1. Tag every surviving test with its language specification section

- What: each kept test module gets a `pytestmark` or per-test marker `spec('S5.2')` naming the outline id it exercises.
- Benefits: a section with no test shows up in one query; a test with no section is a candidate for deletion.
- Caveats: tags are only as good as the outline ids; L1 freezes ids before tagging starts.
- Fail points: tags copied without reading the test; noticed when the T26 report lists a section a test never reaches.
- Order and dependencies: after L1 publishes the outline; before T7 and T11.
- Verification: a `--spec-report` run lists every S id with its test count and no kept test lacks a tag.
- Breaking: no.
- Files: every kept test file of the four repos; preserved whole, plus a marker per test; TB1, TB2, TB4 gain their missing assertion in the same PRs.

## T2. Remove hidden fixture coupling in the two conftest.py files

- What: replace `request.module.EXAMPLE_DIR` in `work_dir`, `plain_dir`, and `assembly` with an explicit `example` fixture parameter; move the formatter check out of `work_dir` into its own test; move import-time generation in `act-oracle/tests/test_mini_trn.py:20` into a session fixture.
- Benefits: a module can exercise several examples; a formatting failure fails one test, not a module.
- Caveats: B and O own these trees; this item waits for their tracks to land.
- Fail points: a fixture rename breaks every module at once; noticed by collection errors on the first run.
- Order and dependencies: first PR of the test track in each generator repo; before T3.
- Verification: no `request.module` in either conftest; every `generate_*` call sits inside a fixture.
- Breaking: no.
- Files: `act-backend/tests/conftest.py:463-495`, `act-oracle/tests/test_mini_trn.py:20`; every fixture behaviour preserved, only the wiring changes.

## T3. One shared test kit for helpers duplicated across repos

- What: move `load_bf16_matrix`, `import_oracle`, `load_golden`, `bf16`, `attention_reference`, and the element type tables into two packages the test suites import.
- What: `taidl.testing` holds the corpus loader and the tolerance implementation; `act/testkit/` holds goldens and the oracle import.
- Benefits: one tolerance policy, one golden format, one way to import a generated oracle; the taidl wheel does not depend on an oracle.
- Caveats: adds a dependency to each test requirements file.
- Fail points: the kit drifts from a generator's API; noticed by import errors in that repo's CI only.
- Order and dependencies: after T2; feeds T13 and T14.
- Verification: `grep -n 'def load_golden' */tests/conftest.py` finds nothing; both suites import the kit.
- Breaking: no.
- Files: both `conftest.py` files; the five duplicated helpers move, their callers import the kit.

## T4. Delete tests that pin generated text or internal names

- What: delete the layout tests of `act-backend/tests/test_generators.py:175-393`, the egraph text pins `test_mini_trn.py:68-78` and `test_mini_trn_double.py:70-80`, and the regex reads of `egraph.rs` and `dtype.rs` at `test_refusals.py:164-205`.
- What: the layout tests go because rustfmt runs at generation and one test per tree checks the output is formatted.
- What: the test at `test_generators.py:243`, which shows a mark reaching the e-graph, stays as a per-PR impl test, since the markers stay.
- Benefits: a rename inside the generated crate no longer fails a test; what remains fails only on a behaviour change.
- Caveats: the arity check at `test_refusals.py:164-176` stays an impl test until the T12 slow tier runs per pull request.
- Caveats: the egraph regex tests guard a real invariant, that the generator and the Rust reader agree on words; T11 and T12 carry it before these go.
- Fail points: deleting early loses the invariant; noticed only by a word that the generator accepts and the reader rejects at compile time.
- Order and dependencies: after T11, T13, T14, and T20; T28 maps each deletion to a successor.
- Verification: `grep -rn "TensorOp::" */tests/` finds nothing; the invariant has a T11 test.
- Breaking: no.
- Files: `act-backend/tests/test_generators.py:175-393`, `test_refusals.py:164-205`, `test_mini_trn*.py` text pins; the invariants survive in T11 and T14, the text pins do not.
