# Critic review: response

Rows follow the critic's verdict, the per-item table, then the missing items.

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| Verdict: B6 keeps conditions that index RHS positions | accepted | `ir2ir_rewrites.rs:78-85,103-135` | B6 |
| Verdict: string payload, discriminant-only `matches` | accepted | `egraph.rs:63-110,163-165` | B31 |
| Verdict: two goldens are a thin net for B7 | accepted | Oracle simulation exists (`tests/conftest.py`) | B3 |
| B1: same commit as B10 and B5 moves | partly | B1 moves the extractor and reader files with B28 once; B10 and B5 are now behavioral and stay separate commits (one concern per commit) | B1 |
| B2: generate the struct for both sides from B13 | partly | One schema, but Q4.11 makes the Rust struct the schema and cbindgen derives the C header (owner decision 28) | B2, B13 |
| B3: oracle-differential gate for behavioral steps | accepted | `tests/test_extraction.py`, `test_matching.py` specs | B3 |
| B4 keep | accepted | `parser.h:16-33` | none |
| B5: defer the split | accepted | `tests/test_reader.py` pins messages; no driver for a split | B5 (redefined) |
| B6: bind variables by name | accepted | Same evidence as the verdict row | B6 |
| B6: attributes as child e-nodes so egg matches them | partly | Typed fields (B31) remove the string parse; attributes as e-classes would mix integers with values in one `Language` and conditions still read them; `find_matches` stays only where a condition reads an unbound inner e-node | B31, B6 |
| B7 keep; needs typed payload | accepted | egg `Analysis::make` docs | B7 depends on B31 |
| B8: fires and near-miss tests | accepted | One named rule per test is egg practice | B8 |
| B9: add the constant node in `Analysis::modify` | partly | Constant-ness in class data accepted; a node added in `modify` grows the e-graph during saturation against `NODE_LIMIT` (`lib.rs:20`); candidate building emits it instead | B9 |
| B10: keep both strategies | partly | Merge accepted; two strategies replaced by one k-best extractor, since the default cost counts `api.` lines (`count.cc:17`) and k-best still gives the allocator several candidates | B10 |
| B10: dedupe candidates across passes | accepted | Unchanged e-graph re-extracted (`pipeline.rs:544-547`) | B34 (new) |
| B11 keep | accepted | `inplace_fix.rs:9-18` | B11 |
| B12: no formatter run needed | rejected | Owner decision 29: rustfmt runs on the generated crate at generation | B12 |
| B13 keep; emit the B2 struct | partly | B13 kept; the B2 struct is not emitted from it (Q4.11) | B13 |
| B14 keep | accepted | `spec_checks.py:10-19` | none |
| B15 keep | accepted | -- | none |
| B16 keep, Q2 | accepted | Largest item; depends on L1 and O | none |
| B17 to B20 keep | accepted | -- | none |
| B21 to B26 keep | accepted | Each traces to a read bug | none |
| B27, B28 keep | accepted | -- | none |
| Drop the Q2 data-driven applier experiment | accepted | Q4.4: no separate experiment; the data form is a spike on the first two B6 rules | B6; decisions.md |
| Keep the allocator port rejected | accepted | `solver.h:26-63` | none |
| Missing: typed payload | accepted | `egraph.rs:63-110,137-150` | B31 (new) |
| Missing: target constants in rewrite 0 | accepted | Checked: `16` and `Dtype::I32` at `ir2ir_rewrites.rs:20-27,100` | B33 (new) |
| Missing: rule sets and phasing | accepted | Rules 2, 3, 4, 7 are `(?x) => ...` (`ir2ir_rewrites.txt`); Isaria phases rules | B32 (new) |
| Missing: search budget | accepted | `TIME_LIMIT` 1 s, `N` 1 (`lib.rs:18-24`) | B34 (new) |
| Missing: differential testing as the gate | accepted | Same as B3 row | B3 |
| Fail point: B6 `dot_?` relies on `FromOp` | accepted | `egraph.rs:222-335` | B31 caveat |
| Fail point: B2 `--log` files read by tests | accepted | `tests/test_extraction.py:109,135,496` | B2 |
| Dependency: B12 after B6 | accepted | B6 rewrites the templates B12 removes | B12 |
| Dependency: B9 before B10 | accepted | Both touch `annotate_constants` | B9, B10 |
