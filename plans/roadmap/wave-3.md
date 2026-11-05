# Wave 3

One front end checks for every generator; 59 items, weight 128.
Wave 3 ends at M2.
Later-quarter items with no wave 1 or 2 dependent start here, so wave 1 stays measurable.
The check function and the traced surface land here, in the Q11.4 order: `check` on the string front end, generators on the `Model`, then the tracer.

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| L8 | Spec consistency checks in CI | L1 | S | -- |
| B4 | Remove the C++ candidate parser | B2 | M | -- |
| B9 | Extraction without cloning or mutating the e-graph | B7 | S | -- |
| B6 | One rewrite value per rule, variables bound by name | B7, B9 | M | -- |
| B13 | One selectable-instruction model built from the front end's `Model` | B1, U2 | M | -- |
| B26 | Extraction errors as dropped candidates, unset shape as `Option`, shape disagreement as a coded fault (D7, D8) | B7 | S | -- |
| B30 | Pair relations in the allocator: tie equality and a narrowed touch exception | B2, B39 | M | -- |
| B5 | Reader checks HLO annotations against the inferred shapes | B7 | S | -- |
| B10 | One k-best extractor on (class, buffer) | B29, B37, B9 | M | -- |
| B16 | Read-only state folded at compile time | B13, B24, L1 | M | -- |
| B18 | Element types `s2`, `s4`, `u2`, `u4`, `f8e4m3fn`, `f8e5m2` | B7, L17 | S | -- |
| B40 | Allocation rows per relation | B30 | S | -- |
| B41 | Every candidate-level failure is a dropped candidate with a recorded reason (D5, D7, D8, D17 to D21) | B25, B26 | M | -- |
| O8 | Round float literals once, from the exact decimal | L16 | S | -- |
| O20 | Unroll every loop; delete loop algebra, `Unroll`, `while` emission, `held`, `fixed` | O19, O8 | M | -- |
| O12 | Drop the matmul-precision flag; drop x64 after a feed test | -- | S | -- |
| O23 | Fixed values for integer divide, remainder and float-to-integer convert; no data faults | O22 | S | -- |
| O28 | Build the C++ debug handler once per install and jax version | -- | S | -- |
| O29 | Study: migration plan of the oracle onto fsim-lib | -- | M | fsim-lib study only (owner 23) |
| O31 | Failure policy: errors only on invalid input, naming instruction and clause; audit every raise site | O20, O6 | M | -- |
| T21 | Decision: expectations table is the source, coverage.md is rendered | T11 | S | -- |
| T15 | Golden policy: one regenerate, one record, one tolerance | B29, T14 | S | -- |
| A4 | Rename vague identifiers and comments in tests/ | A1 | S | -- |
| A25 | Write the quickstart | A1 | S | -- |
| A20 | Run the quickstart in the image | A25 | S | -- |
| A6 | Comment pass on docker/ scripts | -- | S | -- |
| A17 | Check sibling pins and dependency floors against act | A15 | M | -- |
| A18 | Add a scheduled compatibility build | A16 | M | -- |
| A19 | Cover state variables and constraints in the end-to-end test | -- | M | -- |
| A26 | Gate the tag with a fresh-venv install | A25 | S | -- |
| C6 | Added element types: evidence for L17 | L17 | S | -- |
| C7 | Numeric annotations: evidence for L21 | C10 | S | -- |
| C8 | Data-dependent addressing: study that admits L23 | C11 | M | -- |
| C12 | Re-run the case studies on v3, add NVDLA | C1, C10, C11, C8 | S | -- |
| C5 | Body operations: evidence that admits L20 | C12 | S | -- |
| P8 | Normative execute function and reference-interpreter status | P2 | M | -- |
| P9 | Generated instruction documentation pages | P1, P2 | S | -- |
| U3 | The check function | L5, U2 | M | `check` from the string front end first (Q11.4 step 2) |
| L43 | Check function `taidl.check` returning a frozen `Model`, and generator checks | U2, U3 | M | the type barrier of Q11.3 |
| U10 | Generators read only the checked model | U3 | S | generators on the `Model` before the tracer (Q11.4 step 2) |
| L42 | Python surface by tracing | L43 | L | tracer after `check` (Q11.1, Q11.4 step 3) |
| U12 | Authoring surface by tracing | U3 | L | tracer after `check` (Q11.4 step 3) |
| L14 | Test the plain-zero divisor rule | L42 | S | -- |
| L40 | Generator metadata in the typed contract | U2, U3 | S | -- |
| L41 | Guide: adding a generator | L40 | S | -- |
| B21 | Entry takes a `Model` or calls `check` once; delete the backend's region count check (D1, D2) | U3 | S | absorbs B22 (Q8.7) |
| U4 | Shape and per-word element-type rules in the front end | L3, U3 | L | absorbs L12 (Q8.4 step b) |
| U5 | Element-type table and literal check | L16, O8, U2 | M | -- |
| U6 | Integer folding in the front end | L14, L6, U2 | S | -- |
| A27 | Document how to add a new generator | A25, A8, U10, U2, U3 | M | -- |
| D2 | Call sites and exception classes in the front end | D1, U3 | M | -- |
| D17 | Rejections at the author's line | D1, D2, U12 | S | -- |
| D7 | Backend runtime diagnostics (Rust and C++) | B25, B26, B4, B5, D1 | L | -- |
| D8 | Oracle run-time and call diagnostics | D1, D14, L6, O11, O6 | M | -- |
| D9 | Silent fallbacks become errors | B24, B26, D1, L5 | S | -- |
| D12 | act umbrella: scripts and entry points | A15, A6, D1 | S | -- |
| D15 | Removed-names rejection | D1, L15, U2 | S | -- |
| F4 | State variable semantics across both generators | F3 | L | -- |
| F5 | Numeric classes, bounds, reduction identities | F6 | M | -- |
