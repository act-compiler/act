# Architect review: response

Rows follow the review's "Plan step by step" table, then its "What the plan would lose" and target-architecture points.

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| Step 1: B3 goldens neither prove an unchanged selection nor stay stable | accepted | Goldens are simulated outputs (`README.md:109`); search is wall-clock bounded (`pipeline.rs:155-180`, `lib.rs:18`) | B29 (new); B3; decisions.md |
| Step 2: B1, B28 first, before lanes split | accepted | Pure moves touch every `use` line; cheapest before parallel work | overhaul.md step 2 |
| Step 3: B13 from a typed front-end IR, not ANTLR | partly | Emitters re-walk ANTLR (`isel_ir2isa_rewrites_txt_generator.py:56-65`); the front-end contract is a language-track item with no id yet, so B13 builds from `instruction.py` objects until it lands | B13 |
| Step 4: typed payloads, then B7, then B6 | accepted | B6 first regenerates `set_shapes_N` (`applier.rs:43-56`) that B7 deletes; `payload_ints` (`egraph.rs:63-94`) survives B6 | isel.md order; B31 |
| Step 4: `dot_?` caveat inverted; payloads never match | accepted | `Language::matches` compares discriminants (`egraph.rs:163-165`); rule 6 relies on `precond_6` (`ir2ir_rewrites.rs:514-526`) | B31, B6, B8; decisions.md |
| Step 5: B2 misses `metadata.json`; `hold_tensors` is allocator logic | accepted | `act_malloc.cc:147-153`; `parser.cc:146,335` | B2, B4 |
| Step 6: B10 freezes a cost-blind DFS and an exponential enumeration | accepted | `fast.rs:88-128`; `slow.rs:117`; default cost counts `api.` lines (`count.cc:17`) | B10 |
| Step 7: B12 over-engineered; decide data vs code now | partly | B12 after B6 accepted (B6 rewrites what B12 formats); the data-vs-code choice is open: the critic argues the opposite on evidence, so the panel settles it | B12, B6 |
| Step 7: B14 removes the import cycle | accepted | `spec_checks.py:9-18` | none |
| Step 8: B7, B9, B26 move to Q1 | accepted | `merge` makes failures loud (`egraph.rs:406-411`); the B3 gate makes the risk measurable | isel.md; defects.md; overhaul.md |
| Step 8: B9 right after B7, S-sized | accepted | `is_const` exists in class data (`egraph.rs:370-374`) | B9 |
| Step 9: B21, B22, B24, B27 agree; B23 rules from the B7 table | accepted | One rule table serves both | B23 |
| Step 9: B23 re-homed to the language track | partly | An ill-typed semantics is a defined error under owner decision 16 (`bugs.md` D3); which track raises it is pending panel (R25, R62) | B23 |
| Step 10: B11 upgraded to a solver equality | accepted | Owner decision 17; `constraint.cc:214-230` only skips the interference edge | B30 (new); D14 |
| Step 10: B15 to B20 keep in Q2 | accepted | Cheaper after B7 and B6 | none |
| B5 over-engineered | accepted | After B7 the reader only checks annotations; 17 `set_analysis_data` calls go | B5 (redefined) |
| Missing: typed payloads | accepted | `egraph.rs:10-28,63-94` | B31 (new) |
| Missing: rule scheduling | accepted | Bare-variable rules (`ir2ir_rewrites.txt` lines 3, 4, 5, 8) end saturation at `NODE_LIMIT` (`lib.rs:20`) | B32 (new) |
| Missing: staffing for 3-4 engineers | accepted | Owner decision 11 | overhaul.md lanes; plan/README.md |
| Keep C++ allocator | accepted | `cpp/malloc/include/solver.h:26-63` | none |
| Rewrite from scratch rejected except dispatch and extractors | accepted | `applier.rs:15-58`; `fast.rs`, `slow.rs` keep only the (class, buffer) state and drop reasons | overhaul.md; decisions.md |
| Lose: B10 candidate order keeps cost-blind selection | accepted | The B3 gate pins the new order once | B10 |
| Lose: B4 could delete `hold_tensors` | accepted | `parser.cc:335` | B4 |
| Lose: B2 `--log` printer must stay byte-compatible | accepted | `tests/test_extraction.py:109,135,496` read those files | B2 |
| Revised order table | partly | Adopted with B30 in lane B Q1 and B5, B12, B23 grouped; B12 depends on B6 | overhaul.md |
| Target: `search` returns `Result`; deterministic mode | accepted | `pipeline.rs:155-180` | B29 |
