# Architecture decisions

Each row is a decision in force for the backend plan, with its evidence.
Rows marked `panel` wait on the debate panels that settle the tech-lead proposals.

| Topic | Decision | Evidence | Items |
| --- | --- | --- | --- |
| Test gate | Deterministic search mode first; gate on identical chosen assembly and candidate files, plus goldens | Search is wall-clock bounded (`pipeline.rs:155-180`, `lib.rs:18`); goldens are simulated outputs (`README.md:109`) | B29, B3 |
| Behavioral gate | Oracle-differential run over the extraction and matching specs | Two goldens of one accelerator family are a thin net; the oracle is in `tests/conftest.py` | B3 |
| Two parsers | Delete the C++ candidate parser behind a typed FFI; keep `hold_tensors`; the reader checks annotations | `act_malloc.cc:45-66` re-reads Rust's text; `parser.cc:146,335` is allocator logic | B2, B4, B5 |
| E-graph order | Typed payloads, then shape in `make`, then no clone, then rewrite values | B6 first would regenerate `set_shapes_N` that B7 deletes (`applier.rs:43-56`) | B31, B7, B9, B6 |
| Rewrite form | One value per rule, variables bound by name; code or data form open | `ir2ir_rewrites.rs:78-85,103-135` index positions; panel (R19) | B6 |
| Payload matching | Payloads never match in patterns; conditions carry them | `Language::matches` compares discriminants only (`egraph.rs:163-165`) | B31, B6, B8 |
| Constants | Folded value in class data, no node in the e-graph | `pipeline.rs:234-240` clones because `slow.rs:24` mutates | B9 |
| Extractors | One cost-guided extractor keeping the best few candidates | Default cost counts `api.` lines (`count.cc:17`); fast is cost-blind (`fast.rs:88-128`) | B10 |
| In-place | Output and marked input share storage, as a solver equality | Owner decision 17; `constraint.cc:214-230` only permits overlap | B30, B11 |
| Rule scheduling | IR-to-IR expansion and selection rules in separate phases | Bare-variable rules end saturation at `NODE_LIMIT` (`ir2ir_rewrites.txt`, `lib.rs:20`) | B32 |
| Engine | Keep egg | Phases need separate `Runner`s only; no egglog feature is required | B32 |
| Templates | No layout twins; no formatter run at generation | No gate checks the generated layout (`tests/lint.py:19,27`, `README.md:110`) | B12 |
| Allocator language | Keep C++ and OR-Tools; typed FFI only | `cpp/malloc/include/solver.h:26-63` | B2, B4 |
| Rewrite from scratch | Rejected, except the rewrite dispatch and the extractors | Each stage's logic is correct as far as read | all |
| Overlapping IR-to-ISA rules | panel (R22) | -- | -- |
