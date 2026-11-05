# Architecture decisions

Each row is a decision in force for the backend plan, with its evidence and the panel question or owner decision that settles it.

| Topic | Decision | Evidence | Settled by | Items |
| --- | --- | --- | --- | --- |
| Test gate | Deterministic search mode first and the only test mode; chosen assembly and `--log` files gate structural steps | Search is wall-clock bounded (`pipeline.rs:155-180`, `lib.rs:18`); goldens are simulated outputs (`README.md:109`) | Q4.2 | B29, B3 |
| Behavioral gate | Oracle-differential run over the extraction and matching specifications; no backend program faults on the oracle | Two goldens of one accelerator family are a thin net; the oracle is in `tests/conftest.py` | Q4.2, Q1.12 | B3 |
| User search mode | Wall-clock bound stays the user default; a hit bound may vary the choice, never to a wrong kernel | `TIME_LIMIT` 1000 ms (`lib.rs:18`) | owner 27 | B29, B34 |
| Two parsers | Delete the C++ candidate parser behind a typed FFI; keep `create_instruction`, `hold`, `hold_tensors`; the reader checks annotations | `act_malloc.cc:45-66` re-reads Rust's text; `parser.cc:146,335` is allocator logic | Q4.11 | B2, B4, B5 |
| FFI schema | The Rust `#[repr(C)]` struct is the schema; cbindgen derives the C header | cbindgen is absent from `Cargo.toml` today | Q4.11, owner 28 | B2 |
| E-graph order | Typed payloads, then shape in `make`, then no clone, then rewrite values | B6 first would regenerate `set_shapes_N` that B7 deletes (`applier.rs:43-56`) | Q4.3 | B31, B7, B9, B6 |
| Rewrite form | IR-to-IR hand-written named Rust; IR-to-ISA generated Rust bound by name; one shared re-match helper; a two-rule data-form spike may overturn it | `applier.rs:15-58` binds by number; egg's `Subst` holds no payloads | Q4.4 | B6 |
| Payload matching | Payloads never match in patterns; conditions carry them | `Language::matches` compares discriminants only (`egraph.rs:163-165`) | Q4.3 | B31, B6, B8 |
| Constants | Folded value in class data, no node in the e-graph | `pipeline.rs:234-240` clones because `slow.rs:24` mutates | Q4.3 | B9 |
| Shape faults | A disagreeing declared and computed dim is the shared coded shape fault, not a panic | `egraph.rs:403-409` panics | Q3.6, Q3.11 | B26 |
| Front-end checks | The backend entry takes a `Model` or calls `check` once, reads only the `Model`, and keeps no parallel checker | `check_regions` handles literals only (`spec_checks.py:413-414`) | Q3.2, Q4.8, Q11.3, owner 42 | B21, B7 |
| Backend checks | Plain functions after `check`: an unknown key in the backend's `meta` namespace rejects; an empty usable set rejects | A registry or checker base class is abstraction before a second use | Q11.3, owner 42 | B14 |
| Reject versus exclude | A problem the architect can correct rejects; an instruction the backend cannot model (state, other than one output, a size over an addressing attribute, a constraint the allocator cannot hold, an unsupported word) is excluded with a reason | `check_state`, `spec_checks.py:370-386`, `:22-29`, `:277-290` reject a valid ISA the architect cannot correct | Q11.3, owner 40 | B14, B46 |
| Exclusion report | Code, section id, and author line, printed at generation and stored in the crate; a kernel with no covering stops at kernel compile (`K`), listing excluded instructions as a hint | Silent dropping is the worst outcome for the architect | Q11.3 | B46, B42 |
| Extractors | One k-best extractor on (class, buffer); enumerator test-only | Default cost counts `api.` lines (`count.cc:17`); fast is cost-blind (`fast.rs:88-128`) | Q4.5 | B37, B10 |
| Overlapping IR-to-ISA rules | Allowed; cost chooses; ties by declaration index; exact duplicates warned | Fixture declares overlap on purpose (`tests/test_extraction.py:405-414`) | Q4.1 | B35 |
| Unwritten storage | A candidate reading unwritten on-chip storage of a buffer with no declared initial value is dropped with a reason; no zeroing; a declared value may be read before any write | Every read in a selected graph has a producer except partial writes and reuse | Q4.9, Q1.4, joint decision on unwritten reads | B36 |
| Addressing attributes | The solver domain of an addressing attribute is its declared width-layer range, `s64` by default; bounds apply to the evaluated address; no meaning keyed on a name | `malloc_instructions_h_generator.py:456` bounds attributes at 0 | owner 34, 38; panel 10 Q10.1 | B38 |
| Integers | Checked i64 per intermediate; overflow is a defined error | `expression_writer.py:71-96` | Q4.9 | -- |
| State | Read-only `@s.` folds at compile time in v3; updates after v3; an instruction with any other use is excluded with a reason | `spec_checks.py:249-251`; `generator.py:119-122` | Q4.7, Q11.3 | B24, B16 |
| Rule scheduling | Guards on bare-variable rules; phases only if node-limit stops remain | Bare-variable rules end saturation at `NODE_LIMIT` (`ir2ir_rewrites.txt`, `lib.rs:20`) | Q4.10 | B32 |
| Engine | Keep egg until B52; no egglog; `SimpleScheduler` once guards end node-limit stops | `BackoffScheduler` already in force (`pipeline.rs:491-495`) | Q4.10, Q12.3 | B32, B52 |
| Templates | No layout twins; rustfmt runs on the generated crate at generation | `template_loader.py:8-14` re-implements rustfmt layout | Q4.12, owner 29 | B12 |
| Allocator language | Keep C++ and OR-Tools; typed FFI only | `cpp/malloc/include/solver.h:26-63` | Q4.11 | B2, B4 |
| Rewrite from scratch | Rejected, except the rewrite dispatch and the extractors | Each stage's logic is correct as far as read | Q4.4, Q4.5 | all |
| Pair relation | Each same-buffer input-output pair has one relation (tied, apart, exact-or-apart, may overlap) from one record; the default is exact-or-apart | A touching pair has no constraint and only the tie exempts an overlapping pair (`constraint.cc:206-230`) | Q9.3, Q9.4, owner 17, 31 | B39, B30 |
| Touch exception | The allocator applies the pair's relation where live ranges touch; overlapping ranges keep their edge; tied is an equality | `[a,b]` and `[b,c]` only touch (`constraint.cc:206`) | Q9.4 | B30 |
| Copy pass | Reads the tie only; copies a shared tied input; errors on an HBM-pinned or shared-view operand | `inplace_fix.rs:6-17,59-80` | Q9.4 | B11 |
| Default pair limit | If exact-or-apart overloads the solver, default pairs may be allocated apart; a backend limit, never a language rule | Apart is stricter and never faults | Q9.4 | B30 |
| Derived attributes | The allocator solves the encoded attribute with its type domain; a derived attribute is a closed-form expression of it, never a solver variable | `solver.h:26-33` floor division; `malloc_instructions_h_generator.py:510` routes address constraints | panel 10 Q10.3 | B44 |
| Assembler and disassembler | One generator from the encoding layer, landing last with the block, the derived rule, and a round-trip test; the backend keeps printing text | `str_piece.txt:1`; `emitter.cc:507-509` | panel 10 Q10.2, Q10.4 | B45 |
| Failure policy | A candidate-level failure is a dropped candidate with a recorded reason; the backend stops only on an invalid input kernel, one outside the input specification, or an unwritable path | `pipeline.rs:95-102` exits on a cost failure; `act_malloc.cc:182` already returns reasons | owner 40 | B41, B42 |
| Input formats | XLA HLO text only today; other formats are detected and named; StableHLO portable artifacts are the supported MLIR input, read by B47; torch inputs enter through B53 | `isel/initializer/lib.rs:36`; `pipeline.rs:312-318` | owner 40, Q12.1, Q12.2 | B43, B47, B53 |
| Port order | Importer as a text bridge, then dialect and in-house library, then the engine swap through the B2 C ABI, then one C++ driver; front end stays Python; MLIR only in the importer, dialect, and bridge | MLIR has no stable C++ API; the reader takes HLO text only (`isel/initializer/lib.rs:36`) | owner 43, Q12.1 | B47 to B56 |
| Dialect form | One generic instruction op by default; per-ISA ODS only if it builds from U15's prebuilt package with no further install | No consumer needs per-ISA op classes | Q12.1 | B48, B50 |
| Torch producer | torch-mlir when its latest release imports the latest released torch's `ExportedProgram`, else an ACT-owned walker; strict by default | torch-mlir's CI pins a torch nightly | owner 44, Q12.2 | B53 |
| E-graph library | In-house C++, no MLIR header; backtracking matcher on per-operator storage; extraction stays in the backend; three-tier swap gate | `merge` panics on disagreement (`ir/egraph.rs:403-414`); `BackoffScheduler` in force (`pipeline.rs:490-495`) | owner 45, 46, Q12.3 | B49, B51, B52 |
| Versions | act-backend 2.0 in the v3 release; released pins only | -- | owner 26, 33 | all breaking items |
