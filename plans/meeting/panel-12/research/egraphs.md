# E-graphs: libraries, e-matching, and compiler use

Evidence for Q12.3 and owner decision 45: what e-graph systems exist, how they work, what they cost, and what the current backend uses.
The brief takes no position; each bullet ends with its source, a URL or a file:line.
Backend paths are relative to `act-backend/act_backend/generic/src/`.

## 1. egg

- egg is a Rust e-graph library whose two named contributions are deferred rebuilding and e-class analyses (https://arxiv.org/abs/2004.03082).
- Deferred rebuilding restores congruence once per iteration instead of after every merge (https://arxiv.org/pdf/2004.03082, Section 3).
- Measured over egg's tests, deferred rebuilding made congruence closure 88x faster and whole equality saturation 21x faster by geometric mean (https://arxiv.org/pdf/2004.03082, Section 3.4).
- An e-class analysis has `make` for a new e-node, `join` for merged classes, and an optional `modify` that may add e-nodes (https://arxiv.org/pdf/2004.03082, Section 4.1).
- The paper asks that the domain with `join` form a join-semilattice (https://arxiv.org/pdf/2004.03082, Section 4.1).
- The Rust trait states only `DidMerge` flag rules for `merge`, not a semilattice law (https://docs.rs/egg/latest/egg/trait.Analysis.html).
- The trait also offers `pre_union`, `remake`, and `allow_ematching_cycles` hooks (https://docs.rs/egg/latest/egg/trait.Analysis.html).
- Patterns compile to a small e-matching virtual machine after de Moura and Bjorner 2007 (https://arxiv.org/pdf/2004.03082, Section 5.3).
- Conditional rewrites go through `ConditionalApplier`, which checks the condition just before `apply_one` (https://docs.rs/egg/latest/egg/struct.ConditionalApplier.html).
- The default scheduler is `BackoffScheduler` (https://docs.rs/egg/latest/egg/struct.Runner.html).
- It bans a rule whose matches exceed a limit, default 1,000, for 5 iterations, and doubles both on each ban (https://docs.rs/egg/latest/egg/struct.BackoffScheduler.html).
- Runner defaults are 30 iterations, 10,000 e-nodes, and 5 seconds (https://docs.rs/egg/latest/egg/struct.Runner.html).
- The built-in `Extractor` takes a local cost function and picks the cheapest e-node per class by fixed point (https://arxiv.org/pdf/2004.03082, Section 4.3).
- Explanations of equalities are opt-in through `with_explanations_enabled` (https://docs.rs/egg/latest/egg/struct.EGraph.html).
- Reported case studies:
  - Herbie: the egg backend is over 3000x faster than Herbie's first simplifier, reached through a C-level FFI (https://arxiv.org/pdf/2004.03082, Section 6.1).
  - Spores: derives all 84 hand-written SystemML sum-product rules and finds rewrites worth 1.2x to 5x end to end (https://arxiv.org/pdf/2004.03082, Section 6.2).
  - Szalinski: moving from a custom OCaml e-graph to egg made the tool about 1000x faster, anecdotally (https://arxiv.org/pdf/2004.03082, Section 6.3).

## 2. egglog

- egglog joins Datalog with equality saturation: incremental execution, cooperating analyses, and lattices, plus congruence closure and extraction (https://arxiv.org/abs/2304.04332).
- A function declares a `:merge` expression that resolves two outputs for one input, for example `min` for a lattice (https://arxiv.org/pdf/2304.04332, Section 3.2).
- With `:merge` as union this is congruence closure; with other `:merge` it acts like analysis propagation (https://arxiv.org/pdf/2304.04332, Section 5.1).
- Conditions are ordinary query atoms, so egglog has no separate conditional rewrite (https://arxiv.org/pdf/2304.04332, Section 5.2).
- The paper contrasts this with egg, whose guards are opaque Rust that run after the query (https://arxiv.org/pdf/2304.04332, Section 5.2).
- Rules are statically typechecked, unlike egg rewrites (https://arxiv.org/pdf/2304.04332, Section 5.2).
- The query engine uses relational e-matching with Generic Join, plus semi-naive evaluation (https://arxiv.org/pdf/2304.04332, Section 5.1).
- Semi-naive evaluation keeps a delta per iteration so old facts are not matched again (https://arxiv.org/pdf/2304.04332, Section 4.3).
- On egg's math suite without analyses, under the backoff scheduler, results after 100 iterations were:
  - egglog without semi-naive grows the same e-graph as egg 3.34x faster (https://arxiv.org/pdf/2304.04332, Section 5.3).
  - Full egglog grows a slightly larger e-graph 9.27x faster than egg (https://arxiv.org/pdf/2304.04332, Section 5.3).
- egglog computed a points-to analysis 4.96x faster than the fastest sound Souffle encoding (https://arxiv.org/pdf/2304.04332, Section 6.1).
- Release 2.0.0 is the first on a "database-first, highly parallel" backend (https://docs.rs/crate/egglog/2.0.0/source/CHANGELOG.md).
- In 3.0.0, thread count is set per e-graph through a scoped pool (https://docs.rs/crate/egglog/3.0.0/source/CHANGELOG.md).
- The README says most users run single-threaded, and the `-j` thread count defaults to 1 (https://docs.rs/crate/egglog/latest/source/README.md).
- The README points to egglog-experimental for extra features as plugins (https://docs.rs/crate/egglog/latest/source/README.md).

## 3. E-matching algorithms

- Backtracking e-matching walks a pattern top-down and tries every e-node of each class (https://arxiv.org/abs/2108.02290).
- Z3's Matching Abstract Machine is a production instance of compiled backtracking matching (https://raw.githubusercontent.com/Z3Prover/z3/master/src/smt/mam.h, line 30).
- Relational e-matching turns a pattern into a conjunctive query over one relation per operator (https://arxiv.org/abs/2108.02290).
- With Generic Join the paper claims worst-case optimal run time, bounded by e-graph size and output size (https://arxiv.org/abs/2108.02290).
- On egg's math and lambda suites, at 1e5 to 3e6 e-nodes, Generic Join was over six orders of magnitude faster on complex patterns (https://arxiv.org/pdf/2108.02290, Section 5.2).
- Speedup grows as the pattern grows and turns non-linear, and as the output shrinks (https://arxiv.org/pdf/2108.02290, Section 5.2).
- Dense patterns with large outputs gain little or no speedup (https://arxiv.org/pdf/2108.02290, Section 5.2).
- Results are given with and without building the index tries (https://arxiv.org/pdf/2108.02290, Section 5.1).
- That prototype rebuilds a database from the e-graph at each match; egglog avoids the copy by being the database (https://arxiv.org/pdf/2304.04332, footnote 6).
- The paper noted no reusable Generic Join implementation at its writing (https://arxiv.org/pdf/2108.02290, footnote 6).

## 4. E-graphs inside compilers

| System | Host | E-graph engine | Saturates | Extraction | Source |
|---|---|---|---|---|---|
| Cranelift aegraph | Rust compiler mid-end | Own acyclic e-graph, ISLE rules | No, rewrites eagerly on insert | One-pass DP ignoring sharing | https://cfallin.org/blog/2026/04/09/aegraph/ |
| TENSAT | Tensor graph superoptimizer | egg | Yes, with limits | ILP via SCIP in OR-tools | https://arxiv.org/pdf/2101.01332 |
| eqsat dialect | xDSL, mirrors MLIR | E-classes as IR operations | Yes | Not described | https://arxiv.org/html/2505.09363v2 |
| DialEgg | MLIR, C++ | egglog, external | Yes | egglog | https://www.cs.mcgill.ca/~azayed2/files/dialegg/DialEgg_EGRAPHS.pdf |
| Glenside | Tensor IR for hardware mapping | egg | Yes | Optional CPLEX ILP, no longer in core | https://www.github.com/gussmith23/glenside |
| Szalinski | CAD decompiler | egg | Yes | egg | https://arxiv.org/pdf/2004.03082 |

- Cranelift:
  - Pure operators float in a sea of nodes beside a CFG skeleton, and scoped elaboration puts them back (https://cfallin.org/blog/2026/04/09/aegraph/).
  - Equivalences are union nodes in the SSA value space, with no parent pointers or rebuild (https://cfallin.org/blog/2026/04/09/aegraph/).
  - The aegraph pipeline costs about 7-8% more compile time than a classical one and yields about 2% speedup (https://cfallin.org/blog/2026/04/09/aegraph/).
  - An earlier egg-based prototype began at over 2x overhead (https://cfallin.org/blog/2026/04/09/aegraph/).
  - The average e-class holds 1.13 e-nodes, and the rule set excludes associativity and commutativity (https://cfallin.org/blog/2026/04/09/aegraph/).
- TENSAT:
  - Finds graphs up to 16% faster than the prior state of the art, with on average 48x less optimization time (https://arxiv.org/abs/2101.01332).
  - Supports multi-pattern rules, rewrites with more than one matched output (https://arxiv.org/pdf/2101.01332, Section 4).
- eqsat dialect:
  - Defines `eqsat.egraph`, `eqsat.eclass`, and `eqsat.yield`; matches with PDL lowered to `pdl_interp` (https://arxiv.org/html/2505.09363v2).
  - A custom interpreter backtracks over e-class members, and CSE keeps congruence, which the authors call possibly less efficient than rebuilding (https://arxiv.org/html/2505.09363v2).
  - The workshop paper reports no benchmark numbers (https://arxiv.org/html/2505.09363v2).
  - A follow-up paper presents e-graphs as a persistent abstraction across IR levels in xDSL (https://arxiv.org/abs/2602.16707).
- DialEgg:
  - Published at CGO 2025 by Zayed and Dubach (https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog).
  - MLIR types map to a `Type` datatype, attributes to an `Attr` datatype, and each operation carries its type (https://www.cs.mcgill.ca/~azayed2/files/dialegg/DialEgg_EGRAPHS.pdf, slides "MLIR to Egglog").
  - Saturation took 8.7 ms on a 3-matmul chain and 4939.3 ms on an 80-matmul chain of 162 ops (https://www.cs.mcgill.ca/~azayed2/files/dialegg/DialEgg_EGRAPHS.pdf, "Compile Time and Scalability").
  - Translating MLIR to egglog took 7.3 ms or less on those benchmarks (same slide).

## 5. Existing C++ e-graph code

- Z3 has a C++ e-graph, `euf::egraph`, in `src/ast/euf/euf_egraph.h` (https://raw.githubusercontent.com/Z3Prover/z3/master/src/ast/euf/euf_egraph.h, line 85).
- Z3's e-graph stores justifications on merges and theory-variable hooks for SMT (same file, lines 78-98 and 115-157).
- Luau shipped a header-only C++ template `EGraph` with union-find `find`, analysis `make` and `join`, and `rebuild` (https://raw.githubusercontent.com/luau-lang/luau/0.700/EqSat/include/Luau/EGraph.h, lines 19-50 and 124-243).
- That header has no e-matching or extraction code; matches for those words were none and a FIXME (same file, lines 365-369).
- `EqSat/include/Luau/EGraph.h` exists at Luau tag 0.701 and is absent from tag 0.702 and from master (raw.githubusercontent.com probes, HTTP 200 then 404).
- A C++ port inspired by egg is on Gitee, Apache-2.0, 0 stars, last updated December 2023 (https://gitee.com/klose28/egraph).
- A GitHub search for C++ repositories on "equality saturation" returned two repositories, both with 0 stars (`gh search repos`, run 2026-11-05).
- One is MLEgg, "Applying Equality Saturation to MLIR", last updated January 2025 (https://github.com/saralahlou1/MLEgg).
- DialEgg is C++ around MLIR but runs egglog, a Rust engine, for saturation (https://www.cs.mcgill.ca/~azayed2/files/dialegg/DialEgg_EGRAPHS.pdf).
- No search in this brief found a maintained, general C++ equality-saturation library with e-matching and extraction.

## 6. Metadata-heavy tensor rewrites and their preconditions

- TENSAT stores shape, layout, and split locations in egg analysis data and checks shapes before applying a match, as TASO does (https://arxiv.org/pdf/2101.01332, Section 4 and the implementation section).
- TENSAT represents shapes and permutations as strings in e-nodes, and tensor identifiers as `name@dims` (https://arxiv.org/pdf/2101.01332, Section 4).
- Spores composes three analyses: schema, a conservative cost estimate, and constant folding (https://arxiv.org/pdf/2004.03082, Section 6.2).
- Glenside's access pattern gives each tensor two shapes, iterated dimensions and computed dimensions (https://ar5iv.labs.arxiv.org/html/2105.09377).
- In DialEgg a matmul associativity rule reads row and column counts from types inside the rule's query (https://www.cs.mcgill.ca/~azayed2/files/dialegg/DialEgg_EGRAPHS.pdf, "MM Associativity" slide).
- Cost of a precondition differs by design:
  - In egg a guard runs after the full match, so every match is found first and then filtered (https://arxiv.org/pdf/2304.04332, Section 5.2).
  - In egglog conditions sit in the query, so the planner can join on them (https://arxiv.org/pdf/2304.04332, Section 5.2).
- The egg analysis framework propagates data bottom-up only, which one paper calls a bar to type analysis (https://arxiv.org/pdf/2501.02413).
- Rules that create cycles force cycle handling in extraction; TENSAT filters cycles during exploration (https://arxiv.org/pdf/2101.01332, Section 5.2).

## 7. Extraction methods

- Optimal extraction is NP-hard and hard to approximate within any constant ratio (https://cse.hkust.edu.hk/~parreaux/publication/oopsla24a/).
- Greedy:
  - Tree-cost greedy is optimal for a local cost but ignores sharing (https://arxiv.org/pdf/2004.03082, Section 4.3).
  - Cranelift uses a one-pass DP that ignores shared substructure, and alternatives tried did not improve its tradeoff (https://cfallin.org/blog/2026/04/09/aegraph/).
  - Greedy DAG extraction counts sharing but is not optimal (https://inst.eecs.berkeley.edu/~cs294-260/sp24/projects/mcts-extraction).
- ILP:
  - TENSAT's ILP with cycle constraints timed out without a feasible solution at `kmulti = 2` (https://arxiv.org/pdf/2101.01332, Table 5).
  - Dropping the cycle constraints sped ILP solving up about 10x to 1000x on larger e-graphs (https://arxiv.org/pdf/2101.01332, Table 5).
  - TENSAT reports greedy extraction never picks its split nodes, so it misses some rewrites (https://arxiv.org/pdf/2101.01332, Table 4).
- Parameterized: extraction is tractable at low treewidth, in time 2^O(w^2) poly(w, n) (https://arxiv.org/pdf/2408.17042).
- Other methods:
  - SmoothE, differentiable, reports ILP-like quality in seconds on GPUs (https://www.csl.cornell.edu/~zhiruz/pdfs/smoothe-asplos2025.pdf).
  - e-boost reports 558x over ILP and 19.04% better than the prior framework, per a summary (https://aitopics.org/doc/arxivorg:E77047CF).
  - Answer set programming encodings are compared against ILP and faster-greedy-dag (https://arxiv.org/pdf/2606.10644).
- k-best: no e-graph-specific k-best extraction paper turned up.
- The nearest k-best result is for valued and-or DAGs, in O(E k log k + E log Ev + V k log Ev) (https://dspace.mit.edu/handle/1721.1/41540).

## 8. What the current backend uses

- Engine: egg 0.10.0 (`../Cargo.toml`, line 12).
- Language:
  - `TensorOp` is one enum of IR operators plus generated ISA variants, marked `{{ISA_ENUM_VARIANTS}}` (ir/egraph.rs:6-32).
  - Op metadata (dims, shape, slice, perm, dtype, constant value) is a `String` payload inside the e-node (ir/egraph.rs:12-28).
  - Payload integers are parsed from text at each check (ir/egraph.rs:62-109).
  - `matches` compares only the variant, so patterns ignore payloads (ir/egraph.rs:162-165).
- Analysis:
  - `TensorInfo` is shape, dtype, and `is_const` per class, and its equality ignores `is_const` (ir/egraph.rs:370-391).
  - `make` sets no shape or dtype, only `is_const` (ir/egraph.rs:396-401).
  - `merge` panics when two set shapes or dtypes differ, ORs `is_const`, and returns `b_merged = false` (ir/egraph.rs:404-415).
  - The egg contract asks `b_merged` to be true when `b` differs from the merged `a` (https://docs.rs/egg/latest/egg/trait.Analysis.html).
  - Shapes on RHS classes come from `set_analysis_data`, called 22 times in ir2ir_rewrites.rs (isel/rewrites/applier.rs:127).
- Rewrites:
  - Ten IR-to-IR rules are text; ISA rules are a template, `{{IR2ISA_REWRITES}}` (isel/rewrites/ir2ir_rewrites.txt:1-10, isel/rewrites/ir2isa_rewrites.txt:1).
  - Each rule has three generated functions: precondition, RHS metadata, and RHS shapes (isel/rewrites/applier.rs:15-56 and 62-66).
  - A precondition reads shape and dtype of every matched class (isel/rewrites/ir2ir_rewrites.rs:9-56).
  - The custom applier re-matches the LHS by its own backtracking to recover matched e-nodes and payloads (isel/rewrites/applier.rs:106-113 and 167-215).
  - Several rules introduce identities on any class, such as `(?x) => (copy ?x)` (isel/rewrites/ir2ir_rewrites.txt:3-5 and 8).
  - All 18 `{{...}}` template markers in the Rust sources are filled by Python generators (`act_backend/*_generator.py`).
- Run loop:
  - The Runner uses egg's default scheduler, so `BackoffScheduler` (pipeline.rs:491-495).
  - Limits are 1000 ms, 10,000 nodes, and 30 iterations (lib.rs:17-22).
  - Extraction runs from the Runner hook every `N = 1` iterations (pipeline.rs:496-501, lib.rs:24).
- Extraction:
  - Fast extraction yields one graph, each class once per buffer, by depth-first search with rollback (isel/extractor/fast.rs:12 and 86-114).
  - Its selection map is a `BTreeMap` so e-node order keeps numbering stable (isel/extractor/fast.rs:24-25).
  - Slow extraction enumerates repeat-free graphs up to `limit` e-nodes, then merges equal subtrees by value numbering (isel/extractor/slow.rs:13 and 164-165).
  - Slow extraction starts at 10 e-nodes, grows by one per pass, and stops above 15 (lib.rs:26-28, pipeline.rs:499-501, isel/extractor/lib.rs:39).
  - Both extractors first add `DetectedConst` e-nodes for constant classes, mutating the e-graph (isel/extractor/fast.rs:19-22, isel/extractor/utils.rs:44-60).
  - Candidates are ranked by an external C cost model on emitted assembly, not by a cost function in the e-graph (cost/c.rs:5-8, pipeline.rs:93).
- Buffer placement:
  - Placement is chosen in extraction: each instruction maps to output and input buffers, with `HBM`, generated buffers, and `ANY` (ir/buffer.rs:4-8 and 20-30).
  - Extraction state is keyed by class and buffer (isel/extractor/fast.rs:50-55, isel/extractor/slow.rs:71-75).
  - In-place e-nodes are tried last, and a later pass copies shared overwritten operands (isel/extractor/fast.rs:78-85, isel/extractor/inplace_fix.rs:6-9).
- Not used: egg explanations; a search for `explan` in the backend sources finds nothing.
