# Architecture decisions

This session had no sub-agent tool, so the track lead argued the three panel positions in turn and adjudicated.
Each topic lists the positions, the verdict and its rationale.

## Panel

| Role | Stance |
| --- | --- |
| Incremental simplifier | Smallest safe deletions and merges; byte-identical output at every step. |
| Overhaul architect | Target layout from scratch: one typed model per stage, no text between stages. |
| Outside critic | Compare with egg and egglog practice and judge both. |

## Why two parsers

- Simplifier: the C++ `parser.cc` re-reads the candidate text Rust wrote (`act_malloc.cc:45-66`); keep it but document the format.
- Architect: the allocator is linked into the same process (`generic/src/malloc/cpp.rs:15-24`); pass a struct and delete the parser.
- Critic: an in-process text round trip is the weakest boundary in the system; the kernel reader is a real second parser and is fine.
- Verdict: delete the C++ candidate parser behind a typed FFI (B2, B4); reshape, not replace, the Rust kernel reader (B5).
- Rationale: one format with two hand-written ends is the source of D10 and half of D5.

## Rewrites and applier

- Simplifier: give rewrites names, keep the text files.
- Architect: generate Rust rewrite values; drop the run-time pattern parse and number dispatch (`applier.rs:15-58`).
- Critic: egg's `rewrite!` and `Applier` exist for this; the string table is a re-implementation with weaker checks.
- Verdict: replace the dispatch (B6), keep every pattern and condition verbatim, then name and test (B8).
- Rationale: the replace is justified because the current form turns a numbering mistake into a start-up panic (D6) and nothing of the dispatch code is worth keeping.

## E-graph analysis

- Simplifier: leave `set_shapes_N`; they work.
- Architect: infer shape and type in `Analysis::make`, as egg intends.
- Critic: per-rewrite shape setting is unusual and is why every rewrite needs three functions; but it is the riskiest change.
- Verdict: do it (B7), after the structural steps and the golden harness (B3), in Q2.
- Rationale: it deletes most generated per-rewrite code; the risk is caught by `merge` and goldens.

## Extractors: slow and fast

- Simplifier: merge shared code, keep both strategies.
- Architect: one enumeration with a strategy parameter.
- Critic: cost is known only after allocation and the cost model, so egg's `Extractor` with a `CostFunction` does not fit; enumerating candidates is justified, duplicated code is not.
- Verdict: merge into one module with two strategies (B10); keep enumeration.

## Pipeline

- All agree: split CLI from search, keep limits and stop reasons, stop cloning the e-graph per pass (B9).

## spec_checks and generators

- Simplifier: rename and leave structure.
- Architect: checks build one instruction model; emitters read it; no check imports an emitter (today `spec_checks.py:9-18`).
- Critic: the formatter-twin templates are an unusual cost; generated code is normally formatted after emission.
- Verdict: one model (B13, B14); replace twin templates with a formatter run (B12), pending the clang-format question.

## Replace from scratch

- Rejected for every module: each stage's logic is correct as far as read, and the goldens make byte-identical refactoring possible.

## Round two: contested points

### Generate code, or generate data

- Architect: emit one table per instruction (pattern, payload, conditions as expression trees) and keep one generic Rust applier and one generic C++ allocator class that interpret it; most generator modules and templates disappear.
- Simplifier: the generated conditions today are straight-line Rust (`isel_ir2isa_rewrites_rs_generator.py:565-627`); an interpreter is a new component with its own bugs, and goldens would not catch slow-downs.
- Critic: egglog interprets rules from data and is fast enough; the expression trees here are small; but an interpreter changes overflow behavior unless it reuses the checked arithmetic of `operand_sizes.py`.
- Architect answers: the interpreter reuses one checked-arithmetic function; that is less code than three writers.
- Simplifier answers: then do it after B6, when the code is already one value per rewrite, and measure.
- Verdict: B6 generates typed Rust first (Q1); a data-driven applier is a Q2 experiment gated on equal goldens and compile time within 10 percent, recorded as a follow-up of B6, not a commitment.

### Typed FFI, or port the allocator to Rust

- Architect: one language would remove the FFI and CMake step (`generic/build.rs:51-68`).
- Critic: the allocator is built on OR-Tools' C++ constraint solver (`cpp/malloc/include/solver.h:26-63`); Rust bindings would be a new dependency and a rewrite of working code.
- Simplifier: the C++ beyond `parser.cc` is the most solid part of the backend.
- Verdict: keep C++; typed FFI only (B2, B4).

### Shapes in the analysis now, or later

- Simplifier: B7 is the one high-risk step; defer until coverage work needs it.
- Architect: every new operation (B17) without B7 costs three functions per rewrite.
- Critic: B7 is standard practice and `merge` already checks agreement, so failures are loud.
- Verdict: B7 in Q2, before B17 and B18, after all structural steps.

### Formatter run, or layout templates

- Simplifier: twin templates work and `tests/lint.sh` proves it.
- Architect: they are the largest single source of generator code.
- Critic: emitting then formatting is the norm; the only cost is `clang-format` at generation time.
- Verdict: B12 for Rust now (rustfmt ships with the cargo toolchain generation already needs); C++ templates stay until the owner rules on `clang-format`.
