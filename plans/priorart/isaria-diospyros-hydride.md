# Isaria, Diospyros and Hydride

## What it is

- Diospyros (ASPLOS 2021) symbolically evaluates a fixed-size kernel with Rosette into a scalar DAG, then runs equality saturation in egg with hand-written rules for vector ops, shuffles and multiply-accumulate, extracts by a cost model and emits Tensilica DSP intrinsics.
- Isaria (ASPLOS 2024) replaces the hand-written rules: it synthesizes candidate rewrite rules from the ISA's semantics with an extended Ruler, classifies them into coarse phases (expansion, compilation, optimization) by an abstract cost model, and applies the phases in sequence at compile time; it reports up to 6.9x over DSP libraries.
- Hydride (ASPLOS 2024) parses vendor pseudocode for x86, HVX and ARM into formal semantics, groups similar instructions into parameterized AutoLLVM IR ops (3557 ISA ops to 397), synthesizes a code generator with SyGuS under a latency-sum cost model, and reports 12 percent to 100 percent over Halide's LLVM backend.
- Sources: https://github.com/cucapra/diospyros, https://cs.utexas.edu/~sgt/papers/compgen.pdf, https://hydride.cs.illinois.edu, https://hydride.cs.illinois.edu/files/2024/05/Hydride.pdf.

## TAIDL today, for comparison

- ACT's `ir2isa` rules are generated from TAIDL semantics, the Hydride and Isaria direction, while `ir2ir` expansion rules are a hand-written list of 10 (act-backend/act_backend/generic/src/isel/rewrites/ir2ir_rewrites.txt:1-10).
- All rules run in one `Runner` with one node, iteration and time limit (act-backend/act_backend/generic/src/pipeline.rs:491-495); Isaria's point is that phasing is what makes saturation tractable.
- Cost is an external whole-program C function (act-backend/act_backend/generic/src/cost/c.rs:6-24); Diospyros and Hydride use per-op costs during extraction.

## Borrow

- Phased saturation: expansion rules (`slice` through `convert`, `reshape` pairs, `broadcast` removal) to a bound, then mapping rules, then cleanup; each phase with its own stop reason.
- Rule provenance and tests: Isaria's synthesized rules are each validated; every `ir2ir` rule should carry a name and a test that checks it against the oracle on random shapes.
- Parameterized instruction families: Hydride's similarity grouping is what TAIDL's `@c.` attributes already express; a generator should treat an instruction with attributes as one family, not one rule per attribute value.
- Per-op cost for extraction: derive a per-e-node estimate from the C cost model or from a declared per-instruction latency so `extract_slow`'s enumeration (act-backend/act_backend/generic/src/isel/extractor/slow.rs:14-50) is pruned by cost.

## Avoid

- Synthesis at compile time: Hydride's SyGuS windows and Isaria's rule synthesis run offline; keep anything solver-based in the generator, never in the per-kernel compile.
- Scalar DAGs: Diospyros's scalar unrolling does not scale to tensor kernels; TAIDL's tensor-level e-graph is the right level.

## Relevance

- High for the backend's rule schedule and cost; medium for the language.
