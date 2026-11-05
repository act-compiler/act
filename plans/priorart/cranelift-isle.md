# Cranelift ISLE

## What it is

- ISLE is a typed term-rewriting DSL compiled to Rust for Cranelift's instruction lowering.
- Terms are declared with `decl`; `rule` pairs a pattern with an expression; `extractor` decomposes on the left, `constructor` builds on the right; `extern` binds either to Rust.
- Partial constructors may fail, `if-let` adds fallible side conditions, `pure` marks side-effect-free terms, and `(rule 1 LHS RHS)` gives an explicit priority.
- Rules compile to a decision trie with backtracking; the first successful match wins; overlapping rules at equal priority are allowed and resolved by an unspecified "more specific wins" heuristic.
- Source: https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/isle/docs/language-reference.md.

## TAIDL today, for comparison

- The backend's precondition, metadata and set-shapes functions are three parallel `match name { "0" => precond_0, ... }` tables (act-backend/act_backend/generic/src/isel/rewrites/applier.rs:15-50), where ISLE would have one typed term per rule.
- The e-class analysis `TensorInfo::merge` panics when a rewrite joins values of different shape or dtype (act-backend/act_backend/generic/src/ir/egraph.rs:404-415); ISLE would reject the rule at compile time by typing.

## Borrow

- Typed rule terms: give rewrite patterns element type and shape variables so a rule that could join unlike shapes is rejected when the backend is generated, not at compile time of a kernel.
- Extractors as the unit of reuse: shape recovery, constant detection and attribute solving become named extern extractors generated once, with rules composing them.
- Explicit priorities with a stated semantics: where two instructions can select the same subgraph, the preference is declared, not left to e-node order (act-backend/act_backend/generic/src/isel/extractor/fast.rs:24-26 relies on e-node order for stable numbering).
- A `Context` trait boundary: the generated Rust and the generic Rust meet at one trait, which is a clean interface for the shared-code track.

## Avoid

- Unspecified overlap resolution: do not depend on specificity heuristics; state a total order or reject overlap.
- Backtracking tries for the search itself: the e-graph remains the search; ISLE-style compilation fits the per-rule side conditions only.

## Relevance

- High for rule representation and typing; medium otherwise.
