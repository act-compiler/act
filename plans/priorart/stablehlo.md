# StableHLO spec style

## What it is

- Each op has the same sections in order: Semantics, Inputs table (Label, Name, Type, Constraints), Outputs table, numbered Constraints such as `(C1) type(lhs) = type(rhs) = type(result)`, and Examples with `// %input` and `// %result` comments.
- Constraints are single predicates over helper functions `type()`, `shape()`, `element_type()`, `rank()`, `dim()`, `axes()`; the Inputs table cites constraint numbers rather than repeating them.
- Three words with fixed meanings: "implementation-defined" for open layout choices, "undefined" for violated preconditions (Cholesky on a non-positive-definite input), and "TBD" with an issue link for open semantics.
- Grammar is EBNF; decompositions of quantized ops are Python-style pseudocode; a reference interpreter executes the spec and the examples are its tests.
- Source: https://github.com/openxla/stablehlo/blob/main/docs/spec.md.

## TAIDL today, for comparison

- TAIDL's S8.2 table gives operand counts and attribute forms per operation (taidl/docs/operations.md:38-70) but S9 says "shapes and element types are not checked against S8" (taidl/docs/conformance.md, last section).
- Checks that StableHLO states as constraints live in generator code: the oracle's `disagreement` and `check_shapes` (act-oracle/taidl_to/oracle_visitor.py:285-301) and the backend's shape checks (act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:588-603).
- TAIDL's tensor operators already map one-to-one onto StableHLO names, with `dot` and `broadcast` as the stated exceptions (taidl/docs/operations.md:5-8).

## Borrow

- Numbered constraints per operation in S8: each with `(C<n>)` and a predicate over `shape`, `element_type` and attributes, cited by both generators' error messages.
- Inputs and Outputs tables with constraint references: one row per operand and per result.
- A three-word vocabulary with fixed meanings: rejected, fault, defined value; each S1.3 item classified under one word.
- Examples as tests: each S8 row gets one semantics line example with expected output that the oracle executes and the backend's conformance corpus reuses.
- Helper-function notation: define `shape(%v)`, `element_type(%v)`, `rank(%v)` once in S7 and use them in every constraint.

## Avoid

- "TBD" rows left open across a release: every S8 row is closed before the major release or the operation is removed.
- Quantized and dynamic-shape machinery: TAIDL excludes both (S7.3) and gains nothing from importing their constraint cases.

## Relevance

- High for the spec track.
