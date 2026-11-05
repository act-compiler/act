# Plan: spec adoption and ambiguity fixes

The spec draft lives in `taidl/docs/`, starting at `overview.md`, with stable section ids S1 to S10.
This plan closes the open cases S1.3 lists and makes the spec the one source each generator follows.
Items are ordered so the cheap, high-leverage fixes land in Q1 and the shape rules land before generator audits.
Sizes assume two engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L1 | Ship the spec skeleton and section ids | none | S | Q1 |
| L2 | Pin the reference operation version | L1 | S | Q1 |
| L3 | Normative shape and element type rules per operation | L2, O6, O7 | L | Q1 |
| L4 | Fix the `reduce_<reducer>` initial values and `constant` fill | L2, O, B | S | Q1 |
| L5 | Close region edge cases: bounds, zero sizes, output overlap | L1 | M | Q1 |
| L6 | Fix integer arithmetic: `/`, `%`, overflow | L1, B, O | S | Q1 |
| L7 | Per-generator unsupported-feature tables against section ids | L1, B, O | M | Q1 |
| L8 | Spec consistency checks in CI | L1, T | M | Q2 |
| L9 | Worked examples per section | L3 | M | Q2 |

## L1 Ship the spec skeleton and section ids

- What: land `overview.md`, `grammar.md`, `conformance.md`, `glossary.md`, and the section-id lines on the five existing pages.
- Changes: `docs/*.md` gain section ids and four new pages; every fact of the old pages is kept.
- Breaking: no.
- Benefits: other tracks cite S-ids today; generators get one entry point.
- Caveats: existing pages keep their long-sentence style until L9 rewrites them.
- Fail points: a section id is renamed later and breaks citations; notice by a grep of `S[0-9]` in B and O docs.
- Order and dependencies: first; B and O unsupported-feature docs (L7) cite it.
- Verification: every id in `outline.md` resolves to a heading or a page line; ASCII check is empty.

## L2 Pin the reference operation version

- What: S8.4 names one StableHLO spec version (and XLA-HLO revision for XLA-only words) as the reference.
- Changes: `docs/operations.md` S8.4 gains a version line; the word tables are kept.
- Breaking: no.
- Benefits: removes "which StableHLO" ambiguity; edge cases (NaN, overflow) become citable.
- Caveats: StableHLO evolves; a bump becomes a spec change.
- Fail points: O's JAX/XLA pin drifts from the named version; notice by the O env pin and the spec pin disagreeing.
- Order and dependencies: after L1; must match the O image pin (`act-env:py3.14-jax0.11.2`).
- Verification: the version string appears once in `operations.md` and matches O's environment.

## L3 Normative shape and element type rules per operation

- What: one table row per operation word: operand ranks, result shape, element type rule, attribute ranges.
- Changes: `docs/operations.md` S8.2 table and `OPERATIONS` in `taidl/operations.py` gain shape and type columns; existing columns kept.
- Breaking: no.
- Benefits: a new generator (simulator) can be written from the spec alone; front-end shape checks (L12) get a source.
- Caveats: 70 words; restate only what TAIDL's line form changes, cite the reference for the rest.
- Fail points: the table disagrees with O's `shapes.py`; notice by a T test that runs every row through O's shape checker.
- Order and dependencies: after L2; O6 and O7 move O's type and shape rules into it; feeds L12, L9, and B's instruction selector rules.
- Verification: every word in `IDLV2.g4` `OPERATION` has a row; a T test diffs the word lists.

## L4 Fix the `reduce_<reducer>` initial values and `constant` fill

- What: state the initial value per reducer and element type, and that a non-scalar `constant` fills every element.
- Changes: `docs/operations.md` S8.4 lines on `reduce_<reducer>` and `constant`; the operation words are kept.
- Breaking: yes, if a generator changes its current results.
- Benefits: removes two implementation-defined results every generator hits.
- Caveats: `reduce_max` on integers needs the lowest value of the type, not `-inf`.
- Fail points: B and O already differ; notice by a T differential test of each reducer on each type.
- Order and dependencies: after L2; O and B confirm their current behavior first.
- Verification: S8.4 has no "implementation-defined" line for these; T differential test passes.

## L5 Close region edge cases: bounds, zero sizes, output overlap

- What: decide each S1.3 region row: out-of-bounds start or size, size 0 or less, two outputs overlapping.
- Changes: `docs/instructions.md` S5.2 and the S1.3 table; the region form `[data model, starts, sizes]` is kept.
- Breaking: yes, an out-of-bounds region becomes a run-time error.
- Benefits: generators agree on what a run with bad attribute values does.
- Caveats: making them errors at run time needs every generator to check bounds.
- Fail points: a generator silently wraps addresses; notice by a T test with an out-of-bounds region.
- Order and dependencies: after L1; B (allocator) and O (oracle bounds) adopt.
- Verification: S1.3 region rows say "rejected at run time" or "undefined" with a D entry.

## L6 Fix integer arithmetic: `/`, `%`, overflow

- What: confirm `/` is floor division and `%` takes the divisor's sign (S6.2), and decide overflow handling.
- Changes: `docs/expressions.md` S6.2; the operator set is kept.
- Breaking: yes for any generator that truncates today.
- Benefits: B's Rust and O's Python agree on negative operands.
- Caveats: Rust and C++ `/` truncate; B must map explicitly.
- Fail points: `-7 / 2` gives -3 in one generator; notice by a T test of signed cases.
- Order and dependencies: after L1; B and O implement.
- Verification: T test of `/` and `%` on all sign pairs passes in B and O.

## L7 Per-generator unsupported-feature tables

- What: B and O each publish `feature | section id | behavior` rows for what they do not support.
- Changes: B and O docs gain tables; the spec is unchanged.
- Breaking: no.
- Benefits: users see coverage before writing an ISA description.
- Caveats: tables rot; tie them to a test.
- Fail points: a generator accepts an unsupported feature silently; notice by a T test per row expecting an error.
- Order and dependencies: after L1; owned by B and O; A tutorials link them.
- Verification: each row has a test that expects the named error.

## L8 Spec consistency checks in CI

- What: tests that the grammar copy in `grammar.md` matches the `.g4` rules, and the type and word lists match `data_model.TYPES` and `operations.OPERATIONS`.
- Also: re-anchor the existing doc tests (F23) on section ids instead of heading text.
- Changes: `tests/` gains doc checks; existing tests kept.
- Breaking: no.
- Benefits: the doc cannot drift from the grammar.
- Caveats: needs the CI image; no new dependency.
- Fail points: a grammar edit without a doc edit; notice by the failing test.
- Order and dependencies: after L1; T owns the test.
- Verification: the test fails on a deliberate one-word grammar change.

## L9 Worked examples per section

- What: one short accepted and one rejected example per section, and a rewrite of the five older pages to one sentence per line and bullets.
- Changes: `docs/semantics.md` and `docs/operations.md` prose; all facts kept.
- Breaking: no.
- Benefits: readers check their reading against an example.
- Caveats: examples must run in tests to stay true.
- Fail points: an example no longer parses; notice by a T test that loads every doc example.
- Order and dependencies: after L3.
- Verification: doc-example test passes.
