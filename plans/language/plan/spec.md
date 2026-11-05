# Plan: spec adoption and fault rules

The spec draft lives in `taidl/docs/`, starting at `overview.md`, with stable section ids S1 to S12.
This plan makes every S1.3 row hold in each generator and makes the spec the one source each generator follows.
Items are ordered so the cheap, high-leverage fixes land in Q1 and the shape rules land before generator audits.
Sizes assume the shared 3-4 engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L1 | Ship the spec skeleton and section ids | none | S | Q1 |
| L2 | Pin the reference operator version | L1 | S | Q1 |
| L3 | Normative shape and element type rules per operation | L2 | L | Q1 |
| L4 | Check generators against the stated reduce identities and `constant` fill | L2 | S | Q1 |
| L5 | Make the region, in-place, and overlap fault rows hold | L1, L36 | S | Q1 |
| L6 | 64-bit integer arithmetic per operator with a fault | L1 | S | Q1 |
| L7 | ACT-Backend unsupported-feature table against section ids | L1 | S | Q1 |
| L8 | Spec consistency checks in CI | L1 | S | Q2 |
| L9 | Worked examples per section | L3 | M | after v3 |
| L29 | Object model contract test for the five fixed names | L1 | S | Q1 |
| L31 | Float accuracy bounds for `dot`, `reduce_add`, `reduce_multiply` | L2 | S | Q1 |

## L1 Ship the spec skeleton and section ids

- What: land the twelve spec pages, including `object_model.md` (S4.6) and `execution.md` (S5.6), and the section-id lines.
- Changes: `docs/*.md` gain section ids and new pages; every fact of the old pages is kept or replaced by a stated rule.
- Breaking: no.
- Benefits: other tracks cite S-ids and S1.3 row ids today; generators get one entry point.
- Caveats: existing pages keep their long-sentence style until L9 rewrites them.
- Fail points: a section id is renamed later and breaks citations; notice by a grep of `S[0-9]` in B and O docs.
- Order and dependencies: first; L7 cites it.
- Verification: every id in `outline.md` resolves to a heading or a page line; ASCII check is empty.

## L2 Pin the reference operator version

- What: S8.4 names the StableHLO version bundled with the oracle's pinned jax, and that jaxlib's XLA commit for XLA-only words.
- Changes: `docs/operation_meaning.md` S8.4 gains the version line; the word tables are kept.
- Breaking: no.
- Benefits: removes "which StableHLO" ambiguity; edge cases become citable.
- Caveats: a jax bump in the oracle becomes a spec change.
- Fail points: the oracle's jax pin drifts from the named version; notice by the O environment pin and the spec pin disagreeing.
- Order and dependencies: after L1; must match the O image pin (`act-env:py3.14-jax0.11.2`).
- Verification: the version string appears once in `operation_meaning.md` and matches O's environment.

## L3 Normative shape and element type rules per operation

- What: one rule per tensor operator: operand ranks, result shape, element type rule, axis ranges, operand order.
- Changes: `docs/operations.md` S8.2 and `OPERATIONS` in `taidl/operations.py` gain shape and type rules; existing columns kept.
- Breaking: no.
- Benefits: a simulator can be written from the spec alone; the shared shape module of U4 gets a source.
- Caveats: 75 words after L33 removes three; restate only what TAIDL's line form changes, cite the reference for the rest.
- Fail points: the rules disagree with the shape module; notice by a T test that runs every word through it.
- Order and dependencies: after L2 only; L3 is the spec text, and U4 owns the shape and per-word type code, which absorbs L12, O6, and O7.
- Verification: every word of the `OPERATOR` token has a rule; a T test diffs the word lists.

## L4 Check generators against the stated reduce identities and `constant` fill

- What: S8.4 states the identity per reducer and type, matching O's `reduce_init`; B and O are tested against it.
- Changes: a T test per reducer and type; the spec text is in place.
- Breaking: no for O; yes for B if its results differ.
- Benefits: removes two open results every generator hits.
- Caveats: `f8e4m3fn` has no infinity, so `max` and `min` start at its finite extremes.
- Fail points: a generator starts a float sum at -0.0; notice by the T test on an all-negative-zero operand.
- Order and dependencies: after L2.
- Verification: T differential test of each reducer on each type passes.

## L5 Make the region, in-place, and overlap fault rows hold

- What: S1.3.3, S1.3.4, and S1.3.6 hold in every executor as the one constraint fault of S1.3.5, checked before any load (Q9.3, Q9.4).
- Changes: O gains the output-overlap check, the `set_inplace` equality, the default exact-or-apart constraint, and the size term, all from the front end's one list; O evaluates the semantics on the pre-issue state for every relation.
- Changes: `set_inplace` rejects a pair in different buffers (owner 24) or with unequal size text.
- Changes: B honors every constraint, reads each pair's relation and tie, and applies the relation in its touch exception (Q9.4).
- Breaking: yes, an out-of-bounds region, overlapping outputs, or a partial overlap of an unmarked same-buffer pair now fault.
- Benefits: generators agree on what a run with bad attribute values does.
- Caveats: O already rejects an out-of-bounds region at the call; the marks stay as shorthand (owner 30); if exact-or-apart overloads the solver, B may allocate default pairs apart, recorded as a backend limit.
- Fail points: a generator silently wraps addresses, or B lets a dying input share storage at a forbidden offset; notice by T tests with an out-of-bounds region and per-relation allocation rows.
- Order and dependencies: after L1 and L36.
- Verification: a T test per fault row expects the fault and its row id; the corpus runs the five declarations by three placements (apart, exact, partial), plus the rejection, LB5, OD34, input-input, cross-check, and backend rows of Q9.4.

## L6 64-bit integer arithmetic per operator with a fault

- What: S6.2 states that every operator yields a signed 64-bit integer or the issue faults (Q1.2), with floor `/`, divisor-signed `%`, and left-associative grouping.
- Changes: O checks each operator of the tree as written, never its canonical sum of products, and raises instead of warning; B maps floor division and keeps its per-operator checked arithmetic.
- Breaking: yes for any generator that truncates or wraps today.
- Benefits: B's Rust and O's Python agree on negative operands and on overflow.
- Caveats: Rust `/` truncates; B must map explicitly.
- Fail points: `-7 / 2` gives -3 in one generator; notice by a T test of signed cases.
- Order and dependencies: after L1.
- Verification: T test of `/`, `%`, and 64-bit edges on all sign pairs passes in B and O.

## L7 ACT-Backend unsupported-feature table

- What: B publishes `feature | section id | status` rows for what it does not support; O's `docs/coverage.md` already exists and only gains section ids.
- Changes: B docs gain the table; the spec is unchanged.
- Breaking: no.
- Benefits: users see coverage before writing an ISA specification.
- Caveats: tables rot; tie them to a test.
- Fail points: a generator accepts an unsupported feature silently; notice by a T test per row expecting an error.
- Order and dependencies: after L1; owned by B.
- Verification: each row has a test that expects the named error.

## L8 Spec consistency checks in CI

- What: tests that the type and word lists match `data_model.TYPES` and `operations.OPERATIONS`, and doc tests anchored on section ids (LB23).
- Changes: `tests/` gains doc checks; existing tests kept.
- Breaking: no.
- Benefits: the doc cannot drift from the code's lists, including the names S2.1 lists and the `ops` functions.
- Caveats: needs the CI image; no new dependency.
- Fail points: a word added to `OPERATIONS` or `ops` without the doc; notice by the failing test.
- Order and dependencies: after L1; T owns the test.
- Verification: the test fails on a deliberate one-word change to `OPERATIONS`.

## L9 Worked examples per section

- What: one short accepted and one rejected example per section, and a rewrite of the older pages to one sentence per line.
- Changes: `docs/semantics.md` and `docs/operations.md` prose; all facts kept.
- Breaking: no.
- Benefits: readers check their reading against an example.
- Caveats: `tests/test_examples.py` already runs both examples, so this is polish.
- Fail points: an example no longer parses; notice by a T test that loads every doc example.
- Order and dependencies: after L3.
- Verification: doc-example test passes.

## L29 Object model contract test

- What: a test that the `Model` `check` returns is frozen and carries every member S4.6 names with its U2 type.
- Changes: `tests/` gains one test; each member is documented in the `taidl` README; the typed contract model is U2.
- Breaking: no.
- Benefits: generators read a documented contract; a rename breaks the test, not a generator.
- Caveats: no tree is part of the contract; ANTLR trees leave with U14.
- Fail points: a fixed name renamed without the page; notice by the test.
- Order and dependencies: after L1 and L43; L15 and L30 update it.
- Verification: the test fails on a deliberate attribute rename.

## L31 Float accuracy bounds

- What: S1.3.15 and S8.4 state two Higham bounds with u of the result type plus an overflow clause, for `dot`, `reduce_add`, and `reduce_multiply` only (Q1.5); `reduce_max` and `reduce_min` stay exact.
- Changes: `docs/operation_meaning.md` S8.4 and the S1.3 row; T gains a bound test per word and type.
- Breaking: no for O; B computes no values.
- Benefits: closes the last open run-time row under owner decision 16; any order and any accumulation at least as wide as the result type conforms.
- Caveats: for `n*u >= 1` the bound constrains nothing, and the ISA author writes a wider result type.
- Fail points: O's results exceed the bound; notice by the T test.
- Order and dependencies: after L2.
- Verification: S1.3 has no open row; the T bound test passes on both platforms.
