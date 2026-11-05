# Plan: spec adoption and fault rules

The spec draft lives in `taidl/docs/`, starting at `overview.md`, with stable section ids S1 to S10.
This plan makes every S1.3 row hold in each generator and makes the spec the one source each generator follows.
Items are ordered so the cheap, high-leverage fixes land in Q1 and the shape rules land before generator audits.
Sizes assume the shared 3-4 engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L1 | Ship the spec skeleton and section ids | none | S | Q1 |
| L2 | Pin the reference operation version | L1 | S | Q1 |
| L3 | Normative shape and element type rules per operation | L2, O6, O7 | L | Q1 |
| L4 | Check generators against the stated reduce identities and `constant` fill | L2 | S | Q1 |
| L5 | Make the region and in-place fault rows hold | L1 | S | Q1 |
| L6 | Exact integer arithmetic with a 64-bit fault | L1 | S | Q1 |
| L7 | ACT-Backend unsupported-feature table against section ids | L1 | S | Q1 |
| L8 | Spec consistency checks in CI | L1 | S | Q2 |
| L9 | Worked examples per section | L3 | M | after v3 |
| L29 | Object model contract test | L1 | S | Q1 |
| L31 | Float accuracy norm for `dot`, `reduce_*`, and transcendental words | L2 | S | Q1 |

## L1 Ship the spec skeleton and section ids

- What: land the twelve spec pages, including `object_model.md` (S4.6) and `execution.md` (S5.6), and the section-id lines.
- Changes: `docs/*.md` gain section ids and new pages; every fact of the old pages is kept or replaced by a stated rule.
- Breaking: no.
- Benefits: other tracks cite S-ids and S1.3 row ids today; generators get one entry point.
- Caveats: existing pages keep their long-sentence style until L9 rewrites them.
- Fail points: a section id is renamed later and breaks citations; notice by a grep of `S[0-9]` in B and O docs.
- Order and dependencies: first; L7 cites it.
- Verification: every id in `outline.md` resolves to a heading or a page line; ASCII check is empty.

## L2 Pin the reference operation version

- What: S8.4 names the StableHLO version bundled with the oracle's pinned jax, and that jaxlib's XLA commit for XLA-only words.
- Changes: `docs/operation_meaning.md` S8.4 gains the version line; the word tables are kept.
- Breaking: no.
- Benefits: removes "which StableHLO" ambiguity; edge cases become citable.
- Caveats: a jax bump in the oracle becomes a spec change.
- Fail points: the oracle's jax pin drifts from the named version; notice by the O environment pin and the spec pin disagreeing.
- Order and dependencies: after L1; must match the O image pin (`act-env:py3.14-jax0.11.2`).
- Verification: the version string appears once in `operation_meaning.md` and matches O's environment.

## L3 Normative shape and element type rules per operation

- What: one rule per operation word: operand ranks, result shape, element type rule, axis ranges, operand order.
- Changes: `docs/operations.md` S8.2 and `OPERATIONS` in `taidl/operations.py` gain shape and type rules; existing columns kept.
- Breaking: no.
- Benefits: a simulator can be written from the spec alone; the L12 checker gets a source.
- Caveats: 78 words (`IDLV2.g4:45-74`); restate only what TAIDL's line form changes, cite the reference for the rest.
- Fail points: the rules disagree with O's `shapes.py`; notice by a T test that runs every word through O's shape checker.
- Order and dependencies: after L2; feeds L12 and L9.
- Merge of L3, L12, O6, and O7 into one item that moves O's `shapes.py` into `taidl`: pending panel (R62).
- Verification: every word in `IDLV2.g4` `OPERATION` has a rule; a T test diffs the word lists.

## L4 Check generators against the stated reduce identities and `constant` fill

- What: S8.4 states the identity per reducer and type, matching O's `reduce_init`; B and O are tested against it.
- Changes: a T test per reducer and type; the spec text is in place.
- Breaking: no for O; yes for B if its results differ.
- Benefits: removes two open results every generator hits.
- Caveats: `f8e4m3fn` has no infinity, so `max` and `min` start at its finite extremes.
- Fail points: a generator starts a float sum at -0.0; notice by the T test on an all-negative-zero operand.
- Order and dependencies: after L2.
- Verification: T differential test of each reducer on each type passes.

## L5 Make the region and in-place fault rows hold

- What: S1.3.3, S1.3.4, and S1.3.6 hold in every executor, and `set_inplace` rejects a mismatched buffer or sizes.
- Changes: `Instruction.set_inplace` in `taidl/instruction.py` gains the buffer and size check; O gains the output-overlap and in-place start checks.
- Breaking: yes, an out-of-bounds region or overlapping outputs now fault.
- Benefits: generators agree on what a run with bad attribute values does; B's allocator can rely on in-place storage reuse.
- Caveats: O already rejects an out-of-bounds region at the call, so most work is the two new checks.
- Fail points: a generator silently wraps addresses; notice by a T test with an out-of-bounds region.
- Order and dependencies: after L1.
- Verification: a T test per fault row expects the fault and its row id.

## L6 Exact integer arithmetic with a 64-bit fault

- What: S6.2 states exact arithmetic, floor `/`, divisor-signed `%`, and a fault for a final value outside signed 64 bits.
- Changes: O raises instead of warning; B maps floor division and the fault explicitly.
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
- Benefits: users see coverage before writing an ISA description.
- Caveats: tables rot; tie them to a test.
- Fail points: a generator accepts an unsupported feature silently; notice by a T test per row expecting an error.
- Order and dependencies: after L1; owned by B.
- Verification: each row has a test that expects the named error.

## L8 Spec consistency checks in CI

- What: tests that the type and word lists match `data_model.TYPES` and `operations.OPERATIONS`, and doc tests anchored on section ids (F23).
- Changes: `tests/` gains doc checks; existing tests kept.
- Breaking: no.
- Benefits: the doc cannot drift from the code's lists; the grammar page names the `.g4` files instead of copying them.
- Caveats: needs the CI image; no new dependency.
- Fail points: a word added to the grammar without the doc; notice by the failing test.
- Order and dependencies: after L1; T owns the test.
- Verification: the test fails on a deliberate one-word grammar change.

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

- What: a test that every attribute S4.6 names exists on an accepted accelerator with the stated type.
- Changes: `tests/` gains one test; the front end is unchanged.
- Breaking: no.
- Benefits: generators read a documented contract; a rename breaks the test, not a generator.
- Caveats: renames beyond owner decision 19 wait for the panel.
- Fail points: an attribute added without a row; notice by review of S4.6.
- Order and dependencies: after L1; L15 and L30 update it.
- Verification: the test fails on a deliberate attribute rename.

## L31 Float accuracy norm

- What: replace S1.3.15 with a defined norm for float `dot` and `reduce_*` accumulation and transcendental accuracy.
- Changes: `docs/operation_meaning.md` S8.4 and the S1.3 row.
- Breaking: no for O; B computes no values.
- Benefits: closes the last open run-time row under owner decision 16.
- Caveats: the norm is pending panel (R8); a fixed summation order would bind XLA-backed oracles to one schedule.
- Fail points: O's per-word test bounds exceed the stated tolerance; notice by the T differential test.
- Order and dependencies: after L2 and the panel.
- Verification: S1.3 has no open row.
