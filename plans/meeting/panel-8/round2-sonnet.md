# Panel 8, round 2: Sonnet (engineer executing the plan)

## Verified facts used below

- The regenerated tables hold 190 IDs; size weights S = 1, M = 3, L = 6 give 163 (Q1) + 254 (Q2) + 26 (later) = 443 unassisted weeks.
- The later group in roadmap/other-core.md is 1 L, 6 M, 2 S = 26 weeks; load.md prints the later item count (10), and a total of 427 mistakes that count for weight.
- 443 weeks against 78 to 104 capacity weeks implies an assist factor of 4.3 to 5.7; no document measures it.
- The shape-rule group sums to 31 weeks by table sizes: L3 6, L12 6, S4 6, F1 6, O7 3, B23 3, O6 1 (roadmap/q2-core.md:9, :11, :29, :41-42; q2-supporting.md:32; q1-supporting.md:45).
- The cycle edges are L3 -> O6, O7 (q2-core.md:9) against O6 -> L3 (:41), and O7 -> B23, L12, O6 (:42) against B23 -> L12, L3 (:29).
- S4 also depends on B23, O7, and S3 (q2-supporting.md:32), so S4 is a consumer-last item in the table and a rule-first item in R62.
- S3 depends on B21 and B22 (q2-supporting.md:31); S8 depends on S2 to S7 (:36).
- T20 "binds T4 and T18" (tests/decisions/rulings.md:22) and the table has T20 -> T18 (q2-core.md:74, :76).
- O19 is "after O8 and O21, so the baseline outputs are the fixed ones" (oracle/plan/architecture.md:26); O8 -> L16 -> L3 (q2-core.md:43, :13, :9).
- A16 and A10 have no dependencies (q1-core.md:84, :88); A18 depends on A16 (q2-core.md:87).

### Q8.1

- Where all three agree: B, rebuild the waves from the 190-item tables with a machine check that every ID sits in exactly one row.
- Strongest opposing argument: Fable adds that the tables and other-core.md disagree with tiers.md on L18 to L23, L27, L28 (other-core.md:9-18 against tiers.md:27, :33), so the rebuild must state which view wins per item.
- Concession: the rebuild rule is "the plan's own dependency edges decide the wave, the table quarter is a hint"; the checker fails on a later-wave dependency, which settles each of those items without a hand ruling.
- Final position: B, with the checker covering zero or double placement, unknown tokens, and reading order.
- Confidence: high.
- Still divides the panel: nothing.

### Q8.2

- Where all three agree: B, order B29, B1, B31, B7, B9, B6, with B29 as the first item and not a hold on the group.
- Strongest opposing argument: Fable finds a hidden edge no table records: B7 "needs each operation's shape rule in Rust, sharing a table with the front end where one exists" (backend/plan/isel.md:32), while the shape rules land in Q2 (L3), so B7 may write a second copy that R62 forbids.
- Concession: the risk is real and I had not weighed it; the table has no edge from B7 to a rule source.
- Rebuttal in part: the same line says "where one exists", so B7 may read the rule list the oracle already holds; the fix is an exit line on B7 ("one rule table, generated from the single source"), and a risk entry, not a reorder.
- Final position: B, plus an explicit B7 exit line naming the source of its Rust shape table, and B29 and B3 named in M1 so the M2 exit "goldens unchanged under R17" has an item behind it.
- Confidence: high on the order, medium on the B7 source.
- Still divides the panel: whether the B7 source question is a risk entry (my view) or a new dependency edge (Fable leans edge); both fit one row.

### Q8.3

- Strongest opposing argument: Opus and Fable both pick B for the shared items, plus D for the labels.
- Opus: prefix schemes rely on every author typing the prefix, and the existing errors are exactly a missing prefix; item IDs are internal, so renaming costs one generator pass.
- Fable: the generator resolves a bare "S8" and cannot tell item from section, and nine shared items are cheaper to rename than ten spec sections plus every S9 row.
- Concession: I move to B plus D.
- Evidence that decides it: with a rename, the checker needs no discipline, because any dependency token that is not an item ID (including a bare section id such as S10) fails loudly; with typed prefixes, a missing prefix stays a silent miss unless the checker also rejects untyped tokens, which is the same rule at higher cost.
- My load.md case (the chain S1 -> T5 reading a section as the shared item) shows the failure; a rename removes its cause.
- Spec section ids stay, because they ship in error messages (R44) and are tagged in tests (T1).
- Final position: B plus D. Shared items S1 to S9 take a letter no other ID space uses (default U), the registry of reserved prefixes lives in the checker, and the backend labels (D1) to (D5) and bug ids X1 to X10 become free text or a distinct prefix.
- Confidence: medium-high.
- Still divides the panel: nothing material; the letter is cosmetic.

### Q8.4

- Strongest opposing argument: Fable (option C) holds that two items are enough: the module move into taidl, then the front-end checks, with each absorbed item kept as a named exit test; B23 stays out of the front-end item because SD2 makes it a backend edit.
- Opus (option D in the shape of C) orders L3 first, then S4 plus L12, then the consumers O6, O7, B23.
- Rebuttal: the three groups differ from Fable's two only in keeping the rule text apart; the rule text has its own exit (every operation has a normative rule, so a clause can cite it) that does not wait on code, and it is the only part the generators and the oracle both depend on.
- Concession: Fable is right that B23 is a backend edit and that F1 needs a named exit; both are covered below.
- Sizes: the group is 31 weeks by table sizes; one item of size L hides 25 of them, so the parts keep their table sizes.
- Final position (shape and type): D with three acyclic groups, each with its own exit.
  - (a) L3 rule text per operation, depends on L2 and S3 only.
  - (b) taidl-side implementation: L12, S4, F1, O6, exit "shapes.py lives in taidl and the S8.3 table is generated".
  - (c) consumer deletions: O7, B23, then D5.
- The edges that make the cycle (L3 -> O6, O7; S4 -> B23, O7) are reversed so consumers depend on rules.
- Final position (T18 and T20): E. T20 has no dependency, T4 and T18 depend on it, and it runs with T22 in wave 1.
- It is test policy ("impl tests survive with a header"), so G is not needed; the judgement per test happens in the T18 PRs.
- Confidence: medium-high.
- Still divides the panel: two groups (Fable) against three (Opus and I); the exits are the same, so this is a naming choice.

### Q8.5

- Where all three agree: A, PEP 440 pre-release tags of taidl (3.0.0aN, then rcN) that generators pin, and the A16 checklist un-merged from S8.
- Strongest opposing argument: Opus holds that an early contract-only tag (option C) is a second major in effect under owner decision 10 if later waves still break the contract, and that A16 belongs in wave 1 so A18 runs against every pre-release.
- Concession on C: the S2 and S3 tag is the first pre-release of A and carries no separate name; nothing is called a release before the v3 tag, and pip skips pre-releases unless asked.
- Concession on A16: it has no dependency, so wave 1 costs nothing and gives A18 the earliest start; I move it from wave 2 to wave 1.
- Fable's addition is kept: each pre-release runs the README snippet install (see Q8.7).
- Final position: A, A16 un-merged and scheduled before the first pre-release tag (wave 1), A18 before the final tag.
- Confidence: medium-high.
- Still divides the panel: nothing; the owner reading of decision 10 is flagged in the escalation section.

### Q8.6

- Where all three agree: B for every merge (named sub-item, own exit, own table weight), and D for merges and drops that rest on R5, R20, R29.
- Strongest opposing argument: Fable and Opus add that S7 into T7 loses the mini_trn merge and generator CI parts that SD18 keeps (reviews/shared-diagnostics/review.md SD18); Opus adds that merge only where both plan texts describe the same work (C as a filter).
- Concession: S7 is on my list; S7 (L8, S3) and T7 (T1, T5, T6) have different edges (q2-supporting.md:35, q2-core.md:63), which is evidence the work differs.
- Evidence on O10: O10 (output reuses the marked input's storage, depends on L5, size S) implements owner decision 17 in the oracle, and B30 (in-place storage reuse as a solver equality, size M) implements it in the backend (q1-core.md:57, :49).
- The audit finds R5 reverses the semantics ruling, so O10 and the F2 meaning (F2 is a size M item with no dependency, q1-supporting.md:46) stay as sub-items until the owning panels rule.
- O9 depends on L3 and is "measure, then widen"; R60 removes `triangular_solve`, so the drop is a removal and stands as an item-less note.
- Final position: B for all nine merges plus the S7 split, D for O10, F2, P6, F9, O9 stands.
- Confidence: medium-high.
- Still divides the panel: nothing.

### Q8.7

- Strongest opposing argument: Opus (option D) derives each item list from the exit criteria so that an item sits in the earliest milestone whose criterion needs it; that rule moves C5 to C8 into M1 or splits "all sketches parse".
- Fable (option A) adds one standing criterion from M2: install from the milestone pre-release tag and run the README snippet.
- Rebuttal to the C5 to C8 move: C5 and C8 are size L and C6 and C7 size M (q2-supporting.md:9-12), 18 weeks that M1 does not need for a spec outcome; rewording the M1 exit to name its sketches is an outcome statement, not a retreat.
- Concession on method: the derivation check is right, and it is a checker rule: every item named in an exit is in that milestone or an earlier one.
- Concession on the standing criterion: it is an outcome ("a newcomer's install works at every milestone") and costs one CI job once the pre-release tags exist.
- Final position: A (five outcome milestones) with the exit-to-item derivation check, the standing install criterion from M2, the v3 tag as the last event of M5, and S9 as a listed follow-on.
- Confidence: medium.
- Still divides the panel: Opus would move C5 to C8 into M1; Fable and I would reword the M1 exit.

### Q8.8

- Where all three agree: C plus D. Mechanical criteria map to a named CI job; the "anyone" criteria use a cold-start trial by a person outside the team; each milestone adds a negative criterion that a test checks.
- Strongest opposing argument: Opus names a Claude-driven fresh-container run as the fallback when no outside person exists.
- Rebuttal: a Claude run shows the instructions are executable, not that a newcomer succeeds, and the engineers already develop with Claude; it is an extra CI job and not a substitute for the person.
- The fallback is a non-author teammate with a recorded transcript, stated as the weaker form.
- Fable's addition is kept: the observables already exist in the plans' verification lines (no `set_shapes_` in the crate, backend/plan/isel.md:35), so negative criteria cost little.
- Final position: C plus D, tutorial trial only on the migrated tutorial (R63).
- Confidence: high.
- Still divides the panel: nothing; the number of people and the time limit are the owner's call.

### Q8.9

- Strongest opposing argument: Fable (option A now) holds that a quarter label on an outcome milestone invites closing on the date, which the milestones text forbids, so no label until wave 1 is measured.
- Rebuttal: my position already carries no dates before M1; "C then D" publishes weights and the recomputed chain beside the milestones and rebaselines at M1, which is Fable's sequence.
- Correction to Opus: the total is 443, not 427, because load.md prints the later count (10) where the weight is 26; the implied factor is 4.3 to 5.7.
- Evidence on the chain: with merges applied, T5 fixed to L1, and T20 before T18, my longest chain is 40 weeks, and 36 without the O19 -> O8 edge; the published 59 contains the cycles and a misread S1 (roadmap/load.md).
- Capacity, not the chain, binds: a chain of 40 needs a factor of 1.5 against 26 weeks, and the total needs 4.3 or more.
- Final position: order only until M1, then C and D at M1 with a measured factor, weights beside milestones.
- Confidence: medium-high.
- Still divides the panel: nothing material.

### Q8.10

- Where all three agree: B, a wave 1 gate row with a named outcome per gate, A10 and the Ascend study moved into it.
- Strongest opposing argument: Opus and Fable say the O19 -> O8 edge exists for output baselines, not for the compile-time measurement, so the spike can run in wave 1 and the edge should go.
- Concession: the architecture text gives the reason as "so the baseline outputs are the fixed ones" (oracle/plan/architecture.md:26), and a compile-time table does not read outputs.
- I moved from "ask the oracle owner" to "split": O19 measures compile time after O21 only, and its output comparison runs after O8.
- Fable's alternative for the Ascend study (drop it from the R59 count by decision) removes work; nothing is cut, so the study moves to wave 1 or 2 as its own item.
- The R28 bound is an owner input recorded before the spike (architecture.md:23).
- Final position: B, with the O19 split, and the gate outcomes: R45 and R29 approved or refused, R28 bound set, A10 failing-syntax list, Ascend sketch parses.
- Confidence: high.
- Still divides the panel: nothing.

### Q8.11

- Where all three agree: B plus C without the drop rule; each risk has an ID, an observable tripwire, the exit criterion it endangers, a response, and an owner role; the R18 conflict leaves the register.
- Strongest opposing argument: Fable adds candidates I lacked: B7 Rust shape rules against L3 (Q8.2), the O19 chain behind L3, a jax bump becoming a spec change (language/plan/spec.md:39-40), and release order written last.
- Concession: all four join the list; the O19 chain risk is closed once the Q8.10 split lands.
- Candidates joined from the reviews and audit: v3 migration (RA11), R56 against i64 checking, ANTLR exposure (SD4), waves 3 to 5 compression.
- Candidates joined from this panel: unmeasured assist factor (tripwire: the M1 ratio below 4.3), graph integrity (tripwire: the checker fails).
- Final position: B plus C, with D's three categories as a tag column.
- Confidence: medium-high.
- Still divides the panel: nothing.

## Escalation to the owner

- E1: decision 10 and published pre-release tags.
  - The panel converges on pre-release tags (3.0.0aN, rcN) that generators pin; pip skips them unless asked, so users see one release.
  - Opus would change position if the owner counts any published tag as a release.
  - Needed: the owner confirms pre-release tags are not releases under decision 10.
- E2: decision 11 with nothing cut.
  - The tables total 443 unassisted weeks against 78 to 104 capacity weeks, an implied assist factor of 4.3 to 5.7, and nothing measures it.
  - This is not a counter to the decision; it is a risk the owner may want to bound before M1, for example by stating a fallback order if M1 measures below 4.3.
  - Needed: the owner states whether a measured factor below that triggers a re-plan or only a re-label.
