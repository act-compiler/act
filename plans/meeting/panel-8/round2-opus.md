# Panel 8, round 2: Opus (program manager)

Lens: one coherent plan, generated from one acyclic graph, with outcome milestones that can be shown done.
All three round-1 files read the regenerated tables (190 items) as current and the draft waves (175 items) as stale.

### Q8.1

- All three agree: B, rebuild the waves from the 190-item tables with a mechanical check.
- Strongest addition (Fable): other-core.md:9-18 marks L9, L18 to L23, L27, L28 as later, while tiers.md:27, :33 schedule them in waves 2 and 3.
- Concession: the rebuild must state per item which view wins; I add it as a check: an item marked later in other-core.md appears in no wave, or carries a stated reason.
- Final position: B, with a checker that fails on an ID in zero or two waves, an unknown dependency token, a cycle, or a later-marked item in a wave without a reason.
- Changed: no, extended.
- Confidence: high.
- Still divides: nothing.

### Q8.2

- All three agree on the order B29, B1, B31, B7, B9, B6, with B29 first as the gate, not D.
- Strongest addition (Fable): B7 "needs each operation's shape rule in Rust, sharing a table with the front end where one exists" (backend/plan/isel.md:32), and L3 is Q2 (q2-core.md:9), so B7 in Q1 writes a second copy of the shape rules.
- Verified: isel.md:32 says exactly that, and no table edge records a B7 to L3 relation.
- Concession: this is a real hidden dependency; it does not change the order inside the backend group, but it needs an edge or a stated choice.
- Program choice: B7 lands with a Rust rule table whose exit test compares it against the front-end table once L3 lands; the comparison is a named item after L3, not a block on B7.
- Rationale: blocking B7 on L3 would put the whole backend lane behind the front-end hinge, the longest chain already.
- Sonnet's point that the B6 form (R19) is decided before B6 starts, not before B7, is correct and adds no conflict.
- Final position: B, plus a B7-to-L3 agreement test scheduled after L3.
- Changed: no, extended.
- Confidence: high on order, medium on the agreement-test remedy.
- Still divides: whether B7 should wait for L3 (Fable leans that way) or ship a checked duplicate (my position).

### Q8.3

- Strongest opposing argument (Sonnet): spec ids cannot move, and renaming S1 to S9 edits every plan and ruling that cites them; typed tokens ("item:", "spec:") in dependency columns fix the tool input at lower cost.
- Rebuttal: typed tokens fix only the columns the generator reads; prose in milestones, risks, and rulings still says "S8" for two different things.
- A plain count does not settle cost: plans/ holds 2018 bare S-digit tokens and 1273 dotted section tokens, so bare S1 to S9 is mixed across both spaces and no grep separates them.
- That ambiguity is itself the argument for B: if a grep cannot tell the spaces apart, neither can a reader or a Claude-assisted edit.
- Fable notes the generator parses no typed prefix today (Fable Q8.3 step 2); both B and C change the generator, so tool cost is equal.
- Sonnet's rename of the "(D1)" to "(D5)" labels on B21 to B25 (q1-core.md:41-42) to defect names matches my D part.
- Concession: if the owner prefers not to rename, C is the acceptable second choice, provided the checker rejects any untyped token.
- Final position: B for shared items, D for defect labels and bug ids, and a registry of reserved prefixes the checker enforces.
- Changed: no.
- Confidence: medium.
- Still divides: rename (Opus, Fable) against typed tokens (Sonnet); both fix the data, they differ on prose.

### Q8.4

- The three positions converge: D (Sonnet), C with named exit tests (Fable), D in the shape of C (Opus).
- Strongest opposing argument (Fable): two items, not three or more, because the module move and the front-end checks have different consumers and the move can land first.
- Strongest opposing argument (Sonnet): three steps, with generator deletions (O7, B23, then D5) as their own step.
- Concession to Sonnet: consumer deletions are a third, separate step; they belong to the consumer tracks and SD2 makes B23 a backend edit (Fable Q8.4 step 3).
- Final shape: (a) L3 rule text, after L2; (b) S4 plus L12, O6, F1, module and checker in taidl, exit "shapes.py lives in taidl and the S8.3 table is generated"; (c) O7, B23 deletions, after (b).
- Each absorbed ID stays a named exit test inside its step, so 31 to 34 unassisted weeks stay visible.
- T18 and T20: all agree on E, T20 with no dependency and before T4 and T18.
- Sonnet places T20 in wave 1 as engineering test policy; Fable records it as the panel 6 outcome; both are compatible if panel 6 rules before wave 1 ends.
- Final position: D as three acyclic steps with sub-exit tests; E for T20, scheduled in wave 1.
- Changed: no, sharpened to three steps.
- Confidence: medium-high.
- Still divides: two steps (Fable) against three (Sonnet, Opus); a labeling difference, not a scope one.

### Q8.5

- All three agree: A plus D, PEP 440 pre-release tags that generators pin, and A16 un-merged from S8.
- Residual difference: I placed A16 in wave 1; Sonnet and Fable place it in wave 2.
- Concession: A16 has no dependency (q1-core.md:88), but the first pre-release follows S2 and S3, which sit in Q2 (dependencies.md:30-31); A16 is needed before that first tag, so wave 2 is sufficient.
- Final position: A plus D, A16 no later than wave 2 and before the first pre-release tag.
- Changed: yes, wave 1 to "no later than wave 2".
- Confidence: high.
- Still divides: nothing.

### Q8.6

- All three agree: B for every merge (named sub-item, own exit, real weight) and D for merges and drops on R5, R20, R29.
- Strongest addition (Sonnet, Fable): O9 under R60 is a clean drop because the work is removal; S7 into T7 loses parts SD18 keeps.
- Concession: O9's drop stands; S7 keeps a sub-item for the mini_trn merge and generator CI.
- Final position: B plus D; O10 is held, not dropped, until the R5 ruling is checked against owner decision 17; O9 dropped.
- Changed: no, O9 made explicit.
- Confidence: medium-high.
- Still divides: nothing.

### Q8.7

- Strongest opposing argument (Sonnet): B21 and B22 stay in M1, because in the regenerated table S3 depends on them (q2-supporting.md:31), not the reverse.
- Verified: S3 depends on B21, B22, L5, S2.
- Concession: my round-1 step 5 moved B21 and B22 next to S3 only if SD2 holds; under the current data they stay in M1, and the earliest-milestone rule places them correctly either way.
- Strongest addition (Fable): a standing criterion from M2 onward, "install from the milestone pre-release tag and run the README snippet", so the user-facing path is exercised four times, not once.
- Adopted: it follows from Q8.5 and is the cheapest early tripwire for "anyone can use".
- Sonnet's rewording of M1 to name only C1 to C4, C9 to C11, C13, with C5 to C8 in M2, is a valid correction under the same rule.
- Final position: five outcome milestones; item lists derived from exit criteria; tag closes M5; A18 before the tag; S9 a follow-on; Fable's standing install criterion from M2.
- Changed: yes, B21 and B22 stay in M1; standing criterion added.
- Confidence: medium-high.
- Still divides: nothing material; A with D (Sonnet, Fable) and D with A's correction (Opus) are the same plan.

### Q8.8

- All three agree: C plus D.
- Strongest addition (Fable): the track plans already carry observables for prose criteria, such as no `set_shapes_` in the crate (isel.md:35).
- Strongest addition (Sonnet): the trial uses only the migrated tutorial, since R63 voids tutorial evidence before A10.
- Adopted both: each prose criterion cites its track verification line; the trial follows A10.
- Trial parameters (number of people, time limit) are an owner value judgment; my default is two people outside the team and no maintainer help.
- Final position: C plus D, with the trial defined as above.
- Changed: no.
- Confidence: high.
- Still divides: nothing; trial numbers go to the owner.

### Q8.9

- Strongest opposing argument (Fable): order only now (A), since any label assumes an unmeasured assist factor of 3.7 to 5.7 and invites closing on a date (milestones.md:4).
- Strongest addition (Sonnet): capacity, not the chain, binds: the corrected chain is 36 to 40 unassisted weeks, which fits 26 calendar weeks at a factor of 1.4 to 1.5, while the total needs 4.3 to 5.7.
- Concession on arithmetic: load.md:18 gives 163 plus 254 weeks for scheduled items, and the 10 later items add weight, so Sonnet's 443 total is the right base, not my 427.
- Concession to Fable: no date or quarter label before M1; my round-1 position already put labels only after the rebaseline.
- Rebuttal to pure A: order-only gives the owner no early signal on decision 11; the published weights and the M1 ratio are that signal.
- Final position: no dates before M1; publish recomputed weights and chain with the milestones (C); measure actual assisted weeks per wave 1 item and set labels at M1 (D).
- Changed: yes on the total (443) and on stating "no dates before M1" outright.
- Confidence: high.
- Still divides: nothing of substance; all three converge on C then D with no early dates.

### Q8.10

- All three agree on B, a wave 1 gate row, with A10 and the Ascend study in it.
- Residual difference: the O19 to O8 edge; Fable drops it, Sonnet asks the oracle owner, I measured on current literals.
- Verified: oracle/plan/architecture.md:24 orders O19 "after O8 and O21, so the baseline outputs are the fixed ones"; the verification is a compile-time table (architecture.md:25).
- Rebuttal: compile time does not depend on literal rounding, so the O8 edge guards output equality, not the measurement; the edge belongs on O20, which acts on the result.
- Architecture.md:23 also requires the owner to fix the bound before the spike runs; that bound is a wave 1 gate outcome.
- Final position: B; move the O8 edge from O19 to O20; owner bound for R28, approvals for R29 and R45, A10, and the Ascend study all in the wave 1 gate row.
- Changed: no, made concrete.
- Confidence: high.
- Still divides: whether the oracle owner must confirm the edge move (Sonnet) or the panel moves it (Fable, Opus).

### Q8.11

- All three agree: B plus C without the drop rule; a risk no criterion covers adds a criterion.
- Sonnet adds a fallback outcome per risk; Fable adds the B7-against-L3 duplicate and release order written last.
- Adopted: each risk row has ID, tripwire, endangered exit criterion, fallback outcome, and category tag (D as a column).
- Final position: as above, with the R18 conflict removed from the register and kept only in corrections.
- Changed: no, extended.
- Confidence: high.
- Still divides: nothing.

## Escalation to the owner

- Decision 11 against "nothing is cut": the scheduled items total about 443 unassisted engineer-weeks against 78 to 104 available, so the plan holds only with an assist factor of 4.3 to 5.7 that no source states.
- Request: the owner confirms that the milestone dates float on the M1 measurement, rather than scope, if the measured factor falls below 4.3.
