# Panel 8, round 2, Fable: risk red team

Paths are relative to /home/user/act/plans unless they start with work/.
Line numbers below were re-read this round against the current files.

### Q8.1

All three take B: rebuild the waves from the regenerated 190-item tables with a machine check, and no hand patches.
Sonnet adds two facts I verified: wave 1 holds O1 (work/roadmap_extra/tiers.md:22) while O1 depends on O20 (roadmap/q2-core.md:38); wave 1 holds B1 and B3 (tiers.md:20) while both depend on B29 (roadmap/q1-core.md:30, :32), and B29 is in no wave.
So the draft wave 1 cannot start as written, which closes option A.
I add that the rebuild must state which view wins where other-core.md and tiers.md disagree: L27, L28 are "later" (roadmap/other-core.md:16-17) and wave 2 (tiers.md:27); L18 to L23 are "later" (other-core.md:10-15) and wave 3 (tiers.md:33).
Final position: B, with the checker Sonnet specifies plus a rule that a "later" item enters a wave only by a stated decision.
Confidence: high.
Still divides the panel: nothing.

### Q8.2

All three take B: B29, B1, B31, B7, B9, B6, with B29 as the first item and not a hold.
The table encodes it (q1-core.md:30, :34-37, :48, :50); the reason is the `set_shapes_N` deletion (backend/plan/isel.md:34).
My round-1 risk stands unanswered: B7 "needs each operation's shape rule in Rust, sharing a table with the front end where one exists" (isel.md:32), and L3 is Q2 (q2-core.md:9).
B7 in Q1 writes shape rules that R62 wants in one module; no table edge records it, so it belongs in the register (Q8.11), not in the order.
Final position: B, unchanged, with the B7 against L3 duplicate as a named risk.
Confidence: high on the order, medium on the risk.
Still divides the panel: nothing on the order.

### Q8.3

Strongest opposing argument (Sonnet): keep both ID spaces and type the dependency tokens ("item:" or "spec:"), with a checker that rejects an untyped or unknown token; renaming S1 to S9 edits every plan and ruling that cites them for no gain once tokens are typed.
Rebuttal: the checker is the decision, and both B and C need it; T1 depends on S10 (roadmap/dependencies.md:47), which no item and no spec section defines, so a prefix would not have caught it without a resolver.
A "spec:" token legitimizes an edge that schedules nothing: a dependency on a spec section is not a dependency on work.
The three rows that carry one (T1, T5, T8 at q2-core.md:59, :61, :64) already have item replacements in tiers.md:59-61 (L1; L1; T5 and L3).
So the dependency column holds item IDs only, and the collision is removed at the source by giving the nine shared items a free letter.
Opus's LLVM namespace point agrees.
Concession: Sonnet is right that the chain in load.md (S1 to T5 at roadmap/load.md:38-39) is a live consequence, so the checker lands before any chain is published.
Final position: B plus D, with a checker that resolves dependencies against item IDs only and fails on anything else; unchanged.
Confidence: medium-high.
Still divides the panel: whether a spec-section reference may sit in a dependency column at all (Sonnet yes with a prefix, Opus and I no).

### Q8.4

Strongest opposing argument (Sonnet, and Opus in the same shape): three sized sub-items, not my two; (a) L3 rule text after L2, (b) module and checker in taidl (L12, S4, F1, O6) with the exit "shapes.py lives in taidl and the S8.3 table is generated", (c) consumer deletions (O7, B23, then D5); every edge from the front end to a generator reversed.
Concession: the three-part form is better than my two, because the rule text is a reader-facing deliverable with its own exit (the spec page), and the deletions are already D5's work (q2-supporting.md:44 lists B23, O6, O7 under D5), so (c) adds no fourth owner if it is written as D5 sub-items.
The cycles are verified as edge-direction errors: L3 needs O6 and O7 (q2-core.md:9), O6 needs L3 (:41), O7 needs B23, L12, O6 (:42), B23 needs L12 and L3 (:29), L12 needs L3 (:11).
The hidden weight is 31 by table sizes (L3 6, L12 6, S4 6, F1 6, O7 3, B23 3, O6 1; q2-core.md and q1-supporting.md:45), against 6 in the draft (tiers.md:43), not the agenda's 34.
A residual risk no one named: B23 also depends on B7 (q2-core.md:29), so part (c) in the backend waits on the Q1 e-graph slice, and part (b) must not.
T18 and T20: all three take E; T20 "binds T4 and T18" (tests/decisions/rulings.md:22), the edge at q2-core.md:76 is reversed, and T20 goes next to T22 in wave 1.
Final position: D in three sub-items as Sonnet lists them, with (c) as D5 sub-items; E for T20; changed from C.
Confidence: medium-high.
Still divides the panel: the count of parts only, which the moderator can settle on the exit tests.

### Q8.5

Strongest opposing argument (Opus): A16 moves to wave 1, not wave 2, and taidl cuts PEP 440 pre-release tags at each milestone.
Concession: A16 has no dependency (q1-core.md:88; my round-1 cite of :86 was the wrong line), and the first pre-release at M1 needs the release order before M1, so wave 1 is the only place that makes the first tag reviewable.
A18 depends on A16 alone (q2-core.md:87), so A18 runs against every pre-release and the contract churn risk (work/roadmap_extra/README.md:29) gets a name per bump.
Opus's reading of decision 10 holds: pip does not install a pre-release unless asked, so one coordinated major release remains one.
Final position: A plus D, with A16 in wave 1; changed on the wave.
Confidence: high.
Still divides the panel: nothing.

### Q8.6

All three take B for every merge and D for the merges and the drop that rest on R5, R20, R29.
I add verified evidence that makes the O10 drop a direct conflict, not a hold: O10's text is "owner decision 17: the output reuses the marked input's storage; nothing under `taidl_to/` reads `set_inplace` today" (oracle/plan/fidelity.md:33), and the oracle bug list records that the output is stored only into its own region (oracle/plan/bugs.md:18).
Removing `set_inplace` (R5, minutes/rulings.md:15) changes how the mark is derived, not whether the oracle must honor it; the drop at tiers.md:52 leaves decision 17 with no oracle item.
Opus's point that B30 keeps storage reuse in the backend (q1-core.md:49) shows the asymmetry: the backend has an item, the oracle has none.
Final position: B plus D, and O10 is reinstated now under decision 17 rather than held; unchanged in form, stronger in the O10 case.
Confidence: high.
Still divides the panel: nothing.

### Q8.7

Strongest opposing argument (Sonnet): B21 and B22 can stay in M1 because S3 depends on them (q2-supporting.md:31), not the reverse.
Concession: the table direction stands until the SD2 question is ruled; the M1 exit "the backend rejects a region value whose count or element type mismatches" (work/roadmap_extra/milestones.md:13) is checkable with B21, B22 in M1.
The residual is a double-work risk: SD2 says a copy is written in order to be deleted by D5 (reviews/shared-diagnostics/review.md:19), so it goes in the register.
All three converge on A with D: five outcome milestones, item lists derived from the exit criteria, the tag as the last event of M5, A18 before the tag, S9 as the follow-on under R40.
I keep the standing criterion from M1 onward, "install from the milestone pre-release tag and run the README snippet", which Q8.5 now makes possible from M1 since A16 sits in wave 1.
Final position: A plus D, with the standing install criterion from M1; changed on B21, B22 and on the start of the criterion.
Confidence: medium-high.
Still divides the panel: nothing material.

### Q8.8

All three take C plus D: mechanical criteria map to a named CI job, "anyone" criteria use a cold-start trial by people outside the team, and each milestone adds a negative criterion.
Sonnet's trial definition (two outside people, fresh environment, pinned release, README and tutorial, one small ISA description, a stated time limit) is the checkable form and I adopt it.
R63 holds the trial to the migrated tutorial (minutes/rulings.md:35), so the trial waits on A10, which Q8.10 moves to wave 1.
Final position: C plus D with Sonnet's trial definition; unchanged.
Confidence: high.
Still divides the panel: the trial's numbers, which are the owner's call.

### Q8.9

All three agree: order only before M1, recompute weights and chain after Q8.1 to Q8.4, measure the assist factor in wave 1, rebaseline at M1.
On the arithmetic: load.md:18 gives 163 plus 254 and lists 10 later items without a weight; the later items weigh 26 by other-core.md sizes, so the total is 443, not Opus's 427; the factor against 78 to 104 weeks is 4.3 to 5.7.
Sonnet's corrected chain of 36 to 40 weeks is not verified by me, but its direction is: the published 59 (load.md:22) runs through the two cycles verified under Q8.4 and the S1 to T5 misread (load.md:38-39).
Sonnet's finding that the 175 draft IDs sum to about 414 against the stated 440 (tiers.md:55) is also unverified by me and is a reason to regenerate weights rather than trust them.
Final position: A now, C after Q8.1 to Q8.4, D at M1; unchanged.
Confidence: high on the arithmetic, medium on the remedy.
Still divides the panel: nothing.

### Q8.10

All three take B: one gate row in wave 1 with a named outcome per gate, A10 and the Ascend study in it.
Opus settles the O19 edge I questioned: the O8 edge is "so the baseline outputs are the fixed ones" (oracle/plan/architecture.md:24), a correctness edge for O20's comparison, not for the compile-time measurement, so the spike drops the O8 edge and keeps O21 (Q1, no dependency, q1-core.md:62).
The bound is set by the owner before the spike runs (architecture.md:23; reviews/rulings/audit.md:30), so the gate row carries the bound as an owner question beside T22.
Final position: B, with the O19 to O8 edge dropped and the R28 bound as a wave 1 owner question; unchanged, sharpened.
Confidence: high.
Still divides the panel: nothing.

### Q8.11

All three agree: extend the list, phrase each tripwire as an observable, name the exit criterion it endangers, add a criterion where none covers a risk, and keep D's categories as a tag.
Entries this round adds, each with its tripwire: O10 dropped against decision 17 (tripwire: no oracle test stores an output into the marked input's region); B7 shape rules in Rust against the L3 module (tripwire: a shape rule exists in two places after wave 2); B21, B22 written then deleted (tripwire: D5 deletes a check landed in the same quarter); "later" items scheduled without a decision (tripwire: the Q8.1 checker finds an other-core.md item in a wave).
The R18 entry leaves the register since the tables agree (q1-core.md:34-37).
Final position: B plus C without the drop rule; unchanged.
Confidence: high.
Still divides the panel: nothing.

## Escalation to the owner

Decision 11 and the direction "nothing is cut for capacity" are jointly satisfiable only at an assist factor of 4.3 to 5.7 (443 unassisted weeks against 78 to 104; roadmap/load.md:18 plus roadmap/other-core.md), and no document states or measures such a factor.
This is not a counter to either, but the owner should decide now what happens at M1 if the measured factor is lower: dates float with scope fixed, or the owner re-opens scope then.
