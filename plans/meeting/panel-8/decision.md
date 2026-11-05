# Panel 8 decision: roadmap

| Question | Decision (one line) | Confidence | Owner? |
| -------- | ------------------- | ---------- | ------ |
| Q8.1 | B: rebuild the waves from the regenerated 190-item tables, gated by a graph checker | high | no |
| Q8.2 | B: B29, B1, B31, B7, B9, B6; B29 is the first item, not a hold; B7 shape table checked against taidl | high (order), medium (B7 table) | no |
| Q8.3 | B plus D: shared items take a free letter, defect labels and bug ids renamed, dependency columns hold item IDs only | medium-high | no |
| Q8.4 | D: three acyclic shape-and-type steps at table sizes (31 weeks); E: T20 first, in wave 1 | medium-high | no |
| Q8.5 | A plus D: taidl pre-release tags that generators pin; A16 un-merged into wave 1 | high (mechanics) | yes: decision 10 reading |
| Q8.6 | B plus D: every merge a named sub-item; O10 reinstated; F2 and S7 un-merged; holds on R20, R29, R56 | high (O10, F2), medium-high (rest) | no |
| Q8.7 | A plus D: five outcome milestones, item lists derived from exit criteria, v3 tag closes M5 | medium-high | no |
| Q8.8 | C plus D: CI jobs for mechanical criteria, cold-start trial for "anyone", one negative criterion per milestone | high | yes: trial parameters |
| Q8.9 | No dates before M1; recompute weights and chain; measure the assist factor in wave 1; dates move, scope does not | high | no (owner answered) |
| Q8.10 | B: one wave 1 gate row with a named outcome per gate; the O8 edge moves from O19 to O20 | high (row), medium-high (O19 edge) | no |
| Q8.11 | B plus C: extended register, observable tripwire, endangered exit criterion, response, category tag | high | no |

## Standing owner direction
- Nothing is cut for capacity; the engineers develop with Claude; milestones are defined by outcomes.
- If the measured assist factor is lower than needed, dates move and scope does not.
- Owner decision 23, as clarified: this plan sequences oracle items on the current path; an fsim-lib migration is the follow-up direction and appears once, as a risk in Q8.11.

## Q8.1: item set and dependency data
- Decision: B. The base is the regenerated quarter tables (190 IDs: 88 in Q1, 92 in Q2, 10 later), and the waves are rebuilt from them by a checker.
- Reasoning:
  1. The tables already list B29 to B34, O19 to O24, and L29 to L31, so the 175-item draft is stale, not the tables.
  2. The draft wave 1 cannot start: it holds B1 and B3, which depend on B29 in no wave, and O1, which depends on O20.
  3. A hand patch (A) or a side row group (D) leaves the same drift; dropping the tables (C) loses the one machine-checked view.
  4. The checker fails on an ID in zero or two rows, an unknown dependency token, a cycle, or a dependency on a later wave.
  5. An item that other-core.md marks later (L9, L18 to L23, L27, L28) enters a wave only with a stated reason, since nothing is cut but nothing is silently pulled forward either.
  6. Where a track plan and a table disagree, the track plan's edge wins and the table is regenerated.
- Evidence: roadmap/load.md total row; roadmap/q1-core.md B1, B3, B29 rows; roadmap/q2-core.md O1 row; roadmap/other-core.md L9 to L28 rows against work/roadmap_extra/tiers.md:27, :33.
- Dissent: none.
- Confidence: high.

## Q8.2: backend e-graph order
- Decision: B. Order B29, B1, B31, B7, B9, B6, with B29 the first scheduled item and B3 the gate built on it.
- Reasoning:
  1. The regenerated table already encodes this chain: B1 needs B29, B31 needs B1, B7 needs B31, B9 needs B7, B6 needs B7 and B9.
  2. The backend plan puts B7 before B6 so `set_shapes_N` is never regenerated, and verifies by "no `set_shapes_` in the crate".
  3. Holding the group on B29 (D) is unnecessary because B29 is itself the head of the chain.
  4. B7 needs each operation's shape rule in Rust while the taidl rules (L3, S4) land later, so a second copy is written first.
  5. Blocking B7 on L3 would put the backend lane behind the longest front-end chain; a checked copy is the cheaper route.
  6. So B7 carries an exit line naming its rule source, and a named agreement test compares the Rust table with the taidl shape module once S4 lands.
  7. M1 or M2 lists B29 and B3, so the M2 exit "goldens unchanged under R17" has items behind it.
- Evidence: roadmap/q1-core.md B1, B6, B7, B9, B29, B31 rows; backend/plan/isel.md:32 (shape rule in Rust), :34 (order), :35 (verification).
- Dissent: the B7 duplicate is a risk entry only (Fable) or an exit line (Sonnet) rather than an extra test item (Opus); the decision carries all three, which cost one test.
- Confidence: high on the order, medium on the agreement-test remedy.

## Q8.3: ID collisions
- Decision: B plus D. Shared items S1 to S9 take a letter no other ID space uses (default U), the "(D1)" to "(D5)" labels on B21 to B25 and the bug ids X1 to X10 become free text or a distinct prefix, and spec section ids stay.
- Reasoning:
  1. The live errors are bare tokens read in the wrong space: T5 on S9, T8 on S8, T1 on S1 and an undefined S10, and the load.md chain through S1 to T5.
  2. A typed prefix (C) still fails silently whenever an author omits it, unless the checker rejects untyped tokens, which is the rename rule at higher cost.
  3. A dependency on a spec section schedules no work; the three such rows already have item replacements (L1; L1; T5 and L3).
  4. So dependency columns hold item IDs only, and the checker resolves every token against a registry of reserved prefixes.
  5. Spec section ids ship in error messages and test tags, so they are the side that stays.
- Evidence: roadmap/dependencies.md (T1 on S10); roadmap/q2-core.md T1, T5, T8 rows; roadmap/q1-core.md B21 to B25 rows; work/roadmap_extra/tiers.md:59-61; minutes/rulings.md R44.
- Dissent: Sonnet would allow a spec-section token with a prefix in a dependency column; the reasoning in step 3 outweighs it.
- Confidence: medium-high; the letter is cosmetic.

## Q8.4: dependency cycles and the shape-and-type item
- Decision: D in three acyclic steps, each absorbed ID kept as a named exit test at its table size.
  - (a) L3 rule text per operation, after L2; exit: every operation has a normative rule a clause can cite.
  - (b) taidl implementation: S4, L12, F1, O6; exit: the shape module lives in taidl and the S8.3 table is generated from it.
  - (c) consumer deletions as D5 sub-items: O7 and B23, after (b); B23 also waits on B7.
- T18 and T20: E. T20 has no dependency, T4 and T18 depend on it, and it sits beside T22 in wave 1.
- Reasoning:
  1. The cycles are edge-direction errors (L3 needs O6 and O7, O6 needs L3; O7 needs B23, L12, O6; B23 needs L12 and L3), so reversing them removes every cycle and no work.
  2. The group weighs 31 unassisted weeks by table sizes (L3 6, L12 6, S4 6, F1 6, O7 3, B23 3, O6 1), against 6 in the draft; one item of size L hides 25 weeks.
  3. Owner decision 25 has the generator copy the front end's shape module into each generated oracle, so the module in step (b) is the single source both consumers read.
  4. T20 "binds T4 and T18", so the table edge T20 -> T18 is reversed; it is test policy, so no owner decision (G) is needed.
  5. The oracle side of step (c), O7, runs after (b) on the current oracle path.
- Evidence: roadmap/q2-core.md L3, L12, O6, O7, B23, T18, T20 rows; roadmap/q2-supporting.md S4, D5 rows; roadmap/q1-supporting.md F1 row; tests/decisions/rulings.md:22; reviews/rulings/audit.md R62.
- Dissent: Fable preferred two steps; the third step is the one with a different owner track and its own exit, so three stands.
- Confidence: medium-high.

## Q8.5: pinning taidl before the v3 tag
- Decision: A plus D. taidl cuts pre-release tags (3.0.0aN, then rcN), generators pin the latest, the first follows S2 and S3, and the v3 tag is the one release.
- A16 is un-merged from S8 and scheduled in wave 1; A18 runs against every pre-release and lands before the final tag.
- Reasoning:
  1. T16 and T19 consume "the pinned taidl release" before S8 exists, so the pin needs a tagged artifact earlier than the tag (B).
  2. A pre-release tag gives each contract bump after S2 a named version, which makes the churn risk observable per bump.
  3. A16 has no dependency and A18 depends on A16 alone, so wave 1 costs nothing and gives the first tag a reviewed release order.
  4. An early contract-only tag (C) is subsumed: it is the first pre-release and carries no separate name.
  5. Whether a published pre-release tag counts as a release under decision 10 is a reading of an owner decision, so it goes to the owner.
- Evidence: roadmap/q1-core.md A16 row; roadmap/q2-core.md T16, T19, A18 rows; roadmap/q2-supporting.md S8 row; roadmap/dependencies.md S2, S3 rows.
- Dissent: Opus places A16 "no later than wave 2"; wave 1 holds because the item has no dependency and gates the first tag.
- Owner options on decision 10:
  - Pre-release tags are not releases: one coordinated major release remains one, pip installs a pre-release only when asked, and every milestone gets an installable artifact for the standing install check.
  - Only the v3 tag is published, with commit pins until then (B): no published version can be mistaken for a release, at the cost of commit pins in generator CI and no installable milestone artifact.
- Confidence: high on the mechanics; the decision 10 reading is the owner's.

## Q8.6: merges and drops
- Decision: B plus D.
  - Every merge stays a named sub-item under its absorber with its own exit criterion and its own table weight.
  - O10 is reinstated now as an oracle item under owner decisions 17 and 22.
  - F2 is un-merged from L5 and stays a semantics item, since its merge rests on R5.
  - S7 is un-merged from T7: its mini_trn merge and generator CI parts are separate work.
  - P6 (R20), F9 (R29), and F3 (R56) are held as separate items until their owning panels rule.
  - The O9 drop stands under R60 as a one-line note, since the work is the removal itself.
  - O4 folds into O3, as the table states, not O2 as the draft states.
- Reasoning:
  1. Weight-0 merged rows hide 50 unassisted weeks and lose each item's exit, which conflicts with outcome milestones.
  2. The O10 drop rests on R5 removing `set_inplace`; owner decision 22 keeps `set_inplace` unless the panels show it truly redundant, and decision 17 binds the oracle to reuse the marked input's storage.
  3. The oracle never reads `set_inplace` today and stores the output only into its own region, so without O10 decision 17 has no oracle item while the backend has B30.
  4. F2's merge text "aliasing by identical region text (R5)" is the overlap rule that decision 22 says does not settle read-after-write.
  5. S7 and T7 have disjoint dependency edges (L8, S3 against T1, T5, T6), which is evidence of different work.
  6. The audit marks R20 and R29 questionable and R56 wrong, so merges resting on them wait.
- Evidence: work/roadmap_extra/tiers.md:43-53; oracle/plan/fidelity.md:33; oracle/plan/bugs.md:18; roadmap/q1-core.md O10, B30 rows; roadmap/other-core.md O4 row; reviews/rulings/audit.md R5, R20, R29, R56; reviews/shared-diagnostics/review.md SD18; owner decisions 17, 22, 24.
- Dissent: holding O10 until the R5 ruling (Opus, Sonnet) rather than reinstating it; decisions 17 and 22 already bind, so reinstatement does not wait.
- Confidence: high on O10 and F2, medium-high on the rest.

## Q8.7: milestone structure
- Decision: A plus D. Five outcome milestones; the checker places each item in the earliest milestone whose exit criterion needs it.
  - M1 exit names its sketches (C1 to C4, C9 to C11, C13); C5 to C8 stay in M2.
  - B21 and B22 stay in M1, since S3 depends on them.
  - The v3 tag is the last event of M5; A18 lands before it; S9 is a listed follow-on.
  - A standing criterion holds from the first milestone that cuts a pre-release tag, M2 at latest: install from that tag and run the README snippet.
- Reasoning:
  1. The draft's defects are item lists that disagree with exits, not the outcome structure itself.
  2. Moving C5 to C8 into M1 adds 18 unassisted weeks that the spec outcome does not need; naming the sketches keeps M1 an outcome.
  3. Cross-track milestones (B) restate the same outcomes; two milestones (C) hide progress for a quarter.
  4. The standing install criterion exercises "anyone can use" at every milestone instead of once.
- Evidence: work/roadmap_extra/milestones.md:13, :15, :50, :52; roadmap/q2-supporting.md C5 to C8, S3 rows.
- Dissent: Fable starts the install criterion at M1; it starts only once a tag exists, and the first follows S2 and S3.
- Confidence: medium-high.

## Q8.8: checkable exit criteria
- Decision: C plus D.
  - Each mechanical criterion maps to a named CI job or test item.
  - Each prose criterion cites the track plan verification line that makes it observable.
  - "Anyone" criteria use a cold-start trial by people outside the team on the migrated tutorial, after A10.
  - Each milestone states one negative criterion a test checks, such as no C++ candidate parser or no `set_shapes_` in the crate.
- Reasoning:
  1. Prose signed by a reviewer (B) is not checkable; dropping prose criteria (A) drops the "anyone" outcome.
  2. The track plans already carry observables, so negative criteria cost little.
  3. A Claude-driven fresh-container run shows the steps execute, not that a newcomer succeeds, so it is an extra CI job and not the trial.
  4. Tutorial evidence before A10 is void, so the trial waits on A10.
- Evidence: backend/plan/isel.md:35; work/roadmap_extra/milestones.md:4, :54-63; minutes/rulings.md R63.
- Owner options on trial parameters: two outside people, fresh environment, pinned release, README and tutorial, one small ISA description, no maintainer help, and a stated time limit (panel default); or one person with a non-author teammate and a recorded transcript as the weaker form.
- Dissent: none.
- Confidence: high.

## Q8.9: capacity, time labels, critical path
- Decision: no date or quarter label before M1; publish recomputed weights and the recomputed chain beside the milestones (C); measure assisted weeks per wave 1 item and set labels at M1 (D).
- Reasoning:
  1. The tables total 443 unassisted weeks: 163 in Q1, 254 in Q2, and 26 for the later items by other-core.md sizes; load.md prints the later count (10), not its weight.
  2. Against 78 to 104 engineer-weeks this implies an assist factor of 4.3 to 5.7, which no document measures.
  3. The published chain (weight 59) runs through the L12, B23, O7 cycle, the T18 and T20 cycle, and the S1 to T5 misread, so it is not a path length.
  4. A label before measurement invites closing on a date, which the milestones text forbids.
  5. By the owner's standing direction, a measured factor below the need moves dates at the M1 rebaseline and never scope.
- Evidence: roadmap/load.md (totals, chain); roadmap/other-core.md sizes; work/roadmap_extra/milestones.md:4-5; reviews/rulings/audit.md RA12.
- Dissent: none; a corrected chain of 36 to 40 weeks is unverified and is not adopted; the checker computes it.
- Confidence: high.

## Q8.10: gates and long-lead inputs
- Decision: B. One wave 1 gate row, each gate with a named outcome.
  - T22 owner questions and T20 test policy.
  - R45 `hypothesis` approved or refused.
  - R29 interpreter approved or refused, covering its build route.
  - R28 bound set by the owner before the O19 spike runs.
  - A10: the list of tutorial syntax that fails on the pinned image and API.
  - Ascend study (R15) as its own item in wave 1 or 2, done before R59 is applied in wave 3.
- Reasoning:
  1. Each gate decides later scope, so placing it when its wave starts (A) stalls that wave.
  2. A10 has no dependency, and the Ascend study must exist before R59 counts case studies.
  3. The O8 edge on O19 guards output baselines, not the compile-time measurement, so it moves to O20; O19 keeps O21 and can run in wave 1.
  4. A spike whose bound is agreed afterwards has no fail point, so the bound precedes the spike.
- Evidence: oracle/plan/architecture.md:20-26; roadmap/q1-core.md A10, O21 rows; minutes/rulings.md R15, R28, R29, R45, R59, R63; reviews/rulings/audit.md R15, R28, R29.
- Dissent: Sonnet would have the oracle owner confirm the O19 edge move; the plan text states the edge's purpose, so the panel moves it.
- Confidence: high on the gate row, medium-high on the O19 edge.

## Q8.11: risk register
- Decision: B plus C without the drop rule; a risk no criterion covers adds a criterion.
- Columns: ID, category tag (schedule, technical, dependency), observable tripwire, endangered exit criterion, response, owner role.
- A response moves a date or reorders work; it never cuts scope.
- Entries joined to the draft eight:
  - Unmeasured assist factor: the M1 ratio falls below 4.3.
  - Graph integrity: the Q8.1 checker fails.
  - B7 Rust shape rules against the taidl module: a shape rule exists in two places without the agreement test.
  - B21, B22 written then deleted: D5 deletes a check landed in the same quarter.
  - v3 migration (RA11): a rejected v3 construct without its replacement and clause id.
  - R56 against i64 checking: generated Rust needs i128 or bignums.
  - ANTLR tree exposure (SD4): a generator imports a parse-tree type.
  - Waves 3 to 5 compression: recomputed wave weights after Q8.1.
  - Later fsim-lib migration (decision 23 follow-up) may reorder oracle work: its plan changes an oracle edge that feeds an exit criterion.
  - Release order written last: no A16 checklist at the first pre-release tag.
  - Later items scheduled without a decision: the checker finds an other-core.md item in a wave.
- The R18 conflict leaves the register, since the tables now agree with R18.
- Reasoning:
  1. A risk without an observable is never noticed, and dropping an uncovered risk removes a warning, so the gap becomes a new criterion.
- Evidence: work/roadmap_extra/README.md:26-35; reviews/rulings/audit.md R56, RA11; reviews/shared-diagnostics/review.md SD2, SD4; roadmap/q1-core.md B6 to B9 rows.
- Dissent: none.
- Confidence: high.

## Escalation to the owner
- Decision 10 and pre-release tags (Sonnet): confirm whether published pre-release tags (3.0.0aN, rcN) that generators pin count as a release; options are in Q8.5.
- Decision 11 against "nothing is cut" (Sonnet, Opus, Fable): 443 unassisted weeks against 78 to 104 need an unmeasured assist factor of 4.3 to 5.7; the owner's standing direction answers it: a lower measured factor moves dates and not scope.
- Trial parameters for the "anyone" criterion: number of outside people and time limit; options are in Q8.8.
