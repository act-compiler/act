# Panel 8 agenda: roadmap

Scope: waves, milestones, dependencies, risks.
The draft waves and milestones (work/roadmap_extra: README.md, tiers.md, milestones.md) are one option per question, never a default.
Paths are relative to /home/user/act/plans unless they start with work/ or a repo name.

## Standing constraints for every question

- Owner direction for this panel: nothing is cut for capacity.
- Owner direction for this panel: the engineers develop with Claude.
- Owner direction for this panel: the goal is a project anyone can use.
- Owner direction for this panel: milestones are defined by outcomes.
- Owner decision 10: breaking changes are allowed and batched into one coordinated major release (TAIDL v3 and matching generator releases).
- Owner decision 11: size the roadmap for 3-4 engineers over two quarters.
- Owner decisions 5, 12, 13: planning only, no edits outside plans/, no GitHub issues, no pull request.
- Owner decision 16: no undefined behavior in v3.
- Source of truth for numbers: the generated tables under roadmap/ (q1-core.md, q1-supporting.md, q2-core.md, q2-supporting.md, dependencies.md, load.md).

### Q8.1: Which item set and dependency data feed the waves?

- Context:
  - The quarter tables and the draft waves cover 175 items, and every ID appears in exactly one wave or merged row.
  - The track plans define further items that no table lists: B29 to B34, O19 to O24, and L29 to L31.
  - Some dependency data differs between the tables and the track plans, for example B6 against B7.
- Options:
  - A: Keep the 175-item tables as the base and patch the differences by hand in tiers.md.
  - B: Regenerate the quarter tables from the track plans, then rebuild the waves from the new set.
  - C: Waves cite the track plans directly, and the quarter tables are dropped.
  - D: Add the missing items to the waves as a separate row group and leave the tables unchanged.
- Evidence:
  - backend/plan/search.md:9 (B29), backend/plan/isel.md:10 (B31), backend/plan/extractor.md:12 (B30).
  - oracle/plan/architecture.md:9 (O19), oracle/plan/checks.md:12 (O24), language/plan/spec.md:19 (L29), language/plan/frontend.md:17 (L30).
  - work/roadmap_extra/tiers.md:55 (the check line states 175 IDs).
  - roadmap/load.md (84 items in Q1, 91 in Q2).
- Binding constraints:
  - Owner direction: nothing is cut for capacity, so an omitted item is a scope decision that needs a stated reason.

### Q8.2: What is the order of the backend e-graph items (B6, B7, B9) and the deterministic-search gate?

- Context:
  - The quarter table makes B7 depend on B6 and B9 on B7.
  - R18 orders typed payloads, then B7 and B9, all before B6.
  - The backend track plan already lists B6 after B7 and B9, and adds B31 (typed payloads) and B29 (deterministic search mode) before them.
- Options:
  - A: Table order (B6, B7, B9), as the draft waves keep until a re-ruling.
  - B: R18 order (B31, B7, B9, then B6), following the backend track plan.
  - C: Interleave by outcome: one e-graph slice that types payloads and shapes first, with B6 and B12 in the data form of R19 after it.
  - D: Hold the whole group until the B29 gate (identical chosen assembly and candidate logs) exists, then choose.
- Evidence:
  - roadmap/q1-core.md:32-35 (B6, B7, B9 rows).
  - backend/plan/isel.md:10-13 (B31, B7, B9, B6 rows).
  - backend/plan/decisions.md:8, :11 (test gate and e-graph order rows).
  - minutes/rulings.md R17, R18, R19.
  - work/roadmap_extra/tiers.md:63 and README.md:30 (the conflict is recorded as a risk).
  - work/roadmap_extra/milestones.md:26 (M2 exit "goldens unchanged under R17"; no table item names the deterministic mode).
- Binding constraints:
  - None from the owner decisions; R17 to R19 are draft answers.

### Q8.3: How are item IDs kept from colliding with other ID spaces?

- Context:
  - Shared-code items S1 to S9 share letters with spec sections S1, S8 (S8.3, S8.4), S9, and S10.
  - T1, T5, and T8 list spec sections as dependencies on S* items, and T1 depends on an S10 that no plan defines.
  - Other collisions: backend items cite diagnostics-style labels (D1 to D8) for defects while D1 to D14 are diagnostics items; bug ids X1 to X10 against error class X; semantics items F* against a proposed fault scope F; rulings R* against error class R.
- Options:
  - A: Prefix every spec section reference in roadmap files (for example "spec S8.3"), and leave item IDs as they are.
  - B: Rename one side by track letter (for example shared items to a new letter), and update every index that names them.
  - C: Keep both and make dependency columns carry a typed prefix, "item:" or "spec:".
  - D: Rename only the colliding labels in the backend items, the bug ids, and the diagnostic classes, and keep S* items and spec section ids.
- Evidence:
  - roadmap/dependencies.md (T5 depends on S9, T8 on S8, T1 on S1, and the undefined S10 table).
  - roadmap/q2-core.md:56, :58, :61 (T1, T5, T8 rows).
  - roadmap/q1-core.md (B21, B22, B24, B25 rows carry "(D1)", "(D2)", "(D4)", "(D5)").
  - roadmap/q2-core.md:37 (B23 row carries "(D3)").
  - work/roadmap_extra/tiers.md:59-61 (corrections to T1, T5, T8).
  - reviews/shared-diagnostics/review.md SD15, SD20 (index line, X collision).
  - minutes/rulings.md R44 (L numbers the S9 rows).
- Binding constraints:
  - Writing rule: ASCII only, so no section sign.

### Q8.4: How are the dependency cycles resolved, and what is the shape-and-type item under R62?

- Context:
  - Shape and type rules have a cycle in the tables: L3 needs O6, O6 needs L3; L12 needs O7, O7 needs L12; O7 needs B23, B23 needs L12.
  - T18 depends on T20 and T20 depends on T18; the longest chain in load.md runs through both.
  - R62 merges L3, L12, O6, O7 into one M item; the draft waves absorb L12, O6, O7, S4, B23, and F1 into L3 at size L; SD1 proposes S4 as the single item; the audit asks for L or larger and a split.
- Options:
  - A: One item L3, size L, with S4, B23, F1, L12, O6, O7 as merged rows of weight 0 (the draft waves).
  - B: One item named S4 (SD1), size M, with D4 only attaching clause ids.
  - C: Two items: module move into taidl, then front-end checks including the R31 additions; size the pair L or larger.
  - D: Keep the merged items as separate sub-items, each with its own exit test, ordered without cycles.
  - For T18 and T20: E: decision item T20 first (draft waves); F: fold T20 into T18 as its first step; G: make T20 an owner decision recorded before wave 4.
- Evidence:
  - reviews/shared-diagnostics/review.md SD1, SD2.
  - reviews/rulings/audit.md row R62.
  - roadmap/q2-core.md:9, :12, :37, :45, :46, :71, :73.
  - roadmap/q2-supporting.md:32 (S4 row).
  - roadmap/load.md (longest chain, weight 47).
  - work/roadmap_extra/tiers.md:43 and :62.
  - work/roadmap_extra/README.md:23 (L3 is the hinge).
- Binding constraints:
  - Owner direction: nothing is cut for capacity, so the merged weight (34 unassisted weeks across L3 and the six absorbed items, counted as 6) is a sizing question, not a cut.

### Q8.5: How do the generators pin taidl before the v3 tag exists?

- Context:
  - The corpus runner and the generator CI consume "the pinned taidl release" in wave 4 (T16, T19), while the v3 tag is in wave 5 (S8).
  - The release order and checklist (A16) is merged into S8 in wave 5, yet A18 depends on it, and S9 and A18 are placed after the tag.
  - Front-end contract churn after S2 breaks generator CI on each pin bump.
- Options:
  - A: Pre-release tags of taidl at each milestone, with generators pinning the latest.
  - B: Pin a commit or branch until the tag, and cut one tag in wave 5 (the draft waves).
  - C: Cut the contract slice (S2, S3) as an early tag, and the rest as the final tag.
  - D: Move the A16 checklist and a release-order step to the start of wave 2, ahead of S8.
- Evidence:
  - roadmap/q2-core.md:84 (A18 row), roadmap/q1-core.md:86 (A16 row).
  - roadmap/q2-supporting.md:36 (S8 row), :37 (S9 row).
  - work/roadmap_extra/tiers.md:41-42, :50.
  - work/roadmap_extra/README.md:29 (contract churn risk).
  - reviews/shared-diagnostics/review.md SD19.
  - minutes/rulings.md R40.
- Binding constraints:
  - Owner decision 10: one coordinated major release.

### Q8.6: Which cross-track merges and drops stand, and how are merged items' outcomes kept?

- Context:
  - The draft waves merge S7 into T7, A16 into S8, B15 and O17 into T11, F9 into O13, P6 into B10, F3 into L6, F2 into L5, O4 into O2, and drop O10 and O9.
  - Merged items count as weight 0, and their dependencies move to the absorber.
  - Some merges rest on rulings the audit marks questionable (R5, R20, R29), and R5 keeps `set_inplace` in the semantics track.
- Options:
  - A: As drafted: merged rows of weight 0, dropped items only where a ruling removes the work.
  - B: Keep each merged ID as a named sub-item under the absorber with its own exit criterion.
  - C: Merge only where both tracks' plan texts describe the same work; list the rest as separate items.
  - D: Hold each merge or drop that rests on a questioned ruling until the owning panel decides.
- Evidence:
  - work/roadmap_extra/tiers.md:43-53 (merged and dropped rows).
  - reviews/rulings/audit.md rows R5, R20, R29, R62, RA13.
  - minutes/rulings.md R5, R20, R29, R60.
  - reviews/shared-diagnostics/review.md SD18 (S7 against T5 and T6).
- Binding constraints:
  - Owner decision 17: the output reuses the marked input's storage.
  - Owner decision 16: no undefined behavior.
  - Owner direction: nothing is cut for capacity.

### Q8.7: How is the set of milestones structured?

- Context:
  - The draft gives five milestones, one per wave: M1 spec without undefined behavior, M2 one front end, M3 generators follow the spec, M4 tests prove it, M5 usable by anyone.
  - Owner direction defines milestones by outcomes, and the goal is a project anyone can use.
  - Some items sit in a different milestone than their exit criterion needs: M1 exit "all case-study sketches parse" while C5 to C8 are M2 items; M1 backend rejection (B21, B22) while SD2 makes them calls to the S3 validation in M2; M5 exit "on the v3 tag" while S9 and A18 follow the tag.
- Options:
  - A: Five milestones as drafted, with the item lists corrected to match each exit criterion.
  - B: Outcome milestones that cut across tracks, such as "write an ISA description", "install and run", "trust the results", "learn from one tutorial".
  - C: Two milestones: the v3 spec and front end, then the v3 release with tests and tutorial; waves stay internal.
  - D: Five milestones with the tag moved to the end of M5 and post-tag items listed as a separate follow-on list.
- Evidence:
  - work/roadmap_extra/milestones.md:7-63 (M1 to M5).
  - work/roadmap_extra/milestones.md:13, :15, :50, :52.
  - roadmap/q1-supporting.md and q2-supporting.md (C1 to C13).
  - reviews/shared-diagnostics/review.md SD2.
- Binding constraints:
  - Owner direction: milestones are defined by outcomes.
  - Owner direction: the goal is a project anyone can use.

### Q8.8: What makes an exit criterion checkable, and what proves "usable by anyone"?

- Context:
  - Exit criteria in the draft mix tests ("a CI test fails on drift"), counts ("two case-study users"), and prose ("no silent fallback remains", "C13 finds no undefined case").
  - The M5 table pairs six criteria with a "how we check" column, but the install check names a fresh environment and the tutorial check names CI, not a person.
  - A milestone is done "when every criterion holds, not when a date passes".
- Options:
  - A: Every criterion maps to a named CI job or test item; prose criteria are rewritten or dropped.
  - B: Criteria stay as prose, and a reviewer signs each at milestone close.
  - C: Hybrid: mechanical criteria are tests, and the "anyone" criteria use a cold-start trial by a person outside the engineering team.
  - D: Add a negative criterion per milestone (what must no longer exist, such as the C++ candidate parser).
- Evidence:
  - work/roadmap_extra/milestones.md:5, :11-15, :21-27, :33-37, :43-46, :54-63.
  - work/roadmap_extra/README.md:5-6 (end state), :11-18 (v3 scope).
  - minutes/rulings.md R63 (tutorials as verification), R48, R49.
- Binding constraints:
  - Owner direction: milestones are defined by outcomes.
  - Owner decision 6: `pldi26` is the latest tutorial and dates from about ACT 1.2.

### Q8.9: How do the waves relate to capacity, time labels, and the critical path?

- Context:
  - Unassisted wave weights are 66, 118, 127, 52, and 25, total 388 engineer-weeks over 158 scheduled items (merged items add 50, dropped 2).
  - Three to four engineers over two quarters is 78 to 104 engineer-weeks unassisted; the plan assumes Claude-assisted development and states no assist factor.
  - The longest chain in load.md (weight 47) rests on the cyclic and mis-IDed dependencies; the README lists a hand-written critical path instead.
- Options:
  - A: Order only: waves and milestones carry no dates or quarter labels.
  - B: Keep indicative quarter labels per milestone, with unassisted weights shown beside them.
  - C: Recompute the critical path and weights after Q8.1 to Q8.4 resolve, and publish them with the milestones.
  - D: Measure the assist factor on wave 1 and rebaseline the later labels at M1.
- Evidence:
  - work/roadmap_extra/tiers.md:55 (weights check; wave sums reproduce from the item sizes in the quarter tables).
  - roadmap/load.md (track weights and chain).
  - work/roadmap_extra/README.md:19-24 (critical path).
  - work/roadmap_extra/milestones.md:9, :19, :31, :41, :50 (indicative timing).
  - reviews/rulings/audit.md row RA12.
- Binding constraints:
  - Owner decision 11: size the roadmap for 3-4 engineers over two quarters.
  - Owner direction: nothing is cut for capacity.
  - Owner direction: the engineers develop with Claude.

### Q8.10: Which gates and long-lead inputs are scheduled up front?

- Context:
  - Several items wait on an input that no wave schedules early: owner approval of `hypothesis` (R45) and the StableHLO interpreter (R29), the R28 unrolling spike and its undefined bound, the Ascend case study (R15) placed at the start of C12 in wave 5, and the A10 tutorial check placed in wave 5.
  - The R59 rule (two case studies per extension) is applied in wave 3, before the Ascend study exists.
  - Wave 1 has one owner-question item (T22) and no spike item.
- Options:
  - A: As drafted: T22 in wave 1, the other gates handled when their wave starts.
  - B: A gate row in wave 1 holding every owner approval, spike, and study with a named outcome each.
  - C: Place each gate one wave before its dependent items.
  - D: Move A10 and the Ascend study to wave 1 or 2 and keep the other gates as drafted.
- Evidence:
  - work/roadmap_extra/tiers.md:41 (C12 in wave 5) and the wave 1 row "Owner questions and act cleanup" (T22).
  - work/roadmap_extra/README.md:31-35 (risks on the R28 spike, R29, R45, R59, R63).
  - roadmap/q2-core.md (A10, A11 rows) and q1-core.md (A10 row has no dependency).
  - oracle/plan/architecture.md:9 (O19 spike), reviews/rulings/audit.md rows R15, R28, R29.
  - minutes/rulings.md R15, R28, R29, R45, R59, R63.
- Binding constraints:
  - Owner decisions 5, 12, 13: planning only.
  - Dependencies on new packages wait on owner approval (R29, R45 state this).

### Q8.11: What does the risk register contain, and how is each risk noticed?

- Context:
  - The draft lists eight risks, each with a way to notice it, and no owner, response, or link to a milestone.
  - Reviews raise further candidates: v3 migration for breaking changes (RA11), R56 against i64 checking (audit), contract exposure of ANTLR trees (SD4), compression of waves 3 to 5 (127, 52, 25 weeks).
  - Some risks are already conflicts in the data, such as R18 against the table order.
- Options:
  - A: Keep the eight risks as drafted.
  - B: Extend the list from the reviews and audit, and give each risk a tripwire phrased as an observable.
  - C: Tie each risk to the milestone exit criterion it endangers, and drop risks that no criterion covers.
  - D: Split into schedule risks, technical risks, and dependency risks, each with a fallback outcome.
- Evidence:
  - work/roadmap_extra/README.md:26-35 (risk list).
  - reviews/rulings/audit.md rows R56, R62, R28, R29, RA11, RA12.
  - reviews/shared-diagnostics/review.md SD3, SD4, SD6.
  - work/roadmap_extra/milestones.md (exit criteria per milestone).
- Binding constraints:
  - Owner decision 10: breaking changes are batched into one release.
  - Owner decision 16: no undefined behavior in v3.

## For the authors

- tiers.md and README.md: "175 plan items" excludes B29 to B34, O19 to O24, and L29 to L31 named in the track plans.
- README.md and tiers.md: the Merged rows keep dependencies that point at other merged items (L12, O7, B23, S4); state the absorbed edges once under L3.
- q2-core.md: T5 and T8 name spec sections S9 and S8 as items; dependencies.md lists only T1 and S10 as undefined.
- q2-core.md: T18 and T20 depend on each other; the corrections list fixes it in tiers.md only.
- load.md: the longest chain contains the T18/T20 cycle and the L12/B23/O7 cycle, so its weight 47 is not a path length.
- milestones.md: the M1 exit "all case-study sketches parse" lists C1 to C4, C9 to C11, C13 as items; C5 to C8 are M2 items.
- milestones.md: M5 lists S9 and A18 as items while the exit holds "on the v3 tag" and tiers.md places both after the tag.
- milestones.md: the M2 exit "goldens unchanged under R17" has no item that builds the deterministic search mode.
- README.md (roadmap_extra): the Files section names q1-core.md, q1-supporting.md, q2-core.md, q2-supporting.md, dependencies.md, load.md; they live in roadmap/, not beside it.
- README.md (roadmap_extra): the R18 conflict appears as a risk and as a correction; keep it in one place.
- q1-core.md and q2-core.md: B21 to B25 carry "(D1)" to "(D5)" labels that read as diagnostics items D1 to D5.
- SD3 (S2 listed after B13 in the table, before it in the text) and SD19 (S9 after S8) apply to the wave order and need no further panel question.
