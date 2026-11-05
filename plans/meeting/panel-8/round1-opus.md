# Panel 8, round 1: Opus (program manager)

Assumptions used throughout:
- The regenerated quarter tables are the numeric source of truth: 88 Q1, 92 Q2, and 10 later items, 190 in all (roadmap/load.md:18).
- The draft waves (work/roadmap_extra/tiers.md) were built on the older 175-item set and are one option, not a base.
- A plan is coherent only if one generator produces tables, waves, milestones, and chain from one acyclic graph with one ID space.

### Q8.1

Position: B, regenerate from the track plans (already done for the tables) and rebuild the waves from the 190-item set by the same generator.
1. The tables now hold 190 items, including B29 to B34, O19 to O24, and L29 to L31, so option A's 175-item base is stale.
2. The regenerated data changes wave placement: O1 now depends on O20 (q2-core.md:38), yet the draft puts O1 in wave 1 (tiers.md:22).
3. B6 now depends on B7 and B9, and B7 on B31 (q1-core.md:34-35), so the hand patch of tiers.md:63 is obsolete.
4. Hand patches (A) and side row groups (D) drift each time a track plan changes; one generated source does not.
5. Dropping the tables (C) loses the per-track load and chain checks that load.md computes.
6. The generator must reject cycles and unknown IDs before it emits waves; it currently emits a chain through two cycles (load.md:22-45).
Evidence: roadmap/load.md:18; q1-core.md:22-24, :48-50; q2-core.md:38, :49-53; work/roadmap_extra/tiers.md:3, :22, :55, :63.
Confidence: high.
Would change my mind: track plans that are themselves inconsistent with each other, so that no generated set is better than a curated one.

### Q8.2

Position: B, the track-plan order B29, B1, B31, B7, B9, B6, with B29 as the entry gate rather than a hold on the group (not D).
1. The backend plan records why: B6 first regenerates `set_shapes_N` that B7 deletes (backend/plan/decisions.md:11).
2. The regenerated table already encodes this order: B1 needs B29, B31 needs B1, B7 needs B31, B9 needs B7, B6 needs B7 and B9 (q1-core.md:30, :34-37, :50).
3. B29 has no dependency and is size M (q1-core.md:48), so it can start in week 1 and need not block anything for long.
4. The M2 exit "goldens unchanged under R17" (milestones.md:26) is uncheckable without B29, so B29 must sit in the wave before that exit.
5. Holding the whole group (D) adds latency with no information gain, since the order is already decided by the deletion argument in step 1.
Evidence: backend/plan/isel.md:10-13; backend/plan/decisions.md:8, :11; roadmap/q1-core.md:30-37, :48; work/roadmap_extra/milestones.md:26.
Confidence: high.
Would change my mind: a B29 spike showing the search cannot be made deterministic within size M, which would make the gate the long pole.

### Q8.3

Position: B for shared items (new letter), D for the backend defect labels, plus a machine-checked registry of ID spaces.
1. The collision is not cosmetic: the generator read T8 -> S8 as the release-train item and moved T8 to Q2 (dependencies.md, T8 row; q2-core.md:64).
2. T5 -> S9 likewise binds the corpus format to docker-script consolidation, an item R40 places after v3 (q2-core.md:61; rulings R40).
3. Spec section ids ship to users inside error messages with clause ids (R44), so they are the stable public space and must not move.
4. Item IDs are internal planning labels, so renaming the shared items costs one generator pass and an index update.
5. Prefixes (A, C) rely on every author typing them; the existing errors are exactly a missing prefix.
6. The "(D1)" to "(D5)" tags on B21 to B25 read as diagnostics items (q1-core.md:41-42) and become free text such as "defect 1".
7. LLVM keeps release tags in a dedicated namespace (`llvmorg-X.Y.Z-rcN`) so no tool confuses them with other names; the same discipline applies to plan IDs.
8. The generator lists the reserved prefixes per space and fails on any dependency that is not a defined item ID.
Evidence: roadmap/dependencies.md (T8, T5, T1 rows; S10 unknown); q2-core.md:59, :61, :64; q1-core.md:41-42; https://www.llvm.org/docs/HowToReleaseLLVM.html.
Confidence: medium-high.
Would change my mind: a count showing more committed files cite shared item IDs than spec section ids, making the rename the larger change.

### Q8.4

Position: D in the shape of C: separate sub-items with own exit tests, ordered acyclically; and E, T20 first with no dependency on T18.
1. The cycles are in the data: L3 needs O6 and O7 (q2-core.md:9), O6 needs L3 (:41), O7 needs L12 (:42), L12 needs L3 (:11).
2. The direction that makes sense is spec text first: L3 states the rules, the module implements them, consumers call it.
3. Acyclic order: L3 (rules text, needs L2) -> S4 plus L12 (module and checker in taidl, needs S3) -> O6, O7, B23 (consumers delete copies).
4. One merged item of weight 6 hides 34 weeks of unassisted work (agenda Q8.4); owner direction forbids cutting it, so it needs sized parts.
5. Separate exit tests let the M2 criterion "shapes.py lives in taidl" (milestones.md:22) be checked without waiting for every consumer.
6. T20 is a decision that "binds T4 and T18" (tiers.md:62), so the T20 -> T18 edge (q2-core.md:76) is reversed, not folded.
7. A decision folded into T18 (F) would be taken during deletion, after the tests it governs are already gone.
Evidence: roadmap/q2-core.md:9, :11, :29, :41-42, :74, :76; q2-supporting.md:32; load.md:22-45; reviews/shared-diagnostics/review.md SD1; work/roadmap_extra/tiers.md:43, :62.
Confidence: medium.
Would change my mind: evidence that panel 3 or 6 settles T20 as an owner-level policy, which would make G the right home.

### Q8.5

Position: A plus D: the A16 checklist moves to wave 1, and taidl cuts PEP 440 pre-release tags (v3.0.0aN, then rcN) at each milestone; generators pin the latest pre-release.
1. Owner decision 10 requires one coordinated major release; pre-releases are not releases to users, since pip skips them unless asked.
2. T16 and T19 consume "the pinned taidl release" in wave 4, before the v3 tag in wave 5 (q2-core.md:72, :75; tiers.md:41).
3. A commit pin (B) has no checklist, no changelog, and no name a user or CI log can cite.
4. LLVM tags rc1 to rc3 over about six weeks before final, so integrators test against named candidates.
5. A16 has no dependency (q1-core.md:88), and A18 needs it (q2-core.md:87), so merging A16 into S8 at the end is backwards.
6. With A16 in wave 1, A18 (scheduled compatibility build) can run against each pre-release and catch contract churn after S2.
7. An early contract-only tag (C) is a second major in effect, against decision 10, if later waves still break the contract.
Evidence: minutes/owner-decisions.md:10; roadmap/q1-core.md:88; q2-core.md:72, :75, :87; work/roadmap_extra/tiers.md:41-42, :50; https://www.llvm.org/docs/HowToReleaseLLVM.html; https://peps.python.org/pep-0440/.
Confidence: medium-high.
Would change my mind: an owner reading of decision 10 that counts any published pre-release tag as a release.

### Q8.6

Position: B with C as the filter and D as a hold: each merge keeps a named sub-item with its own exit criterion and weight; merges on questioned rulings wait.
1. Weight 0 for merged rows hides 50 unassisted weeks (tiers.md:55), which distorts every capacity and chain number.
2. A merged ID with no exit criterion has no way to be shown done, so its outcome can vanish without a decision.
3. C applies: merge only where both plan texts describe the same work; S7 against T5 and T6 (SD18) shows that is not always so.
4. The audit questions R5, R20, R29 (reviews/rulings/audit.md), so F2 into L5, P6 into B10, F9 into O13 are holds, not facts.
5. Dropping O10 on R5 needs a check against owner decision 17 and B30, which keep in-place storage reuse (q1-core.md:49).
6. A16 into S8 is reversed by Q8.5, which shows the merge already broke an ordering.
Evidence: work/roadmap_extra/tiers.md:43-55; reviews/rulings/audit.md rows R5, R20, R29; reviews/shared-diagnostics/review.md SD18; minutes/owner-decisions.md:17; roadmap/q1-core.md:49.
Confidence: medium.
Would change my mind: panel rulings that confirm R5, R20, R29 unchanged, which would let those merges stand as sub-items now.

### Q8.7

Position: D, with the item lists derived from the exit criteria (A's correction) and A18 moved before the tag.
1. The five drafted milestones are already outcomes, so B renames them without new information.
2. Two milestones (C) leave one milestone for most of two quarters, with no early checkpoint to measure the assist factor (Q8.9).
3. Each item belongs to the earliest milestone whose criterion needs it; the generator computes this from the criterion-to-item map.
4. That rule moves C5 to C8 into M1 or splits the exit "all sketches parse" (milestones.md:15, :20).
5. The rule moves B21 and B22 next to S3 if SD2 holds, since they become calls to the S3 validation.
6. "Usable by anyone" holds on the final tag, so the tag closes M5; S9 is a post-tag follow-on (R40).
7. A18 depends only on A16 (q2-core.md:87), so once A16 is early, A18 runs before the tag and checks it.
Evidence: work/roadmap_extra/milestones.md:10-15, :20, :50-63; reviews/shared-diagnostics/review.md SD2; minutes/rulings.md R40; roadmap/q2-core.md:87.
Confidence: medium.
Would change my mind: an owner preference for user-facing milestone names, which B serves without changing the item sets.

### Q8.8

Position: C plus D: mechanical criteria map to a named CI job; "anyone" criteria use a recorded cold-start trial by a person outside the team; each milestone adds a negative criterion checked by a test.
1. A criterion is checkable when a named procedure returns pass or fail without a judgment call.
2. Most draft criteria are already mechanical, for example "a CI test fails on drift" (milestones.md:43).
3. Prose such as "no silent fallback remains" (milestones.md:33) becomes a test over the D10 registry or a grep job with an empty-result rule.
4. "Usable by anyone" is about people; a CI install proves only that the pins resolve (milestones.md:56).
5. A cold-start trial is defined: a person outside the team, given only the README and tutorial, installs v3 and finishes the tutorial; the transcript is kept and every blocker becomes an item.
6. Negative criteria (D) are cheap tests: the C++ candidate parser path and `docs/operations.md` must be absent.
7. Reviewer sign-off on prose (B) gives no repeatable result and fails the outcome rule.
Evidence: work/roadmap_extra/milestones.md:4, :14, :26, :33, :43, :54-61; minutes/rulings.md R63; https://www.llvm.org/docs/HowToReleaseLLVM.html (release candidates are tested by people outside the core team before final).
Confidence: high.
Would change my mind: no person outside the team being available, which would leave a Claude-driven fresh-container trial as the fallback.

### Q8.9

Position: C then D: recompute weights and critical path after Q8.1 to Q8.4, publish order first, and rebaseline indicative labels at M1 from a measured assist factor.
1. The regenerated total is 163 + 254 + 10 = 427 unassisted weeks (load.md:18), against 78 to 104 available (decision 11).
2. Meeting decision 11 without cuts therefore needs an assist factor of about 4.1 to 5.5; no source states one.
3. The current longest chain (weight 59) passes through two cycles (load.md:22-45), so it is not a path length.
4. Labels without a measured factor are guesses; wave 1 is the first point where planned and actual weight can be compared.
5. Order-only (A) leaves the owner no signal that decision 11 is at risk until the end.
6. LLVM and PyTorch fix release dates and let scope float; here scope is fixed by owner direction, so dates must float and be re-forecast.
Evidence: roadmap/load.md:18, :22-45; minutes/owner-decisions.md:11; work/roadmap_extra/milestones.md:5, :9; reviews/rulings/audit.md RA12; https://www.llvm.org/docs/HowToReleaseLLVM.html; https://github.com/pytorch/pytorch/blob/main/RELEASE.md.
Confidence: high on C, medium on D.
Would change my mind: an owner-stated assist factor, which would let labels be computed now.

### Q8.10

Position: B: one gate row in wave 1 with every owner approval, spike, and long-lead study, each with a named outcome and a fallback.
1. A gate handled "when its wave starts" (A) makes its wait time sit on the critical path.
2. Owner approvals for `hypothesis` (R45) and the StableHLO interpreter (R29) cost the engineers nothing to request early.
3. The O19 spike sits late because it depends on O21 and O8, and O8 through L16 on L3 (q2-core.md:49, :43, :13).
4. The spike can measure compile time on current literals; its edge to O8 is a correctness edge for O20, not for the measurement.
5. A10 has no dependency (q1-core.md:84), so the tutorial check can report its gap list in wave 1.
6. The Ascend study must exist before R59 counts case studies in wave 3, so it starts in wave 1, not in C12 in wave 5.
7. Each gate states its outcome, for example "approved or alternative named", so a missed gate is visible.
Evidence: work/roadmap_extra/README.md:31-35; tiers.md:24, :41; roadmap/q1-core.md:84; q2-core.md:13, :43, :49; minutes/rulings.md R15, R28, R29, R45, R59.
Confidence: medium-high.
Would change my mind: the oracle track showing the O19 measurement is invalid without O8 rounding, which keeps the spike in place.

### Q8.11

Position: B plus C, without C's drop: each risk gets an ID, an observable tripwire, the exit criterion it endangers, a response, and an owner role.
1. The draft's eight risks have tripwires but no response, so noticing a risk triggers nothing (README.md:26-35).
2. Linking each risk to an exit criterion shows which milestone slips when it fires.
3. A risk that no criterion covers points to a missing criterion; it is added, not dropped.
4. Candidates to add: RA11 v3 migration, R56 against i64 checking, SD4 ANTLR tree exposure, and waves 3 to 5 compression.
5. Candidates to add from this panel: graph integrity (cycles and ID collisions recur on regeneration) and an assist factor below 4.
6. The R18 risk is closed by the regenerated order (Q8.2), so it leaves the register.
7. StableHLO publishes a compatibility policy with stated guarantees; a v3 migration risk needs the same: what breaks, and the guide that covers it.
Evidence: work/roadmap_extra/README.md:26-35; reviews/rulings/audit.md rows R56, R62, R28, R29, RA11, RA12; reviews/shared-diagnostics/review.md SD3, SD4, SD6; https://openxla.org/stablehlo/compatibility.
Confidence: medium.
Would change my mind: a panel 7 ruling that puts the migration guide inside an exit criterion already, which removes RA11 as a separate risk.
