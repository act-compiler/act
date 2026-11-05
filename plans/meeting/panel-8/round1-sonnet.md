# Panel 8, round 1: Sonnet (engineer executing the plan)

## Assumptions and method

- I parsed every table row in roadmap/ (q1-core, q1-supporting, q2-core, q2-supporting, other-core) into an item graph: 190 IDs.
- I applied the draft merges and drops from work/roadmap_extra/tiers.md:43-53 to compute chains; sizes are S = 1, M = 3, L = 6 unassisted weeks.
- My own sums: the table sizes total 443 unassisted weeks over 190 IDs, against 78 to 104 weeks of capacity (3-4 engineers, 26 weeks).
- The implied assist factor is 4.3 to 5.7; nothing in the plan measures it.
- External practice: StableHLO versions by minor bump with weekly patch releases (openxla.org/stablehlo/compatibility).
- External practice: LLVM develops incrementally in-tree, lands on main first, and backports to release branches (llvm.org/docs/DeveloperPolicy.html).
- Owner direction applies throughout: nothing cut for capacity, Claude-assisted engineers, outcome milestones, final milestone "usable by anyone".

## Facts the answers rest on

- F-a: the regenerated tables already contain the 15 IDs the draft waves lack (L29-L31, B29-B34, O19-O24); my script finds exactly 15 table IDs in no wave and not merged or dropped.
- F-b: the regenerated tables already use R18 order: B7 depends on B31, B9 on B7, B6 on B7 and B9 (roadmap/q1-core.md, backend rows).
- F-c: three cycles remain in the tables: L3/O6, the L3/O7/B23/L12 group, and T18/T20 (my DFS over q2-core.md rows).
- F-d: one undefined ID remains: T1 depends on S10 (roadmap/dependencies.md).
- F-e: roadmap/load.md reports a chain of weight 59 that runs through S1 -> T5; that S1 is the shared item (emitter divergences), but T5 means spec section S1.
- F-f: with merges applied, T5 fixed to L1, and T20 before T18, my longest chain is 40 unassisted weeks (B29, B1, B31, B7, L3, L16, O8, O19, O20, O1, O2, O3, O22, O23).
- F-g: if the O19 -> O8 edge is dropped as well, my longest chain is 36 weeks and runs through the backend rewrite items (B6, B8, P3, P4, B10, B11, B19).
- F-h: tiers.md wave 1 holds O1, yet O1 depends on O20 -> O19 -> O8 -> L16 -> L3, which is wave 2 (roadmap/q2-core.md O1, O19, O20 rows).
- F-i: wave 1 holds B1 and B3, which depend on B29, and B29 is in no wave (roadmap/q1-core.md B1, B3, B29 rows).

### Q8.1: Which item set and dependency data feed the waves?

- Position: B. The tables are already regenerated from the track plans (190 IDs); rebuild the waves from them with a machine check, and do not patch by hand.
- Argument:
  1. The agenda's premise of 175 table items is out of date: the tables hold 190 IDs including all 15 listed as missing (F-a).
  2. The remaining work is to rebuild tiers.md from that set, and that rebuild is not cosmetic: wave 1 as drafted cannot run, because B1 and B3 wait on an unscheduled B29 (F-i) and O1 waits on a wave 2 chain (F-h).
  3. Option A (hand patches) repeats the failure that produced the 175 count; option C drops the only artifact that carries sizes and rulings per item; option D schedules the 15 items beside waves they depend on and so reorders nothing.
  4. A checker makes the set auditable: it fails if a table ID is in zero or two wave rows, if a dependency token is not an item ID, or if a dependency sits in a later wave or later in the same wave.
  5. No item is omitted without a stated ruling, which satisfies "nothing is cut for capacity".
- Evidence: roadmap/q1-core.md B1, B3, B29 rows; roadmap/q2-core.md O1, O19, O20 rows; work/roadmap_extra/tiers.md:55; my parse (F-a, F-h, F-i).
- Confidence: high.
- Would change my mind: a reason the track plans, not the tables, hold the authoritative dependency for a given item (then that item's row is corrected at the plan, and the tables regenerated).

### Q8.2: What is the order of B6, B7, B9 and the deterministic-search gate?

- Position: B. Order B29, B1, B31, B7, B9, B6; B12 and B8 after B6 in the R19 data form.
- Argument:
  1. The regenerated table already says B7 after B31 and B6 after B7 and B9 (F-b), so the conflict recorded as a risk in README.md:30 is gone from the data and remains only in the draft waves.
  2. The reason is mechanical: B6 first regenerates `set_shapes_N`, which B7 deletes (backend/plan/decisions.md:11).
  3. B29 has no dependencies and is the root of B1 and B3, so the gate is built first by construction; option D adds no hold, since B29 is already the first item.
  4. Option C bundles B6 and B12 with a slice they do not need; keeping them separate lets the backend lane finish B7 and B9 and ship B6 and B12 once, in the form R19 picks.
  5. M2's exit "goldens unchanged under R17" needs B29 and B3 in M1, so both must be named in the M1 item list.
  6. The B6 form (code or data) is open; the plan row B6 depends on a B6/B12 decision before B6 starts, not before B7.
- Evidence: roadmap/q1-core.md B6, B7, B9, B29, B31 rows; backend/plan/isel.md:10-13; backend/plan/decisions.md:8, :11; work/roadmap_extra/milestones.md:26.
- Confidence: high.
- Would change my mind: evidence that B7 needs rewrite names from B6 (for example a B7 test that keys on rule identity).

### Q8.3: How are item IDs kept from colliding with other ID spaces?

- Position: C with the cheap renames of D. Dependency columns carry typed tokens ("item:" or "spec:"), a checker rejects untyped or unknown tokens, and labels that are only labels are renamed.
- Argument:
  1. The collision already produced a wrong result: load.md's longest chain runs S1 -> T5, reading the spec section as the shared item (F-e), which inflates the chain and misleads scheduling.
  2. Spec section ids are published, tagged in tests (T1), and cited in code, so they cannot move; item IDs are internal, but renaming S1-S9 (option B) edits every plan and ruling that cites them for no gain once the tokens are typed.
  3. Only three dependency rows need a human choice: T1 (S1, S10), T5 (S1, S9), T8 (S8); tiers.md:59-61 already gives the answers (L1; L1; T5 and L3).
  4. A typed prefix lets the generator fail loudly on S10, which is currently a warning in dependencies.md.
  5. The labels (D1) to (D5) on B21-B25 read as diagnostics items D1-D5; rename them to the defect names (for example "region count mismatch") and keep D* for diagnostics.
  6. Option A leaves item-to-item ambiguity in the dependency data, which is what the tools read.
- Evidence: roadmap/load.md chain (S1, T5); roadmap/q2-core.md:56, :58, :61; roadmap/q1-core.md B21-B25 rows; work/roadmap_extra/tiers.md:59-61; roadmap/dependencies.md (S10).
- Confidence: high on typed tokens, medium on which labels to rename.
- Would change my mind: a tool in the meeting plan that cannot carry a prefix in the dependency column.

### Q8.4: How are cycles resolved, and what is the shape-and-type item?

- Position: D for L3 and friends; E for T18/T20.
- Argument:
  1. The cycles are edge-direction errors: the front end owns the rules (R62), so O6, O7, and B23 must depend on L3 and L12, never the reverse (F-c).
  2. Today L3 depends on O6 and O7, and B23 depends on B7 (the e-graph), so a spec text item waits on the backend; the chain B29 -> B31 -> B7 -> B23 -> O7 -> L3 shows it (load.md chain).
  3. Option A (one item, weight 0 for six absorbed rows) hides 31 unassisted weeks by my sum of the table sizes (L3 6, L12 6, S4 6, F1 6, O7 3, B23 3, O6 1) inside one L of 6; an executor cannot plan against that.
  4. Option B (single item named S4, size M) repeats the downsizing the audit calls unsupported; option C splits only two ways and still needs sub-exits.
  5. Option D, with real sizes: (a) rule text per operation (L3, depends on L2); (b) the checker and module move into taidl (L12, O6, S4, F1; exit: shapes.py lives in taidl and the S8.3 table is generated); (c) generator deletions of duplicate checks (O7, B23, then D5).
  6. After this, L3 depends on L2 and S3 only; the chain I computed is 40 weeks (F-f), not 59.
  7. T18/T20: T20 is a decision that "binds T4 and T18" (tiers.md:62), so T4 and T18 depend on T20 and T20 has no dependency; schedule it with T22 in wave 1 (E). An owner decision (G) is not needed: it is test policy, inside engineering authority.
- Evidence: roadmap/q2-core.md:9, :12, :37, :45, :46, :71, :73; reviews/rulings/audit.md row R62; reviews/shared-diagnostics/review.md SD1, SD2; work/roadmap_extra/README.md:23; my DFS (F-c, F-f).
- Confidence: medium-high.
- Would change my mind: a showing that L3's rule text really needs backend inference results to be written (not just checked).

### Q8.5: How do the generators pin taidl before the v3 tag exists?

- Position: A plus D. Pre-release tags at each milestone; the A16 checklist is a named wave 2 item, not merged into S8.
- Argument:
  1. Wave 4 items (T16, T19) consume a pinned release, and S8 (the final tag) is wave 5, so some pin must exist earlier (q2-core T16 row, q2-supporting S8 row).
  2. Option B (pin a commit) works but leaves every generator CI exposed to unannounced contract churn after S2 (README.md:29); a tagged pre-release makes each bump one reviewed change.
  3. Pre-releases keep owner decision 10 intact: only one release is a v3 release, the others are numbered pre-releases (PEP 440 style, for example 3.0.0a1), and the coordinated generator releases still happen once.
  4. StableHLO's practice supports small, frequent, numbered releases with compatibility suites (openxla.org/stablehlo/compatibility); LLVM lands on main first and cuts from there (llvm.org/docs/DeveloperPolicy.html).
  5. Option C is the first pre-release, cut after S2 and S3, and is a subset of A.
  6. A16 has no dependencies (roadmap/q1-core.md) and A18 depends on A16 (roadmap/q2-core.md:84); merging it into S8 (tiers.md:50) makes the checklist untestable until the last day. Un-merging it lets pre-release cutting rehearse the final train.
- Evidence: roadmap/q1-core.md:86 (A16 row); roadmap/q2-core.md:84; roadmap/q2-supporting.md:36-37; work/roadmap_extra/README.md:29; reviews/shared-diagnostics/review.md SD19.
- Confidence: medium-high.
- Would change my mind: a packaging fact showing generators cannot install a pre-release pin from the taidl distribution.

### Q8.6: Which cross-track merges and drops stand?

- Position: B for every merge, plus D for the three that rest on questioned rulings (R5, R20, R29) and for the drop of O10.
- Argument:
  1. A weight-0 merged row has no exit test, so a merged outcome can silently vanish; a named sub-item with one exit criterion keeps it checkable at near-zero cost.
  2. O10 is dropped because R5 removes `set_inplace`; owner decision 17 says the output reuses the marked input's storage, and the audit flags that R5 reverses semantics/rulings.md without saying so and breaks act-backend/tests/test_extraction.py:507.
  3. Dropping O10 before the semantics and oracle owners rule leaves the decision 17 behavior (aliasing, then reuse) with no item that implements it in the oracle; keep O10 or fold it into L5 as a sub-item until the panel rules.
  4. R20 (P6 into B10) and R29 (F9 into O13) are marked questionable for cost ABI and interpreter build route; fold the outcome now but keep a sub-item for the unresolved part (rank through the cost ABI; pin the interpreter to the R6 commit).
  5. O9 (R60 removes `triangular_solve`) is a straightforward removal ruling and stands, listed with its removal as an item-less note.
  6. Option C alone (merge only where plan texts match) needs a per-pair reading I have not done for all nine pairs; the sub-item form is safe without it.
- Evidence: work/roadmap_extra/tiers.md:43-53; reviews/rulings/audit.md rows R5, R20, R29; minutes/rulings.md R5, R20, R29, R60; minutes/owner-decisions.md decision 17; reviews/shared-diagnostics/review.md SD18.
- Confidence: medium.
- Would change my mind: a read of the L5 and O10 plan texts showing O10 is fully covered by L5's exit.

### Q8.7: How is the set of milestones structured?

- Position: A with D. Keep five outcome milestones, correct each item list to its exit, and make the v3 tag the last event of M5.
- Argument:
  1. The five already read as outcomes (spec, one front end, generators follow, tests prove, usable); option B's four user-journey milestones cut across tracks and make each exit depend on all four lanes at once, which blocks the engineers on each other.
  2. Option C (two milestones) removes the only mid-course checkpoints and so the measure of the assist factor.
  3. Exit-to-item mismatches are item-list fixes: M1's "all case-study sketches parse" names only C1-C4, C9-C11, C13, so word the M1 exit that way and carry C5-C8 and the Ascend study in M2's gap-matrix exit.
  4. B21 and B22 can stay in M1 because in the regenerated table S3 depends on them, not the reverse (roadmap/q2-supporting.md S3 row).
  5. M5's exit "on the v3 tag" conflicts with S9 and A18 following the tag; A18 depends only on A16, so build it before the tag, and list S9 as the follow-on (R40: not on the v3 critical path).
  6. The follow-on list is scheduled work, not a cut.
- Evidence: work/roadmap_extra/milestones.md:7-63, :13, :15, :50, :52; roadmap/q1-supporting.md and q2-supporting.md; roadmap/q2-supporting.md S3 and S9 rows; minutes/rulings.md R40.
- Confidence: medium-high.
- Would change my mind: owner wants user-journey milestones for outside communication (then B is a second naming layer on the same five, not a replacement).

### Q8.8: What makes an exit criterion checkable, and what proves "usable by anyone"?

- Position: C. Mechanical criteria map to a named CI job; the "anyone" criteria add a cold-start trial by people outside the engineering team; D's negative criteria are added where a file or code path must disappear.
- Argument:
  1. A criterion I cannot run is one I cannot know I have finished, so every mechanical criterion needs a job name and a pass condition (option A).
  2. "Usable by anyone" is a claim about people; CI can show the install and the tutorial run, but not that a newcomer succeeds, so a person must try it (option C).
  3. The trial: two people who are not on the engineering team (default), a fresh environment, only the pinned release, README and tutorial, write and run one small ISA description; pass means success within a stated time with no maintainer help.
  4. Negative criteria are cheap to check and match the simplification goal: no `parser.cc` candidate parser, no line splitter, no `docs/operations.md`, no synonym or non-ASCII hit (milestones.md:25-27).
  5. Prose criteria such as "no silent fallback remains" become a grep or registry test (D9 and D10) or are rewritten.
  6. R63 says a tutorial run is not evidence until A10 migrates the tutorials, so the trial uses the migrated tutorial only.
- Evidence: work/roadmap_extra/milestones.md:5, :11-15, :21-27, :43-46, :54-63; minutes/rulings.md R63; minutes/owner-decisions.md decision 6; work/roadmap_extra/README.md:5-6.
- Confidence: high on the hybrid, low on the trial's numbers (people, time limit are the owner's call).
- Would change my mind: no outside people are available, in which case a recorded fresh-clone walkthrough by a non-author teammate is the weaker fallback, stated as such.

### Q8.9: How do waves relate to capacity, time labels, and the critical path?

- Position: C then D, with A's discipline. Recompute the chain after Q8.1-Q8.4, publish it with weights, measure the assist factor at M1, and rebaseline then; no dates before M1.
- Argument:
  1. Capacity, not the chain, is the binding constraint: the corrected chain is 36-40 unassisted weeks (F-f, F-g) and needs an assist factor of only 1.4-1.5 to fit 26 weeks, while the total of 443 needs 4.3-5.7.
  2. The published chain of 59 (load.md) is not a path length because it contains cycle edges and a misread ID (F-e), so planning against it misallocates attention.
  3. Four lanes keep 3-4 engineers unblocked: language/semantics/shared (spec and front end), backend, oracle, and tests/act/diagnostics/case studies; track weights are in load.md.
  4. Option B (quarter labels) turns an unmeasured factor into a calendar promise; option A alone gives no way to notice slippage.
  5. Wave 1 weights (66 in tiers.md) are the measurement: engineers record actual assisted weeks per item and publish the ratio at M1.
  6. The wave sums in tiers.md:55 do not reproduce from the regenerated tables (the 175 draft IDs sum to about 414 in my parse against 440 stated), so weights must be regenerated, not trusted.
- Evidence: roadmap/load.md; work/roadmap_extra/tiers.md:55; work/roadmap_extra/README.md:19-24; reviews/rulings/audit.md row RA12; my sums (F-e to F-g).
- Confidence: medium (the 4.3-5.7 range depends on my sum and the 26-week assumption).
- Would change my mind: any measured assist factor from earlier Claude-assisted work on these repos.

### Q8.10: Which gates and long-lead inputs are scheduled up front?

- Position: B, with C as the rule: a wave 1 gate row, each gate placed at least one wave before its dependents, including A10 and the Ascend case study.
- Argument:
  1. Several gates are cheap to start and costly to discover late: owner approval of `hypothesis` (R45) and the StableHLO interpreter (R29) block T9 and O13 in wave 3, and approvals take owner time, not engineer time.
  2. A10 has no dependencies (roadmap/q1-core.md) and A11, A13, A21, A22 all follow it; running it in wave 1 turns R63's unknown into a list of failing syntax before wave 5.
  3. The Ascend case study is the only R15 item; R59 (two case studies per extension) is applied in wave 3, so the study must exist by wave 2, as its own item separate from the C12 re-run.
  4. The R28 spike (O19) must come first in the oracle lane because O20 and O1 follow it, and the audit says its bound is undefined; the bound (compile time on the attention tests and examples) is the first gate outcome.
  5. Because O19 waits on O8 -> L16 -> L3 in the table, the spike is gated by the front-end hinge; ask the oracle owner whether that edge is real, since dropping it shortens the chain to 36 weeks (F-g).
  6. Each gate row has a named outcome: approved or refused (R45, R29), bound set and spike result (R28), failing-syntax list (A10), study sketch parses (Ascend).
- Evidence: work/roadmap_extra/tiers.md:41; work/roadmap_extra/README.md:31-35; roadmap/q2-core.md A10-A22 rows; oracle/plan/architecture.md:9; reviews/rulings/audit.md rows R15, R28, R29; minutes/rulings.md R15, R28, R29, R45, R59, R63.
- Confidence: high on A10 and approvals, medium on the O19 edge.
- Would change my mind: a reason O19's measurement is invalid before float literals are rounded once (O8).

### Q8.11: What does the risk register contain, and how is each risk noticed?

- Position: B with C's link. Extend the list, give each risk a tripwire phrased as an observable, and name the milestone exit it endangers; D's three categories are a tag.
- Argument:
  1. A risk with no observable is noticed after the damage; the draft has tripwires but no link to the exit it threatens, which tells the engineer nothing about priority.
  2. Added risks: v3 migration (RA11; tripwire: a rejected construct has no replacement text in the registry test D10); R56 against i64 checking (tripwire: generated Rust arithmetic needs i128 or bignum, or L6 tests disagree between oracle and backend); ANTLR exposure (SD4; tripwire: a sample third-party generator imports `antlr4`); compression of waves 3-5 (tripwire: M2 closes with wave 2 items unfinished); assist factor (tripwire: the M1 ratio below the 4.3 needed); dependency-data drift (tripwire: the Q8.1 checker fails).
  3. The R18 order risk is deleted from the register once the data are fixed, because the tables agree (F-b); keeping it twice (risk and correction) is noise.
  4. Each risk gets a fallback outcome so a tripped wire has an action: for example, R59 defers an extension; R29 keeps O13 as the reference; a failed R28 bound keeps O11 and the loop algebra.
  5. Option C's rule to drop risks that no criterion covers is too strict: ANTLR exposure and migration threaten "anyone can use" without a current criterion, so add the criterion, not remove the risk.
- Evidence: work/roadmap_extra/README.md:26-35; reviews/rulings/audit.md rows R56, R62, R28, R29, RA11, RA12; reviews/shared-diagnostics/review.md SD3, SD4, SD6; work/roadmap_extra/milestones.md.
- Confidence: medium-high.
- Would change my mind: a program rule that risk entries carry no response column (then fallbacks live with the milestone).
