# Panel 8, round 1, Fable: risk red team

Lens: what goes wrong, hidden dependencies, optimistic exit criteria, single points of failure.
Assumption throughout: the regenerated quarter tables (190 items) are the current data, and the draft waves (175 items) are stale against them.
Paths are relative to /home/user/act/plans unless they start with work/.

### Q8.1

Position: B, regenerate the waves from the 190-item tables, with a mechanical check that every table ID appears exactly once in tiers.md.

1. The tables are already regenerated: load.md:18 sums 88 Q1 items, 92 Q2 items, and 10 later items, and the track items B29 to B34, O19 to O24, and L29 to L31 are present (roadmap/q1-core.md:22-24, :48-50; roadmap/q2-core.md:30-32, :49-53).
2. The waves still say 175 (work/roadmap_extra/tiers.md:3, :55), so the data and the plan already disagree, which is exactly the state option A institutionalizes.
3. The new items carry edges that change the order: B1 now depends on B29 and B7 on B31 (roadmap/q1-core.md:30, :35); O1 depends on O20 and O19 moved to Q2 (roadmap/dependencies.md:15, :36).
4. Option D adds items without their edges, so the reading-order invariant (tiers.md:9) silently breaks; option C leaves no per-item size or quarter view to check wave sums against.
5. A second divergence: other-core.md:9-18 marks L9, L18 to L23, L27, L28 as later, while tiers.md:27, :33 schedule L27, L28 in wave 2 and L18 to L23 in wave 3; the rebuild must state which view wins per item.

Evidence: load.md:18; tiers.md:3, :55; q1-core.md:30, :35; dependencies.md:15, :36; other-core.md:9-18; tiers.md:27, :33.
Confidence: high.
Would change my mind: a tiers.md check line that reproduces from the 190-item tables, which would make option A a solved problem.

### Q8.2

Position: B, the R18 order B29, B31, B7, B9, B6, with B29 landed first as the gate; not D, because B29 is the gate and it is already first.

1. The regenerated table already encodes R18: B6 depends on B7 and B9, B7 on B31, B9 on B7 (roadmap/q1-core.md:34-37); the conflict the agenda names survives only in tiers.md:29, which lists B6 before B7.
2. The backend plan gives the same order with the reason that B6 first would regenerate `set_shapes_N` that B7 deletes (backend/plan/decisions.md:11; backend/plan/isel.md:34).
3. B29 is "first item of the plan" and the only thing that makes the B3 gate comparable across machines (backend/plan/search.md:9, :20-21; decisions.md:8); without it every e-graph item's verification line, "B3 gate identical", is unreliable.
4. Option D (hold the group) costs nothing beyond option B, since B29 is M and precedes B31 in reading order anyway; holding past B29 only delays wave 2.
5. Hidden dependency: B7 "needs each operation's shape rule in Rust, sharing a table with the front end where one exists" (isel.md:32), while L3 is Q2 (q2-core.md:9); B7 in Q1 writes a second copy of the shape rules that R62 forbids, and no table edge records it.

Evidence: q1-core.md:34-37; tiers.md:29; decisions.md:8, :11; isel.md:32, :34; search.md:9, :20-21; q2-core.md:9.
Confidence: high on the order, medium on the B7 against L3 risk.
Would change my mind: a B7 design that reads the shape rules from taidl at generation time, which would remove the duplicate copy.

### Q8.3

Position: B for the S collision (rename the shared items to a free letter), plus D for the backend defect labels and the bug prefix; spec section ids stay.

1. The collision already corrupted the data: T1 depends on an undefined S10 (roadmap/dependencies.md:47), and T5, T7, T8, T10, T11, T19 and their dependents moved to Q2 on edges to spec sections S9 and S8 (dependencies.md:18-28).
2. The generator cannot tell "S8" the item from "S8" the section, so option A (discipline in prose) fails at the next regeneration, and option C needs every track plan to carry typed prefixes the generator does not parse today.
3. Spec section ids are the public contract: L1 ships them, every track cites them, and renaming one "breaks citations" (language/plan/spec.md:27-29); R44 puts them in error messages (minutes/rulings.md:82).
4. Nine shared items (q2-supporting.md:29-37) are the cheaper side to rename than ten spec sections and every S9 row.
5. The "(D1)" to "(D5)" labels on B21 to B25 (q1-core.md:41-44; q2-core.md:29) read as diagnostics items D1 to D5, and X1 to X10 collide with class X (reviews/shared-diagnostics/review.md:37); renaming labels is cheaper than renaming the diagnostics track.

Evidence: dependencies.md:18-28, :47; spec.md:27-29; rulings.md:82; q2-supporting.md:29-37; q1-core.md:41-44; review.md:37.
Confidence: medium.
Would change my mind: a table generator that resolves a dependency only against item IDs and rejects unknown ones, which would make option A safe.

### Q8.4

Position: C, two items sized L or larger, with each absorbed item kept as a named exit test (the checkable part of D); for T18 and T20, E with T20 recorded as the panel 6 outcome before wave 4.

1. The cycle is an artifact of six owners for one piece of work (review.md:18), so merging removes it; but one item at weight 6 for work the tracks sized at 34 is a hidden cut on the hinge of the critical path (work/roadmap_extra/README.md:23; reviews/rulings/audit.md:35).
2. The module move (oracle `shapes.py` into taidl, S8 table derived) and the front-end checks (L12, O7 widened by R31, S4) have different exit tests and different consumers, so two items give an early tripwire: the move can land before the checks.
3. B23 is not the same work: SD2 redefines B21 to B23 as calls to `validate()` plus deletion (review.md:19), so absorbing B23 into L3 hides a backend edit.
4. F1 is a semantics item sized L (q1-supporting.md:45) and absorbing it at weight 0 double-counts nothing but also leaves no exit test for the static checker in the front end.
5. T20 is a decision that "binds T4 and T18" (tests/decisions/rulings.md:22), so it belongs before both; a decision item on the longest chain (load.md:45) is schedule noise once panel 6 settles it.

Evidence: review.md:18-19; README.md:23; audit.md:35; q1-supporting.md:45; tests/decisions/rulings.md:22; load.md:45; q2-core.md:74-76.
Confidence: medium.
Would change my mind: evidence that the oracle's `shapes.py` already implements every rule L3 states, which would make the merged item M as R62 claims.

### Q8.5

Position: A plus D, pre-release tags of taidl at each milestone that generators pin, with the A16 release order and checklist written in wave 2.

1. Owner decision 10 is one major release; a pre-release tag is not a release, and the LLVM process (llvm.org, HowToReleaseLLVM) cuts signed rc tags before the final to make the testing window reviewable.
2. Pinning a commit (option B) makes each bump an unnamed diff; the contract churn risk (README.md:29) is noticed only when generator CI breaks, with no name for what changed.
3. The Rust toolchain practice is to pin an exact version so toolchain upgrades become deliberate, reviewable pull requests (rust-lang book, appendix on nightly Rust, and project `rust-toolchain.toml` rationales); a milestone tag gives the same property here.
4. Writing the release order last (A16 merged into S8 in wave 5, tiers.md:41, :50) is a single point of failure: A18 depends on A16 (q2-core.md:87), and the pin order between taidl, both generators, and act is exercised at every bump, not once.
5. The M5 install check (milestones.md:56) runs once at the end; a pre-release tag per milestone makes it run four times before.

Evidence: owner-decisions.md:10; README.md:29; tiers.md:41-42, :50; q2-core.md:87; milestones.md:56; llvm.org HowToReleaseLLVM; rust-lang.github.io book appendix G.
Confidence: high.
Would change my mind: a generator CI that builds against the taidl main branch nightly and reports the first failing commit, which would make commit pins observable.

### Q8.6

Position: B for every merge, and D for the merges and drops that rest on R5, R20, R29.

1. Merged items hide 50 unassisted weeks at weight 0 (tiers.md:55); under "nothing is cut for capacity" a merge that loses an outcome is an unstated cut.
2. O10 is dropped by R5, yet owner decision 17 still requires the output to reuse the marked input's storage, and the audit finds R5 breaks a fixture and reverses the semantics track (audit.md:19); the oracle work of O10 survives even if `set_inplace` goes.
3. P6 into B10 rests on R20, which the audit marks questionable because the cost ABI is pluggable (audit.md:26); the cost-aware part of P6 is a distinct outcome.
4. F9 into O13 rests on R29, whose dependency waits on owner approval (rulings.md:58); if approval fails, O13 falls back to NumPy and F9's "exact classes" outcome is lost unless named.
5. S7 into T7 loses the mini_trn merge and generator CI parts SD18 keeps (review.md:35); O9 by R60 is a clean drop because the work is removal.

Evidence: tiers.md:43-53, :55; owner-decisions.md:17; audit.md:19, :26; rulings.md:58; review.md:35.
Confidence: medium-high.
Would change my mind: panel decisions upholding R5, R20 and R29 as written, after which option A's drops become rulings that remove work.

### Q8.7

Position: A with corrected item lists, plus one standing criterion from M2 onward: "install from the milestone pre-release tag and run the README snippet".

1. Five outcome milestones are already defined by outcomes (milestones.md:7, :17, :29, :39, :48), so option B's restructure buys little beyond relabeling.
2. The item-list mismatches are data errors to fix, not a structure problem: C5 to C8 under M2 while M1 exits on all sketches parsing (milestones.md:15, :20); B21, B22 under M1 while SD2 makes them M2 calls (review.md:19); S9, A18 after the tag under M5 (milestones.md:50-52).
3. The structural risk is that the only user-facing outcome arrives in the last wave of weight 25 after a wave of 127 (tiers.md:55); every install or tutorial defect is found at the end.
4. Option C hides the tripwires further: two milestones mean one check per quarter.
5. Option D is a subset of A once the post-tag items are listed as follow-on; it is folded in.

Evidence: milestones.md:7-63; review.md:19; tiers.md:55.
Confidence: medium.
Would change my mind: a reading of A15, A20, A5 (wave 1 and 2) showing the install path is exercised in CI from wave 1, which would remove the need for the standing criterion.

### Q8.8

Position: C plus D, tests for the mechanical criteria, a cold-start trial by an outside person for the "anyone" criteria, and a negative criterion per milestone.

1. "A fresh environment installs v3" (milestones.md:56) is a CI job, not a person; "usable by anyone" is a claim about a reader, and only a reader can falsify it.
2. R63 forbids any fail point relying on a tutorial run before A10 (rulings.md:35), so the M5 tutorial criterion is unverifiable until A10 lands; A10 belongs earlier (see Q8.10).
3. Prose criteria such as "no silent fallback remains" (milestones.md:33) have no observable; the track plans already give the observable in their verification lines, for example no `set_shapes_` in the crate (isel.md:35), grep empty for old names (frontend.md:83, :105).
4. Negative criteria are the cheapest tests of deletion items and are where regressions hide: a kept C++ parser or a kept line splitter passes every positive test.
5. Option B (reviewer sign-off) is the status quo that produced M1 exits whose items sit in M2.

Evidence: milestones.md:33, :54-61; rulings.md:35; isel.md:35; frontend.md:83, :105.
Confidence: high.
Would change my mind: a documented outside-user trial already scheduled under C12 or A21, which would make C the drafted state.

### Q8.9

Position: A now (order only, no quarter labels), C once Q8.1 to Q8.4 resolve, and D at M1 to set any later label.

1. Decision 11 gives 78 to 104 unassisted engineer-weeks; the waves need 388 (tiers.md:55) and the regenerated tables 443 (load.md:18 plus other-core.md), so the labels assume an assist factor of 3.7 to 5.7 that no document states or measures.
2. The longest chain (load.md:22-45) passes through the B23, O7, L3 cycle and the T18, T20 cycle, so its weight 59 is not a path length and the "indicative timing" lines (milestones.md:9, :19, :31, :41, :50) rest on nothing.
3. Wave 3 holds 127 weeks with one row of 36 (tiers.md:31), so the labels compress the hardest Rust and C++ work into one half-quarter.
4. A quarter label on a milestone defined by outcomes invites closing on the date, which milestones.md:4 forbids; the label is pure downside until measured.
5. The audit already asked for one sized table before rulings bind (audit.md:52); the honest form is weights beside milestones, dates after wave 1 measures the factor.

Evidence: owner-decisions.md:11; tiers.md:31, :55; load.md:18, :22-45; milestones.md:4, :9; audit.md:52.
Confidence: high on the arithmetic, medium on the remedy.
Would change my mind: a measured assist factor from comparable Rust and C++ work by these engineers.

### Q8.10

Position: B, one gate row in wave 1 with a named outcome per gate, and A10 and the Ascend study moved into it.

1. The R28 spike needs its bound fixed before it runs (oracle/plan/architecture.md:23; audit.md:30), and the owner must set it; that is a wave 1 input with no item.
2. The spike is chained behind L3: O19 depends on O8, O8 on L16, L16 on L3 (q2-core.md:49, :43, :13), so O19, O20, O1, O2, O3, O13, S1 and T5 all wait on the Q2 shape merge (load.md:30-39); the O8 edge exists only to fix the baseline (architecture.md:24) and should be dropped so the spike runs in wave 1.
3. R29 and R45 need owner approval first (rulings.md:58, :83); unanswered, O13 and T9 block at wave 3 start (README.md:32), and nothing before then asks.
4. R59 is applied in wave 3 (tiers.md:33) before the Ascend study, which sits at the start of C12 in wave 5 (tiers.md:41; rulings.md:25); the audit finds R15's reason argues against itself (audit.md:25), so the study either moves to wave 1 or is dropped from the R59 count by decision.
5. A10 has no dependency (q1-core.md:84) and R63 blocks every tutorial fail point until it lands (rulings.md:35), so it is the earliest "usable by anyone" tripwire and sits in the last wave for no reason.

Evidence: architecture.md:23-24; audit.md:25, :30; q2-core.md:13, :43, :49; load.md:30-39; rulings.md:25, :35, :58, :83; README.md:32; tiers.md:33, :41; q1-core.md:84.
Confidence: high.
Would change my mind: owner answers to R28, R29, R45 and R15 recorded before wave 1, which would make option A equivalent.

### Q8.11

Position: B plus C, extend the list, phrase each tripwire as an observable, tie each to the criterion it endangers, and add a criterion where none covers a risk instead of dropping the risk.

1. Eight risks without owner, response or milestone (README.md:26-35) are notes, not a register; a risk no criterion covers shows a missing criterion, so option C's drop rule removes the signal.
2. Candidates from the reviews: v3 migration with no converter (audit.md:51); R56 against i128 in generated Rust (audit.md:33); ANTLR trees in the contract tie outside generators to 4.9.2 (review.md:21); a jax bump becomes a spec change (spec.md:39-40).
3. Candidates from this round: the unmeasured assist factor (Q8.9); B7 Rust shape rules against L3 (Q8.2); the O19 chain behind L3 (Q8.10); wave 3 at 127 weeks (tiers.md:55); release order written last (Q8.5).
4. The R18 conflict is now a stale-data item, not a risk (q1-core.md:34-37), and goes in the corrections list only.
5. Option D's three categories are a useful column, not a replacement for the tripwire and the criterion link.

Evidence: README.md:26-35; audit.md:33, :51; review.md:21; spec.md:39-40; tiers.md:55; q1-core.md:34-37.
Confidence: high.
Would change my mind: a milestones.md where every drafted risk already maps to an exit criterion with an observable, which would reduce the work to extension only.
