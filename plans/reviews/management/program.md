# Management review: program

Lens: program manager; timeline, critical path, blockers, priority, and value and impact per milestone.
Read: plans/README.md, roadmap/, minutes/owner-decisions.md, panel decisions 8, 10, 11, panel 12 round 3, reviews/approval.md, and the track plan indexes.
Chain lengths below come from a script over the four wave files, with S = 1, M = 3, L = 6 (roadmap/waves.md:15).

## Verdict

Approved with conditions.

- The wave structure, outcome milestones, gate row, and risk register are sound, and the release order follows owner decisions 10, 26, and 33.
- The dependency graph does not yet encode the release order, so the published critical path (35) understates the real path to M5 (50).
- Three blockers sit at M1 with no item or a contradiction: L46, the Ascend gate, and panel 12's constraints on wave 1 items.
- Usability for architects and retargetability have no exit criterion between M1 and M5, so they are measured only after v3 freezes the surface.

## Value and impact per milestone

| Milestone | Weight | Cumulative share of M1 to M5 | Value delivered | Who sees it |
| --------- | ------ | ---------------------------- | --------------- | ----------- |
| M1 | 164 | 32% | Normative rules, typed contract, deterministic gate, width layer | Team only |
| M2 | 128 | 57% | One `check`, traced surface, allocator relations (fixes backend D14, D16), unrolled oracle (fixes oracle D11) | Team only |
| M3 | 130 | 83% | Generators on the `Model`, failure policy, exclusion report; last 2.x release with the converter | First public deliverable |
| M4 | 74 | 97% | v3 on released pins, differential tests, encoding unit | All users |
| M5 | 14 | 100% | One tutorial, cold-start trial | New users |

- 83% of the weight completes before v3 reaches any user (roadmap/milestones.md:10-16; roadmap/waves.md:23-29).
- M2 carries the most correctness value per week: four high-severity wrong-result bugs close there (backend/plan/bugs.md:21,23,25; oracle/plan/bugs.md:22).

## Critical path

- As written, the longest chain is weight 35 and ends at T34: B29 -> B1 -> B31 -> B7 -> B26 -> D7 -> D16 -> T32 -> T33 -> T34 (roadmap/load.md:20-33).
- U8 has edge depth 24 and A23 only 11, because neither row depends on the encoding unit, L45, T37, or each other (roadmap/wave-4-7.md:97,99).
- With the order that waves.md:54-56 states, the chain to M5 is weight 50: the chain above -> U8 -> A23 -> A24 -> A21.
- The first six links are backend items, two of them L (B7 at roadmap/wave-2.md:23, D7 at roadmap/wave-3.md:62).
- The serial chain is not the binding limit; capacity is (PM5).

## Findings

| ID | Area | Finding | Evidence | Recommendation | Severity |
| -- | ---- | ------- | -------- | -------------- | -------- |
| PM1 | Critical path | The published chain stops at T34 and omits the release train and tutorial, so the M5 path (50) is invisible to the checker and the rebaseline. | roadmap/load.md:20-33; roadmap/wave-4-7.md:97,99,107,111; roadmap/waves.md:54-56 | Add the release edges (PM2), recompute, and publish the M5 chain in load.md. | high |
| PM2 | Dependency graph | Release-order rules live only in row order and prose, which the checker cannot test: U8 has no A16 edge though K10 says U8 waits for it, and L45 and U14 have no T37 edge though K24 says they wait. | roadmap/wave-4-7.md:89-90,97; roadmap/risks.md:27,39; roadmap/waves.md:90 | Add edges U8 <- A16, L45, T37, B45, T34; A23 <- U8; L45 and U14 <- T37. | high |
| PM3 | Mis-ordered | A28 ships the converter in wave 4 with only an A16 edge, yet its own text gates each converted file with T37, which lands in wave 5; it has no edge to L44 or U13. | roadmap/wave-4-7.md:60-62,88; act/plan/versions-release.md:80,86 | Add A28 <- L44, U13; run T37's check on act's files inside A28, or move A28 after T37. | high |
| PM4 | Contradiction | A23 says "U8 and the tutorial (A24) follow it", while the roadmap and U8 release taidl 3.0 first and act 3.0 last. | act/plan/versions-release.md:73; roadmap/waves.md:54-56; shared/plan/U8.md:5 | Change A23's order line to "after U8"; A24 follows. | medium |
| PM5 | Capacity | K1 trips at an assist factor of 4.3, derived from 443 weeks; the scheduled weight is now 541, or 510 through M5, which needs 4.9 to 6.5 against 78 to 104 engineer-weeks. | roadmap/risks.md:18; meeting/panel-8/decision.md:147-148; roadmap/waves.md:30; minutes/owner-decisions.md:11 | Restate K1 against the M5 weight (4.9 at 104 weeks) and recompute it at each table regeneration. | high |
| PM6 | Measurement | Wave 1 holds 39 S and 14 M items and no L item, while 16 L items (weight 96) through M5 carry the riskiest work; the M1 dates come from that sample. | roadmap/wave-1.md:9-61; roadmap/waves.md:16 | Start one L item in wave 1 for measurement (L3 needs only L2, so row order allows it), and record the L factor separately at M1. | medium |
| PM7 | Missing item | L46, the declared initial value, is in the language plan but in no roadmap table or wave; O26 and B36 do not depend on it; approval condition 1 must close before M1. | language/plan/norms.md:10,12; roadmap/wave-4-7.md:26,36; oracle/plan/checks.md:15; reviews/approval.md:39-42 | Regenerate the tables with L46 in wave 2 (after L32, L16) and edges to O26 and B36; K2 has tripped. | high |
| PM8 | Wave 1 gate | The Ascend gate has no item, C12 says no Ascend study is made, and R15 says it is added at the start of C12; the M1 criterion "every wave 1 gate has its named outcome" cannot close. | roadmap/waves.md:45; casestudies/plan/c12_rerun_on_v3.md:4; minutes/rulings.md:25; roadmap/milestones.md:41; roadmap/risks.md:31 | The case-studies lead adds the item or a ruling closes the gate row as not made; one text wins in all three files. | medium |
| PM9 | Long-term platform | Panel 12 round 3 converges on waves 8 to 10 after M5 (weight about 58 to 65) and on constraints to wave 1 to 3 items: B29 runs the no-ban schedule with canonical candidate order, B6 writes rules as language-neutral records, U2 keeps the `Model` plain data, A16 records the pin set. B29 heads the critical path and its text holds none of these. | meeting/panel-12/round3-fable.md:68-73,87; meeting/panel-12/round3-opus.md:83-85,124; backend/plan/search.md:13-23; reviews/approval.md:49-51 | Write the constraints all three debaters share into B29, B6, B31, B2, U2, and A16 before B29 starts; panel 12's decision then adds waves 8 to 10. | high |
| PM10 | Usability | The only outside-user test is the M5 cold-start trial, after taidl 3.0 freezes the traced surface; a breaking fix after that waits for the next batched major. | roadmap/milestones.md:106; roadmap/wave-3.md:50-51; minutes/owner-decisions.md:10,35 | Add an M3 criterion: an architect outside the team writes one ISA specification on the traced surface in the A20 image, and findings reach L30 and D15 before U8. | high |
| PM11 | Retargetability | No criterion after M1 measures accelerator coverage; C12 calls the gap matrix a release criterion, but M4 lists none, and C12 sits in wave 3 without edges to L17 or the v3 surface that its caveat requires. | roadmap/milestones.md:40,82-100; casestudies/plan/c12_rerun_on_v3.md:5-8; roadmap/wave-3.md:43 | Add C12 <- L17, L42, L43, and an M4 criterion that the C12 gap matrix passes on the release candidate. | medium |
| PM12 | Value timing | Fixes for high-severity wrong results land in waves 1 to 3 (B38, B30, B41, O8, O20) but reach users only at M4; owner decision 10 batches breaking changes only. | backend/plan/bugs.md:21-25; oracle/plan/bugs.md:12,22; minutes/owner-decisions.md:10 | Ask the owner whether non-breaking fixes such as B41's drop-not-exit (backend D17) ship in a 2.x patch through the A16 order. | medium |
| PM13 | Ordering | The roadmap holds the encoding unit (weight 24, two L) behind the string-surface deletion as the last work before U8; Q10.4 orders only layer 1 first and the unit whole, and the Q11.4 order never names the unit. | roadmap/waves.md:13; roadmap/wave-4-7.md:67-68; meeting/panel-10/decision.md:103-104; meeting/panel-11/decision.md:128-134 | Develop L38, L39, and B44 from wave 4 on the private heads; keep "lands whole before U8" as the gate, with no scope change. | medium |
| PM14 | Resource | The M5 chain head is all backend (B29 to D7, two L); the backend lead also owns K3, K4, K13, K21, and K25, and backend weight is 27 in wave 3 and 36 in wave 4. | roadmap/wave-1.md:24; roadmap/wave-2.md:23; roadmap/wave-3.md:62; roadmap/risks.md:20-21,29,37,40 | Staff B7 and D7 with two engineers at the M1 rebaseline; make B7 open at wave 2 end a K8-style tripwire. | medium |
| PM15 | Blocker | Approval condition 6 (finite accuracy of `exp`, `log`, `tanh`) blocks T13, but no item owns it; L31 covers only `dot` and two reductions; T13 is an L item that feeds T18, T31, and T36 at M4. | reviews/approval.md:52-54,76; roadmap/wave-2.md:14; roadmap/wave-4-7.md:74-83 | Widen L31 or add a language item before T13, with an edge into T13. | medium |
| PM16 | Graph hygiene | D8 depends on O11 in the roadmap, a fallback that lands only if O19 fails; the diagnostics plan says "O20 or O11". | roadmap/wave-3.md:63; diagnostics/plan/README.md:7,16; reviews/approval.md:67 | Regenerate the row with the edge that O19 resolves. | low |
| PM17 | Load | Wave 5 holds 46 of its 74 weight in the test track, including T13 and T33 (both L), just before the release; K8 watches waves 3 and 4 only. | roadmap/wave-4-7.md:73-96; roadmap/risks.md:25 | Extend K8's tripwire to wave 5 and the test track. | low |

## Top five actions

1. Before B29 starts, write panel 12's shared constraints into B29, B6, B31, B2, U2, and A16 (PM9).
2. Regenerate the tables: add L46 and its edges, the release edges, A28's edges, and the resolved D8 edge; publish the M5 chain (PM1, PM2, PM3, PM7, PM16).
3. Restate K1 against the M5 weight, and measure one L item in wave 1 (PM5, PM6).
4. Add an outside-architect trial of the traced surface before U8, and a C12 gap-matrix criterion at M4 (PM10, PM11).
5. Close the open M1 inputs: the Ascend gate, an owner for the accuracy criterion, the A23 order line, and the owner's answer on a 2.x patch (PM8, PM15, PM4, PM12).
