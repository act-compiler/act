# Management review: summary

Synthesis of the program, product, and execution reviews in this folder.
Each kept claim was re-checked against roadmap/, the track plans, reviews/approval.md, and the panel decisions; stale and pivoting findings are listed at the end.
Paths are relative to plans/.

## Verdict

Approved with conditions: the wave structure, outcome milestones, release order, and architect loop are sound, and 83% of the weight lands before v3 reaches any user.
The risk is in what the dependency graph does not enforce: U8 and A23 descend from no M4 evidence item, so the release can be cut with the encoding unit, T37, and T13 open.
The roadmap lags the plans: L46 and panel 12's waves 8 to 10 are absent, and K1 still trips at 4.3 from the old 443-week total.
Retargetability has no exit criterion after M1, and usability is tested by outsiders only after taidl 3.0 freezes the surface.

## Milestones

| Milestone | Value | Impact | Blockers | Priority |
| --------- | ----- | ------ | -------- | -------- |
| M1 | Normative rules, typed contract (U2), width layer (L37), wave 1 gates closed, assist factor measured; weight 164, 32% cumulative | Team only; the M1 rebaseline sets every date | L46 in no wave; Ascend gate has no item; no item for private development; K1 stale; no L item measured (MA5, MA6, MA8, MA10) | P1 |
| M2 | One `check` for every generator; traced surface; B30, B41, O8, O20 fix five high-severity bugs (backend D14, D16, D17; oracle D1, D11); weight 128, 57% | Team only | Tracer counted twice (L42, U12) on the K26 path; B3 has no dependents (MA4, MA13, MA15) | P1 |
| M3 | Generators read the `Model`; failure policy; exclusion report; converter in the last 2.x release (A28); weight 130, 83% | First public deliverable | Converter contract contradicts itself and A28 lacks edges; U2 version churn trips K19 by design (MA3, MA14) | P2 |
| M4 | v3 on released pins; differential tests (T13); equivalence (T37); encoding unit; weight 74, 97% | All users | Release not gated on M4 evidence; encoding unit last on the chain; no retargetability criterion (MA1, MA2, MA7, MA11) | P1 |
| M5 | One tutorial (A24) and a cold-start trial; weight 14, 100% | New users | Tutorial and trial run only on the released 3.0; input format unstated (MA12, MA18, MA19) | P2 |
| Waves 8 to 10 | StableHLO importer, C++ e-graph library, torch entry (panel 12, 19 items, weight 65) | MLIR and PyTorch users, after M5 | Not in waves.md; no M6 and no risk rows (MA9) | P3 |

- Weights and shares: roadmap/milestones.md:10-16; bug rows: backend/plan/bugs.md:21,23,25 and oracle/plan/bugs.md:12,22, all fixed by wave 3 items (roadmap/wave-3.md:16,22-24).
- The published longest chain is weight 35 and ends at T34 (roadmap/load.md:20-33); with the release order of roadmap/waves.md:54-58 the chain to M5 adds U8, A23, A24, A21 for weight 50.

## Product needs

| Need | Delivered by (item IDs) | Gap |
| ---- | ----------------------- | --- |
| Robustness | L32, L46, O8, O20, B30, B41, D16, T13, T36 | M4 evidence is not a precondition of U8 (MA1); B3 gates no backend rewrite (MA13) |
| Reusability | U2, U3, L43, U10, L40 | Three item pairs describe one deliverable (MA4); U2 has no version policy for additive fields (MA14) |
| Retargetability | L37, L17, C11, C12, B14 | No milestone measures backend coverage per case study or cites the C12 gap matrix; C12 lacks an L17 edge; Ascend undecided (MA10, MA11) |
| Extensibility | L41, A27, U11, T35; B47, B49, B53 (panel 12) | Three items describe one extension path (MA17); panel 12 items are in no wave (MA9) |
| Usability | L42, A10, A24, A21, L30, D15 | Outsiders test only after the freeze (MA12); v3 input format unstated (MA18); quickstart opens on a bitcast idiom (MA19); oracle time per kernel unknown (MA21) |
| Maintainability | U14, L45, T15, A16, B7 with its K3 agreement test | Converter contract inconsistent (MA3); shape rules in three copies (MA16); private repo setup has no item (MA5) |

## Actions

| ID | Action | Source findings | Owner track | Severity | Status |
| -- | ------ | --------------- | ----------- | -------- | ------ |
| MA1 | Add one release-gate item depending on the encoding unit, T12, T13, T36, T37, T38, and L45; make U8 and A23 depend on it; add U8 <- A16, A23 <- U8, and L45, U14 <- T37; change A23's order line (act/plan/versions-release.md:74) to "after U8" | E1, PM2, PM4 | roadmap, act, shared | High | applied: gate item A31; L45 <- T37 carried as U14 <- T37 |
| MA2 | After MA1, recompute and publish in load.md the longest chain to each milestone, M5 included | PM1, PR4 | roadmap | High | applied |
| MA3 | Rule one converter text (language/plan/surface.md:55 says no public 2.x release carries it; act/plan/versions-release.md:80 says the last one does); add A28 <- L44, U13; place A28 after T37 or run T37's check inside A28; freeze the U12 spelling before that release | E3, PM3 | language, act, shared | High | applied: the last 2.x release carries the converter; A28 runs T37's check; A28 <- L44 carried by U13 |
| MA4 | Merge L42/U12, L44/U13, and L45/U14 into one item each, with exits as sub-items, one owner, and unioned edges; state L43 as the layer over U3; update K26 and M2 | E4 | language, shared, roadmap | High | applied: U12, U13, U14 absorb L42, L44, L45; L43 is the layer over U3 |
| MA5 | Add a wave 1 gate item under A16: repo topology, branching, private-to-public sync, private CI owner, and the 2.x maintenance policy | E2, PM12 | act | High | applied: gate item A32 |
| MA6 | Restate K1 against current weight: 4.9 to 6.5 through M5 (510), 5.2 to 6.9 for 541, 5.8 to 7.8 with waves 8 to 10 (606), at 78 to 104 engineer-weeks; recompute at each regeneration; record the factor per work class; start one L item (L3) in wave 1 | PM5, PM6, PR3, E5 | roadmap | High | applied, recomputed: 4.8 to 6.4 (503), 5.1 to 6.8 (528), 5.7 to 7.6 (594) |
| MA7 | Develop the encoding unit (L38, L39, B44, B45, T33, T34) from wave 3 on the private heads and merge it whole after the deletion pass; the gate and K20 stay unchanged | PM13, E14 | language, backend, tests | High | applied |
| MA8 | Regenerate the tables: L46 (S, wave 2, after L32 and L16) with edges to O26 and B36; D8 on "O20 or O11" with the checker taught "or"; L22 to the dropped rows; commit the Q8.1 checker under plans/ | PM7, PM16, PR10, E6 | roadmap | High | applied in the wave files and the regenerated tables; the checker is not committed (owner decision 48) |
| MA9 | Add waves 8 to 10 from the panel 12 decision, an M6 with checkable exits, and risk rows for the MLIR pin (U15), the torch-mlir pairing check (B53), and the swap gate (B52); B53 may sit in wave 8 after B47 | E7, PR13 | roadmap | Medium | applied: wave-8-10.md, M6, K28 to K30; B53 kept in wave 9 with the wave 8 option noted |
| MA10 | Decide the Ascend study once and make waves.md:45, risks.md K15, minutes/rulings.md R15, and casestudies/plan/c12_rerun_on_v3.md:4 agree: a wave 2 item, or the gate row and K15 removed | PM8, PR11 | casestudies, roadmap | Medium | applied: no Ascend study; gate row and K15 removed; R15 and C12 agree |
| MA11 | Add to M4 the C12 gap-matrix pass on the release candidate and a generated backend share of instructions per case study; add C12 <- L17; give each sketch a backend expectations row; list state updates as a later row | PM11, PR1, PR8 | casestudies, backend, roadmap | Medium | applied |
| MA12 | Before U8, draft A24 and run the Claude-driven fresh-container run on the private heads; name the two trial participants by M3; have one architect outside the team write an ISA specification on the traced surface, with findings to L30 and D15 | PM10, E11 | act, language | Medium | applied |
| MA13 | Add edges from B3 to the Rust rewrites (B31, B7, B6, B9, B10); gate wave 3 and 4 merges on a thin oracle-vs-backend run over the T14 ISA specifications | E9 | backend, tests | Medium | applied |
| MA14 | Define U2's version policy: additive fields keep the version, a breaking change bumps it; reserve the layer fields; trip K19 on breaking changes only | E10 | shared | Medium | applied |
| MA15 | Name an owner and a close date per wave 1 gate and per L item; pair a second engineer on B7 and D7; make B7 open at wave 2 end a tripwire | E12, PM14 | roadmap, backend | Medium | applied: owner roles per gate, names and dates at the M1 rebaseline, K27 |
| MA16 | Mark each panel 12 text change on the B chain "needed now" or "waits for B49"; split B7 into shape rules (M1) and class data (before B49); state which shape-rule copy is authoritative after the swap | E8 | backend | Medium | applied: class data split into B57 |
| MA17 | Make L41's worked generator the U11 template and have A27 link to L41 instead of restating it | PR7 | language, shared, act | Medium | applied |
| MA18 | State XLA HLO text as the v3 input in the M4 criteria and the README, with the JAX and PyTorch export route | PR5 | backend, act | Medium | applied |
| MA19 | Make the quickstart's `bitcast_convert` the tutorial's first lesson, with one sentence on why `d0` is `u8` | PR6 | act | Medium | applied |
| MA20 | Extend K8's tripwire to wave 5 and the test track | PM17 | roadmap | Low | applied |
| MA21 | Record per-kernel oracle time for the T10 ISA specifications in T27 and print it in the coverage docs | PR12 | tests | Low | applied |

- Owner decision 47 is applied: non-breaking fixes before v3 ship as 2.x patch releases with released pins, and breaking changes wait for v3 (act/plan/versions-release.md, A32, shared/plan/U8.md, roadmap/waves.md release order).

## Pending in language

- MA3: applied; language/plan/surface.md (L44) states that the last 2.x release carries the converter (A28).
- MA3: applied; the L44 dependency of A28 is carried by U13, which absorbs L44.
- MA4: applied; L42, L44, and L45 in language/plan/surface.md are merged into U12, U13, and U14 (size `-`), whose sub-items hold their exits; L14 depends on U12.
- MA4: applied; L43 is stated as the layer over U3.
- MA1 via MA4: applied; L45's dependency on T37 is carried by U14, and the L45 row is a merged row.
- MA12: applied; L30 takes the findings of the outside architect's ISA specification (A24), beside D15.
- MA17: applied; L41 states that its worked generator is the U11 template.
- MA7: applied; L38 and L39 state development from wave 3 on the private heads, merging whole after the deletion pass.

## Checked and not carried

- PM15, PR9: approval condition 6 is closed; S8.4 bounds the approximate operations and approval.md lists T13 as unblocked.
- PM9: panel 12's text changes are already in B29, B31, B7, B6, B10, and A16 (backend/plan/search.md:16, isel.md:21,32,54, extractor.md:32; act/plan/versions-release.md:39).
- PR13 named B52; the torch adapter is B53 in the panel 12 decision, carried in MA9.
- PR7's move of U11 to wave 3: U11 depends on T12, which is wave 5.
- Pivots, not carried: L19 into wave 4 (PR2), panel 12 items in wave 6 (E13), a stop-loss on the encoding unit (E14, against Q10.4 and K20), and a typed `hbm` in the language (PR6).
