# Final approval

Final approval of the plans against the owner decisions, panel decisions 1 to 11, the two joint decisions, and the resolutions.
Panel 12 is in progress and is not judged here.

## Verdict

Approved with conditions.

- The owner decisions on markers, overlap, failure policy, check function, layers, generator metadata, versions, releases, and dependencies appear in the plans they affect.
- No plan reopens a panel decision that an owner decision or a later panel kept, such as Q2.1 and Q2.2 under OD22 and OD30, or Q4.12 under OD29.
- The spec draft has no undefined run-time case in S1.3 and none of the banned words of the glossary, after the fixes below.
- The conditions below must close before M1, except condition 5, which closes with panel 12.

## Fixes made in place

| File | Fix |
| --- | --- |
| language/spec-draft/*.md (13 pages) | "of the spec" becomes "of the TAIDL language specification"; "the spec" is a banned word (S10) |
| language/spec-draft/layers.md S11.1 | "honors" becomes "applies" (banned word) |
| language/spec-draft/generator_metadata.md S12.1 | "partition hint" becomes "partition choice" ("hint" is banned for generator metadata) |
| language/spec-draft/execution.md S5.6 | "each precondition it cannot see" becomes "each fault case it cannot see" |
| language/spec-draft/conformance.md, failure policy | A generator's rejections now include reserved names (S4.1) and shared fixed bits (S11.3), which S4.1 and S11.3 already state |
| language/spec-draft/semantics.md S7.3 | Names `f4e2m1fn` and `f8e8m0fnu` as the L17 exception to "4-bit floats are outside" (Q2.4) |
| oracle/plan/architecture.md O19, decisions.md, overhaul.md | The unroll bound is the Q5.1 resolution (64 by 64 trips, 60 s, 2x, op-count cap), not "the owner sets" |
| roadmap/q1-core.md, wave-1.md, waves.md, risks.md K16, milestones.md M2 | Same O19 bound; K16 now trips on a missed bound, not a missing one |
| roadmap/milestones.md M5 | Cold-start trial takes the Q8.8 resolution: the Q7.5 example, 4 hours each |
| oracle/plan/decisions.md, checks.md O26 | Applies the joint decision on unwritten reads: declared initial value exempts a buffer; the open item is closed |
| roadmap/q2-core.md, wave-4-7.md | O26 title follows checks.md |
| backend/plan/decisions.md, extractor.md B36 | A buffer with a declared initial value may be read before any write |
| backend/plan/encoding.md | "refused" becomes "rejected" |
| glossary/terms.md, rulings.md, words-to-avoid.md | Coverage statuses are `supported`, `partial`, `excluded`, `rejected`, as S1.2 and both coverage tables use |
| roadmap/wave-4-7.md, waves.md, milestones.md | D13 (after v3) moves from wave 4 to wave 7; wave 4 is 50 items, weight 130; wave 7 is 11 items, weight 31; totals stay 247 and 541 |
| roadmap/waves.md rules, wave-4-7.md wave 7 | The later-item rule covers other-supporting.md as well as other-core.md |
| reviews/issue-log/README.md | The stale "roadmap still lists them" line now states how the roadmap carries F7, F9, A14, and A22 |

## Conditions

1. The per-buffer initial value of the joint decision on unwritten reads is in no language text.
   - The spec omits it: language/spec-draft/declarations.md S4.3, execution.md S5.6 "Storage before the first issue", overview.md S1.3.11, glossary.md "initial value" row.
   - No language item declares it; the decision puts it next to L32 (language/plan/norms.md).
   - O26 and B36 now depend on it; add the item, its spelling on `acc.buffer`, and an edge to O26.
2. `f4e2m1fn` and `f8e8m0fnu` (L17, Q2.4) have no edge cases in the spec.
   - Needed: S7.3 list and count ("19 element types" in grammar.md S2.1 and glossary.md), S8.3 type rows, and S8.4 rows for a type with no zero, sign, or infinity (`f8e8m0fnu`).
   - language/proposals.md row 1 still shows `ops.scaled_dot`, which Q2.4 and S8.5 put outside the language.
3. The expectations vocabulary disagrees with the spec.
   - S1.2 and S12.4 use `supported`, `excluded`, `rejected`; tests/generators.md T11 and T12, tests/corpus/layers.md, failure.md, sections.md, and roadmap/milestones.md M4 (T12) use `supported` or `unsupported`.
   - Pick one set and apply it to T11, T12, and the M4 criterion.
4. The spec uses "program" for a sequence of issues (execution.md S5.6, overview.md S1.3), while S10 has no "program" row and glossary/terms.md bans "program" for the kernel function.
   - Add a term row that separates the two, or rename one.
5. OD43 to OD46 (C++ on MLIR, PyTorch backend, in-house C++ e-graph library) appear in no track plan and no roadmap wave.
   - Panel 12 owns them (meeting/panel-12/agenda.md); its decision must add the items and waves.
   - backend/plan/decisions.md "Engine: Keep egg" must then name OD45 as the later direction.
6. Float accuracy of the approximate words is open for finite results.
   - language/spec-draft/operation_meaning.md S8.4 fixes only their infinity, zero, and NaN results.
   - A finite `exp`, `log`, or `tanh` result has no bound and no normative reference value, so a conforming generator for hardware has no criterion; state a bound or make the pinned reference value normative.
7. Three milestone criteria name no check.
   - roadmap/milestones.md M3 "The failure policy is in force for every generator (D16)": name D16's verification test.
   - M1 "the region, in-place, and overlap fault rows hold (L36, L35, L5)": name the test item that runs those S1.3 rows.
   - M2 "the guides for adding a generator exist (L40, L41, A27)": name the check, for example the U11 template generator built from the guide.
8. diagnostics/plan/README.md D8 depends on O11, a fallback that lands only if O19 fails.
   - Make the edge conditional, or point D8 at O20 or O11, whichever lands.

## Minor items, no condition

- language/spec-draft/execution.md S5.6 says "a test oracle or a simulator"; "simulator" alone is banned for the test oracle.
- grammar.md S3.1 and semantics.md S7.1 examples name attributes `src` and `dst`, banned words for input and output regions in prose.
- layers.md S11.4 writes the example link `A == B*8` while its form is `E == k*D + c`; `A == 8*B` matches the form.
- minutes/owner-decisions.md lists decisions 31 to 33 after 46; numbers 2 to 4, 7 to 9, 18, 36, and 37 have no row; the file is the owner's and is not edited here.
- language/README.md says the draft follows owner decisions 1 to 42; decisions 43 to 46 do not touch the language.

## Missing for an engineer

- The declaration of a per-buffer initial value (condition 1) is the one language feature a decided item needs and no item carries.
- Edge cases of the two new element types (condition 2) block B18, C6, and the oracle's L17 rows.
- An accuracy criterion for approximate words (condition 6) blocks the differential tests T13 for those words.
