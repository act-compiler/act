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

1. Closed: the per-buffer initial value is `initial_value` on `acc.buffer` (S4.3), in S5.6, S1.3.11, S9, and both glossaries; plan item L46 (language/plan/norms.md) feeds O26 and B36.
   - O26's dependency column in oracle/plan/checks.md still lists only L5 and O20; the oracle author adds L46.
2. Closed: S7.3 lists 21 types with a table of zero, sign, infinity, and NaN; S7.5, S8.3, and S8.4 give the edge cases of `f4e2m1fn` and `f8e8m0fnu`; proposals.md row 1 no longer shows `ops.scaled_dot`.
3. Closed: the expectations vocabulary is `supported`, `partial`, `excluded`, `rejected` (S1.2) in tests/generators.md T11 to T14, tests/corpus, tests/README.md, tests/overhaul.md, and the M4 criterion (T12).
4. Closed: S10 and glossary/terms.md define "program" as a sequence of issues and "issue"; the kernel function now bans "program (for the function)", and "generator" is a Python package.
5. OD43 to OD46 (C++ on MLIR, PyTorch backend, in-house C++ e-graph library) appear in no track plan and no roadmap wave.
   - Panel 12 owns them (meeting/panel-12/agenda.md); its decision must add the items and waves.
   - backend/plan/decisions.md "Engine: Keep egg" must then name OD45 as the later direction.
6. Closed: S8.4 "Accuracy of approximate words" and S1.3.19 bound a finite result to `k_t` ulps of the exact value (16 at `f32` and `f64`, 8 at `f16` and `bf16`, 1 at 8- and 4-bit floats); D44 gives the evidence.
7. Closed. roadmap/milestones.md names a check for each criterion.
   - M3 names the D16 endings tests: `act-backend/tests/test_endings.py` and one `act-oracle` test per `KO021` to `KO040` code.
   - M1 names the O30 matrix; U11 and T35 are wave 5, so M2 names the L41 worked generator run in CI and the A27 stub build.
8. Closed. diagnostics/plan/README.md and D08 make D8 depend on "O20 or O11", the unroll outcome of O19.

## Minor items, no condition

- language/spec-draft/execution.md S5.6 says "a test oracle or a simulator"; "simulator" alone is banned for the test oracle.
- grammar.md S3.1 and semantics.md S7.1 examples name attributes `src` and `dst`, banned words for input and output regions in prose.
- layers.md S11.4 writes the example link `A == B*8` while its form is `E == k*D + c`; `A == 8*B` matches the form.
- minutes/owner-decisions.md lists decisions 31 to 33 after 46; numbers 2 to 4, 7 to 9, 18, 36, and 37 have no row; the file is the owner's and is not edited here.
- language/README.md says the draft follows owner decisions 1 to 42; decisions 43 to 46 do not touch the language.
- roadmap/q1-supporting.md and wave-3.md still list D8 on `O11`; q1-core.md and wave-4-7.md still carry the old T12 title; each takes the new text of diagnostics/plan/README.md and tests/generators.md.
- oracle/plan/fidelity.md O30 has no cell for S1.3.3 (region bounds), which the M1 fault-row criterion cites.
- language/spec-draft/generator_metadata.md S12.4 lists `supported`, `excluded`, and `rejected` for the expectations table and omits `partial`.
- diagnostics/plan/D06-unsupported-registries.md says rows carry `partial` or `rejected`; S1.2 adds `excluded`.

## Missing for an engineer

- Nothing beyond the open conditions; L46, the edge cases of L17, and the S8.4 ulp bound now unblock O26, B36, B18, C6, and T13.
