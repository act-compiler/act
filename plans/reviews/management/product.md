# Management review: product owner for the architecture and systems community

Lens: does the plan give an architect a tool to rely on, and a researcher a platform to extend.
Judged against the five product goals of meeting/final-reviews.md:10 and the owner decisions.
Paths are relative to plans/.

## Verdict

Approved for the architect-facing core, with a retargetability gap the milestones do not measure.

- The architect's loop is right: `python -m taidl check spec.py` with every problem at the author's line (language/spec-draft/conformance.md:15-19), a backend that excludes instead of rejecting (backend/plan/generators.md:42-44), and no undefined run-time case (casestudies/strengths.md:10).
- Reusability holds by type: one `Model` from one `check`, generators read nothing else, and a new generator touches no other repo (language/spec-draft/generator_metadata.md:31-38).
- Coverage cannot drift: the expectations table generates `coverage.md` and a case with no row fails collection (tests/generators.md:26-27).
- The weak spot is the compiler backend's reach: it excludes every stateful instruction, accepts XLA HLO text only, and no milestone measures what share of a real ISA it compiles.
- The timeline rests on a stale assist-factor tripwire and on a longest chain that omits the release train and the tutorial.

## Findings

| ID | Area | Finding | Evidence | Recommendation | Severity |
| -- | ---- | ------- | -------- | -------------- | -------- |
| PR1 | Retargetability | The v3 backend excludes any instruction that updates state or uses it beyond a read-only fold; state updates wait until after v3, yet configuration-state instructions appear in AMX, SME, Gemmini, and AIE-ML | backend/plan/coverage.md:5, :33; backend/coverage-table.md:30, :41; casestudies/strengths.md:6; roadmap/wave-3.md:19 | Add an M4 criterion that states, per case study, the share of instructions the backend uses; give state-update support a named post-v3 item and wave | High |
| PR2 | Retargetability | Strided regions are the top gap (High, six public sites) and the idiom loses information, yet L19 sits in wave 7 after M5 | casestudies/gaps.md:18; language/plan/extensions.md:11, :56; roadmap/wave-4-7.md:123 | Make C2 (wave 2) a decision point: if the footprint demonstration shows the idiom fails, L19 enters wave 4 or is the named first 3.1 item | Medium |
| PR3 | Schedule | The K1 tripwire of 4.3 was computed from 443 unassisted weeks; the waves now total 541 (510 through M5), so the need is 4.9 to 6.9 against 78 to 104 engineer-weeks | roadmap/risks.md:18; roadmap/waves.md:30; meeting/panel-8/decision.md:147-148 | Recompute K1 from the current totals and state the need as a range per milestone | High |
| PR4 | Critical path | load.md's longest chain (35) ends at T34; the row-order rules put U8, A23, A24, and A21 after it, so the path to M5 is at least 50 | roadmap/load.md:20-33; roadmap/waves.md:13, :54-58; roadmap/wave-4-7.md:97, :99, :107, :111 | Publish the path to each milestone with ordered-row edges counted; draft A24's exercise text on the private heads before A23 | Medium |
| PR5 | Usability | The v3 backend accepts XLA HLO text only; StableHLO, torch-mlir, and torch.fx are detected and named, and their readers move to the platform waves | backend/plan/failures.md:59-69; roadmap/milestones.md:71; meeting/panel-12/round3-fable.md:73 | State the accepted input format in the M4 criteria and the README, with the documented JAX and PyTorch export path to HLO text | Medium |
| PR6 | Usability | The quickstart's smallest accelerator needs three instructions because `d0` is `u8` and load and store each need `bitcast_convert`; the first thing an architect copies is a typed-view idiom | act/plan/terminology-docs.md:99; casestudies/gaps.md:22 | Let the quickstart declare a typed hbm (an `element_type` on `d0`, or the C1 idiom as one helper), or make the bitcast the tutorial's first lesson with one sentence of why | Medium |
| PR7 | New generators | Three items deliver one extension path: L41 guide with a worked generator (wave 3), A27 act page copied from a tested template (wave 3), U11 template generator (wave 5); M2 requires L41's generator in CI two waves before the template exists | language/plan/layers.md:61-76; act/plan/terminology-docs.md:104-114; shared/plan/U11.md:3; roadmap/wave-3.md:54, :59; roadmap/wave-4-7.md:85; roadmap/milestones.md:58 | One worked generator: U11's template is L41's example, U11 moves to wave 3, and A27 links to L41 instead of restating it | Medium |
| PR8 | Robustness | C12 calls the gap matrix a release criterion and C11 verifies only the front end and the oracle; no milestone cites the gap matrix or a backend run of the sketches | casestudies/plan/c12_rerun_on_v3.md:5, :9; casestudies/plan/c11_executable_corpus.md:10; roadmap/milestones.md:82-100 | Add to M4: each High row of the gap matrix has a decision backed by ran sites, and each sketch has a backend expectations row | Medium |
| PR9 | Robustness | Approximate-word accuracy is open: a hardware generator has no criterion for a finite `exp`, `log`, or `tanh` result, and T13 is blocked for those words | reviews/approval.md:52-54, :76 | Close approval condition 6 before M1 with a stated bound per word | High |
| PR10 | Consistency | L22 is cut in the language plan yet scheduled in wave 7 with weight 6 inside the 541 | language/plan/extensions.md:14, :83; roadmap/wave-4-7.md:126; roadmap/waves.md:8 | Move L22 to the dropped rows and regenerate the totals | Low |
| PR11 | Consistency | The Ascend study is required by the wave 1 gate and K15, ruled "at the start of C12" by R15, and C12 says no Ascend study is made | roadmap/waves.md:45; roadmap/risks.md:31; minutes/rulings.md:25; casestudies/plan/c12_rerun_on_v3.md:4 | Decide once: a C14 Ascend item in wave 2, or remove the gate row and K15 | Low |
| PR12 | Usability | The oracle bound allows 60 s trace plus compile for 16,384 issues, and the oracle stays on Python and JAX until after M5, so an architect cannot size a test kernel from the docs | meeting/resolutions.md:20-21; roadmap/risks.md:26; roadmap/milestones.md:54 | Record per-kernel oracle time for the T10 real ISA specifications in T27 and print it in the coverage docs | Low |
| PR13 | Platform value | B52 (torch adapter) depends only on B47 and B42 but sits in wave 9 behind the e-graph library; the native input door is the platform's first architect-visible value | meeting/panel-12/round3-fable.md:93, :100 | When panel 12 closes, place B52 in wave 8 beside B47 | Low |

## What would make researchers rely on it

- A per-accelerator statement of what the backend compiles, generated like `coverage.md`, so a user of the Gemmini or AMX sketch knows before writing a kernel which instructions the compiler will never pick (PR1, PR8).
- A stated input format and export recipe on the README, since most research kernels start in JAX or PyTorch (PR5).
- A first example that reads like the architect's own datasheet, not a bitcast idiom (PR6).
- One extension path with one worked generator that CI runs, so act-allo and the next generator copy a package rather than a page (PR7).

## Top five actions

1. Add retargetability to M4: a backend share per case study and a gap-matrix High-row decision backed by ran sites (PR1, PR8).
2. Recompute the K1 threshold from 541 weeks and publish the path to M5 with ordered-row edges (PR3, PR4).
3. Decide L19's route at C2 in wave 2 instead of leaving the top gap in wave 7 (PR2).
4. Collapse L41, U11, and A27 onto one worked generator in wave 3 (PR7).
5. State the v3 input format in M4 and the README, and fix the quickstart's first contact (PR5, PR6).
