# C12 Re-run the case studies on v3

- What: Re-express every case study with the v3 constructs, redo the verdict tables and the gap matrix counts, and add the NVDLA study.
- NVDLA: a descriptor-driven engine with public register documentation; it is the second site for convolution and an input to the C8 study.
- No Ascend study is made.
- Benefits: Measures whether v3 closed the gaps; the gap matrix is a release criterion.
- Caveats: Requires the admitted extensions (L17) and C11 to have landed.
- Fail points: A construct lands but does not remove the workaround; noticed as an unchanged verdict row.
- What: The re-run and its gap-matrix pass run on the private v3 heads in wave 3 and feed C5; the generated backend share of instructions is reported per case study.
- What: The M4 exit repeats the gap-matrix pass on the v3 release candidate.
- What: Each sketch carries a backend expectations row: the instructions the backend is expected to support, exclude, or reject, with the code.
- Later: State updates are a later row of the expectations, after v3.
- Order and dependencies: C1, C8, C10, C11, L17.
- Verification: Each High row of the gap matrix has a decision backed by ran sites; each case study has a backend share and an expectations row.
