# C12 Re-run the case studies on v3

- What: Re-express every case study with the v3 constructs, redo the verdict tables and the gap matrix counts, and add the NVDLA study.
- NVDLA: a descriptor-driven engine with public register documentation; it is the second site for convolution and an input to the C8 study. No Ascend study is made.
- Benefits: Measures whether v3 closed the gaps; the gap matrix is a release criterion.
- Caveats: Requires the admitted extensions (L17) and C11 to have landed.
- Fail points: A construct lands but does not remove the workaround; noticed as an unchanged verdict row.
- Order and dependencies: C1, C8, C10, C11.
- Verification: Each High row of the gap matrix has a decision backed by ran sites.
