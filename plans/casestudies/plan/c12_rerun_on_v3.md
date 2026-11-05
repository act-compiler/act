# C12 Re-run the case studies on v3

- What: Re-express every case study with the v3 constructs and redo the verdict tables, adding the missing Ascend study.
- Benefits: Measures whether v3 closed the gaps; the gap matrix becomes a release criterion.
- Caveats: Requires C1 to C9 to have landed; partial landing gives a partial re-run.
- Fail points: A construct lands but does not remove the workaround; noticed as an unchanged verdict row.
- Order and dependencies: C1, C2, C3, C4, C5, C6, C7, C8, C9, C11.
- Verification: Each High row of the gap matrix moves to expressible as is in at least the case studies that raised it.
