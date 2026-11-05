# C2 Strided and multi-dimensional regions

- What: Add an optional step per addressed dimension to a region, and multi-dimensional byte access patterns `[d0, start, sizes, strides]` on `d0`.
- Benefits: Covers DMA descriptors, TMA boxes, NKI access patterns, AIE-ML buffer descriptors, and strided stores without read-modify-write of gap bytes (gap rank 2).
- Caveats: Overlap rules of S5.2 and the placement marks of S5.5 must be restated for strided regions; the backend selector must match strided patterns.
- Fail points: A strided write that aliases itself is silently accepted; noticed by a conformance test; v3 must make it a defined error or a defined aliasing order.
- Order and dependencies: C1 and L, then B and O, then T.
- Verification: Express `dma_copy` (Trainium2), `tma_load_2d` (Hopper), and `tilestored` (AMX) without covering reads; compare oracle outputs against the old sketches.
