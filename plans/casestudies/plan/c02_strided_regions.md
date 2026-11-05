# C2 Strided and multi-dimensional regions

- What: The extension itself is L19; this item holds the case-study evidence and the verification.
- Evidence: DMA descriptors, TMA boxes, NKI access patterns, AIE-ML buffer descriptors, and strided stores, six public sites (gap rank 1).
- Footprint demonstration: write `dma_copy` (Trainium2), `tma_load_2d` (Hopper), and `tilestored` (AMX) with the covering-range idiom and show which false dependences remain on the gap bytes.
- Boundary: stride is a region property here; alignment is a buffer property (P11).
- Fail points: A strided write that aliases itself is silently accepted; noticed by a conformance test; the output-output apart constraint (S1.3.5) is checked on the evaluated strided footprint.
- Order and dependencies: C11 runs the sketches; this item depends on no L item, and L19 depends on this evidence.
- Verification: The demonstration is written down with the ran sketches; L19 is admitted only if the span idiom loses information.
