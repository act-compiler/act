# C5 Body-taking operations and convolution

- What: The extensions are L20 (`convolution`) and the cut L22; this item holds the case-study evidence and the verification.
- Evidence: `tensor_tensor_scan` (Trainium2) and Tensix are the real sites; Eyeriss and CGRA are invented and count zero; NVDLA joins when C12 runs it (gap rank 6).
- Scope: a bounded `scan` is the only body-taking form a study asks for; `reduce` with a custom reducer and `reduce_window` have no site.
- Fail points: `convolution` padding and dilation attributes are misread; noticed by differential tests once L20 lands.
- Order and dependencies: C12 (NVDLA); this item depends on no L item, and L20 depends on this evidence.
- Verification: The Eyeriss `conv_rs_3x3` unrolled sketch is the baseline an admitted `convolution` must match on random inputs.
