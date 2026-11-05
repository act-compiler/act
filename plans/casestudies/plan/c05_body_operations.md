# C5 Body-taking operations and convolution

- What: Admit StableHLO operations with a region body or windows: `reduce` with a custom reducer, `reduce_window`, `convolution`, and a bounded `scan`.
- Benefits: Makes CNN accelerators (Eyeriss, conv-mode units) and scan instructions expressible without unrolling (gap rank 3).
- Caveats: The line grammar of S3.1 must carry nested blocks and convolution dimension labels; ACT-Backend rewrites must learn windows.
- Fail points: `convolution` padding and dilation attributes are misread; noticed by differential tests against the StableHLO interpreter.
- Order and dependencies: P, then L, then O, then B, then T.
- Verification: The Eyeriss `conv_rs_3x3` written as one `convolution` matches the unrolled sketch on random inputs.
