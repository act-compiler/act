# C10 Modeling cookbook and scope notes

- What: A spec appendix of patterns: hidden latch as a data model, register file, accumulate, strided read, mode select, barrier as counter; plus scope notes for SIMT, VLIW bundles, tile arrays, and programmable units.
- Benefits: Answers the doc gaps every case study hit; most were modeling questions, not missing constructs.
- Caveats: Patterns go stale as C1 to C9 land; each must be rewritten in the same release.
- Fail points: A pattern that the front end rejects; noticed when the cookbook snippets run as tests.
- Order and dependencies: L, then T.
- Verification: Every snippet is an ISA description the front end accepts, run in CI.
