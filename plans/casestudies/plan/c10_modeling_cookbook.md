# C10 Modeling cookbook and scope notes

- What: A spec appendix of idioms: hidden latch as a data model, register file, accumulate, strided read, mode select, instruction family by loop, float bits route, field unpacking, typed view, barrier as counter, aliased buffers.
- Scope notes: SIMT, VLIW bundles, tile arrays, programmable units, and spatial-mapping designs (Groq, Cerebras, SambaNova); concurrency is out of v3 and program order is the only order.
- Constraint as wait: sound only for a sequential program order; a program that would legitimately wait faults.
- Doc gap: mixed-type `dot` and the `gather` and `dynamic_slice` attribute forms are given only by reference to StableHLO.
- Caveats: Idioms go stale as extensions land; each is rewritten in the same release.
- Fail points: An idiom that the front end rejects; noticed when the cookbook snippets run as tests.
- Order and dependencies: L, then T.
- Verification: Every snippet is an ISA specification the front end accepts, run in CI.
