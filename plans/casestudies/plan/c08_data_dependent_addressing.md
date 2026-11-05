# C8 Data-dependent addressing

- What: The extension is L23; this item is the one study of data-dependent addressing (Q2.5), its result admits or rejects L23.
- Evidence: Indirect DMA is expressible with a workaround, a covering-range attribute plus `gather` from a parameter region; `ldtilecfg` is expressible as is; what remains is a descriptor written at run time and read later (Trainium DGE, Hopper tensor map) and state updates read from data.
- Scope: no new state kind; descriptors use existing state, attributes, and region reads.
- Fail points: The backend cannot select such instructions and silently skips them; noticed by the coded rejection a backend gives for an unsupported input.
- Order and dependencies: C11; this item depends on no L item, and L23 depends on this evidence.
- Verification: Trainium2 indirect DMA runs under the oracle; the study states whether the Trainium DGE descriptor needs more than state and attributes.
