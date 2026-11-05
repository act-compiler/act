# P11 Typed hardware state kinds and data-model layout properties

## What

- Add declared state kinds beyond integers: descriptor, async group or barrier, read-only configuration; each with the operations an instruction may perform on it.
- Add data-model properties stride and alignment, read by the allocator instead of generated integer checks.
- Needed for the GPU tensor-core case study: mma, wgmma, TMA.

## Benefits

- Backend state support (B16) has a typed target instead of integers in an e-graph.

## Caveats

- Largest language change in the release; needs the owner's terminology decisions for state words.

## Fail points

- A state kind that no case study uses, noticed at the case-study review; drop it.
- Asynchrony that the sequential execute function cannot express, noticed by the formal track.

## Order and dependencies

- Depends on P8, B16, L, C.

## Verification

- The GPU case study loads, runs in the oracle, and the backend selects at least one mma instruction.
