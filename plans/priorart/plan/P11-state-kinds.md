# P11 Alignment as a buffer property

## What

- Add the buffer property alignment, read by the allocator instead of generated integer checks.
- Stride is a region property and belongs to C2 and L19, not here.
- No new state kind in v3: descriptors, barriers, and async groups use plain integer state, attributes, and region reads.

## Benefits

- Backend state support (B16) keeps integer state as its target.
- The GPU case study (the Hopper sketch with mma, wgmma, and TMA) loads without a state kind.

## Caveats

- Typed state kinds return after v3 only if a second ran site shows plain state failing.
- Needs the glossary words for buffer and state.

## Fail points

- An alignment property that no case study uses, noticed at the case-study review; drop it.

## Order and dependencies

- Depends on P8, B16, L, C11.

## Verification

- The Hopper sketch runs in the oracle, and the backend selects at least one mma instruction.
