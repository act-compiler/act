# Plan: language extensions for accelerator coverage

These items extend TAIDL so it can describe the main accelerator families from industry and academia.
Each names the motivating designs, the smallest generic construct, and its cost; none changes the grammar this quarter without its own review.
Sizes assume the shared 3-4 engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L17 | Low-precision and block-scaled element types, `scaled_dot` | L2, L3 | M | Q1 |
| L18 | Write masks on output regions | L3, L24 | M | after v3 |
| L19 | Strided regions | L5 | M | after v3 |
| L20 | `convolution` with window attributes | L3 | M | after v3 |
| L21 | Rounding and saturation attributes on `convert` | L3 | S | after v3 |
| L22 | Sub-computations for `reduce`, `scatter`, `sort`, `map` | L3 | L | cut from v3 |
| L23 | Indirect regions: starts read from a buffer | L5 | M | late Q2 |
| L24 | Multi-core, async DMA, mask, stride, and saturation idioms, documented without syntax | L1 | S | Q1 |

- An item deferred "after v3" has an idiom that describes it today (L24); it returns when a second real ISA description needs the syntax.
- L22 names no accelerator and is the change that lets bodies into the grammar; it is cut from v3.
- The general rule that an extension needs two case studies is pending panel (R59).

## L17 Low-precision and block-scaled element types, `scaled_dot`

- What: add `f4e2m1fn`, `f8e8m0fnu`, and XLA's `scaled_dot` (block scales per operand); `f6e2m3fn`, `f6e3m2fn`, `s1`, and `u1` wait for a second named user.
- Changes: `TYPE` in `IDLV2.g4`, `TYPES` in `taidl/data_model.py`, `OPERATIONS`; existing types kept.
- Breaking: no, additive.
- Benefits: covers MX formats on NVIDIA Blackwell tensor cores, AMD XDNA2, Trainium2 and Gaudi3 FP8, and binary networks.
- Caveats: JAX support for each type varies; O may list some as unsupported (L7).
- Fail points: XLA type spellings change; notice by L2's pinned version.
- Order and dependencies: after L2, L3; grammar `TYPE`, `data_model.TYPES`, B and O type tables.
- Verification: a block-scaled matmul instruction in an example runs in O.

## L18 Write masks on output regions

- What: an optional `pred` value `%Mask<k>` of the region shape; where false, output region k keeps its old content.
- Changes: `check_values` in `taidl/instruction.py` learns `%Mask<k>`; `%In<k>` and `%Out<k>` kept.
- Breaking: no, additive.
- Benefits: covers Arm SME and SVE predicates, Trainium NKI masks, Ascend vector masks, partial blocks at array edges.
- Caveats: a masked output reads its old content, so it acts as an implicit input; S5.2 must say so.
- Today's idiom (L24): pass the output region as an input, `select` by a `pred` region, mark it in place.
- Fail points: B's allocator treats the output as write-only; notice by a T test with a partial mask.
- Order and dependencies: after L3; B and O implement.
- Verification: SME-style predicated store example runs in O and B agrees.

## L19 Strided regions

- What: an optional fourth list in a region, one stride per addressed dimension, default 1.
- Changes: `check_regions` in `taidl/instruction.py`; three-part regions stay valid.
- Breaking: no, additive.
- Benefits: covers NVIDIA TMA boxes with strides, Gemmini and NVDLA strided DMA, Tenstorrent strided reads.
- Caveats: overlap rules in placement marks (S5.5) must account for strides.
- Fail points: placement constraints wrong for strided inputs; notice by a T overlap test.
- Order and dependencies: after L5.
- Verification: strided load example passes in O; placement constraints tested.

## L20 `convolution` with window attributes

- What: XLA `convolution` with `window` and `dim_labels` written as brace lists of integers rather than XLA's `3x3` text.
- Changes: `OPERATIONS` gains `convolution`; S8.5 loses its row.
- Breaking: no, additive.
- Benefits: covers Eyeriss-style and NVDLA conv engines and TPU conv without spelling them as `dot` plus reshapes.
- Caveats: new attribute forms or a fixed attribute set; dimension labels are a new kind of value.
- Fail points: B's e-graph rules cannot match it; notice by B listing it unsupported.
- Order and dependencies: after L3.
- Verification: an NVDLA-style conv instruction in an example runs in O.

## L21 Rounding and saturation attributes on `convert`

- What: optional `rounding=` (nearest even, toward zero, stochastic with a seed input) and `saturate=` on `convert`.
- Changes: `OPERATIONS` row of `convert`; plain `convert` keeps its meaning.
- Breaking: no, additive.
- Benefits: covers Trainium stochastic rounding, Gaudi and Ascend saturating casts, int8 requantization.
- Caveats: stochastic rounding needs a seed operand, so results are a function of state.
- Fail points: no StableHLO reference for stochastic rounding; notice when L2 cannot cite one.
- Order and dependencies: after L3.
- Verification: saturating `s32` to `s8` example matches a hand-computed result.

## L22 Sub-computations

- What: named blocks besides `ENTRY` that `reduce`, `scatter`, `sort`, `select_and_scatter`, and `map` call.
- Changes: `module` rule in `IDLV2.g4` takes named blocks; a single `ENTRY` block stays valid.
- Breaking: no, additive.
- Benefits: removes the largest class of operations that stay out (S8.5); custom reducers like max-with-index.
- Caveats: largest grammar change; B's e-graph must match bodies.
- Fail points: two generators read nested values differently; notice by T differential tests.
- Order and dependencies: after L3; coordinate with B early.
- Verification: `scatter` with an add body runs in O and B.

## L23 Indirect regions

- What: a start given as a value read from another region, as in a gather from a buffer.
- Changes: region starts in `taidl/instruction.py`; expression starts stay valid.
- Breaking: no, additive.
- Benefits: covers Trainium indirect DMA, NVIDIA TMA gather, embedding lookups on MTIA and TPU SparseCore.
- Caveats: starts are no longer known before the instruction runs; B's allocator loses static addresses.
- Fail points: placement marks become unsound; notice by rejecting marks on indirect regions.
- Order and dependencies: after L5 and L17; the second extension of v3.
- Verification: embedding-lookup example runs in O.

## L24 Multi-core and async DMA idioms

- What: document three idioms that need no syntax.
- Idioms: a core grid is an extra addressed dimension; an async DMA is two instructions sharing a state variable as a token; a masked write is a `select` over the output region passed as an in-place input.
- Changes: A tutorials and S1 scope; no code.
- Breaking: no.
- Benefits: covers Tenstorrent Tensix grids, AMD AIE tiles, Cerebras PEs, and NKI DMA queues without syntax.
- Caveats: timing stays outside TAIDL (S1); ordering is the program's.
- Fail points: users expect concurrency semantics; notice by questions in A tutorials.
- Order and dependencies: after L1; A writes a tutorial.
- Verification: a two-core example in A runs in O.
