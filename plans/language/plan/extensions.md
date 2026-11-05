# Plan: language extensions for accelerator coverage

These items extend TAIDL so it can describe the main accelerator families from industry and academia.
Each names the motivating designs, the smallest generic construct, and its cost; none changes the Python surface this quarter without its own review.
Sizes assume the shared 3-4 engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L17 | Element types `f4e2m1fn` and `f8e8m0fnu` | L2, L3 | S | Q1 |
| L18 | Write masks on output regions: an idiom in L24 | L3, L24 | M | after v3 |
| L19 | Strided regions, first in line after v3 | L5 | M | after v3 |
| L20 | `convolution` with window attributes: an idiom in L24 | L3 | M | after v3 |
| L21 | Rounding and saturation attributes on `convert`: an idiom in L24 | L3 | S | after v3 |
| L22 | Sub-computations for `reduce`, `scatter`, `sort`, `map` | L3 | L | cut |
| L23 | Indirect regions: starts read from a buffer, joined with the data-dependent addressing study | L5 | M | after the study |
| L24 | Multi-core, async DMA, mask, stride, saturation, and `scaled_dot` idioms, documented without syntax | L1 | S | Q1 |

- Admission rule (Q2.3): syntax enters only with an idiom-failure argument plus two real, publicly documented sites in owner-15 scope.
- "Idiom fails" means it loses information or cannot be stated, not that it is verbose.
- Sites count per construct in one gap-matrix column; a site counts only after its sketch ran through the front end; invented ISAs count zero.
- The rule applies to every extension, L17 included; the L-track lead writes the count, and a reviewer outside the track checks each cited document and run.
- v3 adds no new state kind and no concurrency (Q2.5, Q2.6); data-dependent addressing is one study (RA9), with a run-time descriptor read later (Trainium DGE) first and NVDLA as its queued input.
- An item deferred "after v3" has an idiom that describes it today (L24).

## L17 Element types `f4e2m1fn` and `f8e8m0fnu`

- What: add `f4e2m1fn` and `f8e8m0fnu` (Q2.4); no `tf32` storage type; `scaled_dot` stays an idiom in L24 (convert, broadcast the scale, multiply, `dot`), exact for power-of-two scales.
- What: their edge cases are in the spec: the S7.3 type table, the S7.5 literal rules, the S8.3 rows, and the S8.4 reducer, `convert`, and accuracy rows.
  - `f4e2m1fn` has no infinity and no NaN: overflow saturates to 6 or -6, and NaN converts to +0.
  - `f8e8m0fnu` has no zero, sign, or infinity: only data-movement words, `compare`, `select`, and `convert` take it; zero, negatives, and overflow convert to NaN.
- Changes: `taidl.types` and `TYPES` in `taidl/data_model.py`; existing types kept.
- Breaking: no, additive.
- Benefits: covers the MX formats of the two public sites, NVIDIA Blackwell and NeuronCore-v4 (Trainium3).
- Caveats: JAX support for each type varies; O may list some as unsupported (L7).
- Fail points: XLA type spellings change; notice by L2's pinned version.
- Order and dependencies: after L2, L3; `taidl.types`, `data_model.TYPES`, B and O type tables.
- Verification: a block-scaled matmul written with the L24 idiom runs in O; a T test per S8.4 row of the two types checks the pinned stack, the sign of zero included; `ml_dtypes` 0.6.0 matches the rows except NaN to `f4e2m1fn`, which O lowers with a `select` to +0.
- Verification: T tests single rounding of a plain number (S7.5): `1 + 2**-8 + 2**-40` at `bf16` and `2**53 + 2**29 + 1` at `f32`.

## L18 Write masks on output regions

- What: an optional `pred` value `%Mask<k>` of the region shape; where false, output region k keeps its old content.
- Changes: `check_values` in `taidl/instruction.py` learns `%Mask<k>`; `%In<k>` and `%Out<k>` kept.
- Breaking: no, additive.
- Benefits: covers Arm SME and SVE predicates, Trainium NKI masks, Ascend vector masks, partial blocks at array edges.
- Caveats: a masked output reads its old content, so it acts as an implicit input; S5.2 must say so.
- Today's idiom (L24): pass the output region as an input and `select` by a `pred` region; the output and its input copy are identical text, so the pair is marked `set_inplace` (S5.5).
- Fail points: B's allocator treats the output as write-only; notice by a T test with a partial mask.
- Order and dependencies: after L3; B and O implement.
- Verification: SME-style predicated store example runs in O and B agrees.

## L19 Strided regions

- What: an optional fourth list in a region, one stride per addressed dimension, default 1.
- Changes: `check_regions` in `taidl/instruction.py`; three-part regions stay valid.
- Breaking: no, additive.
- Benefits: covers NVIDIA TMA boxes with strides, Gemmini and NVDLA strided DMA, Tenstorrent strided reads.
- Caveats: the `apart` and `same` expansions of S5.5 must account for strides.
- Fail points: placement constraints wrong for strided inputs; notice by a T overlap test.
- Order and dependencies: after L5; admitted only if the C2 footprint demonstration shows the span idiom fails (Q2.3, Q2.4).
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

- Status: cut (Q2.4); no accelerator in scope needs bodies in the grammar.
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
- Order and dependencies: waits for the data-dependent addressing study (RA9, Q2.5); covering all of `d0` is costly but expressible today.
- Verification: embedding-lookup example runs in O.

## L24 Multi-core and async DMA idioms

- What: document the idioms that need no syntax, one per line in the cookbook.
- Idioms:
  - A core grid is an extra addressed dimension.
  - An async DMA is two instructions sharing a state variable as a token; program order is the only order.
  - A masked write is a `select` over the output region passed as an input (L18).
  - A strided read declares its span (L19).
  - A convolution is `dot` plus reshapes (L20).
  - A saturating or requantizing cast is `clamp` before `convert` (L21).
  - `scaled_dot` is convert, broadcast the scale, multiply, `dot` (L17).
  - "Apart" and "identical or apart" are `set_apart` and the default; any finer overlap rule is `set_may_overlap` plus a written constraint (S5.5).
- Changes: A tutorials and S1 scope; no code.
- Breaking: no.
- Benefits: covers Tenstorrent Tensix grids, AMD AIE tiles, Cerebras PEs, and NKI DMA queues without syntax.
- Caveats: timing stays outside TAIDL (S1); ordering is the program's.
- Fail points: users expect concurrency semantics; notice by questions in A tutorials.
- Order and dependencies: after L1; A writes a tutorial.
- Verification: a two-core example in A runs in O.
