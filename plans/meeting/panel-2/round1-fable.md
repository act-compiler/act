# Panel 2, round 1, Fable: accelerator architect

Lens: whether the language can describe TPU, Trainium, Hopper and Blackwell tensor cores, AMX, SME, Gaudi, NVDLA, Gemmini, Tensix, AIE-ML, and CGRAs as their public documents define them.
Assumption throughout: a description writer models what the hardware does to storage and state, so a rule that hardware cannot observe (text spelling, a call order) is a cost without a hardware counterpart.
External sources are named by host and page; no source is linked.

### Q2.1

Position: Option C; aliasing is derived from identical region texts, `set_inplace` goes, and any other input-output overlap at issue is a fault.

1. Every real in-place case I know is an accumulator whose input and output regions are spelled identically: AMX `tdpbf16ps` reads and writes tile 2 (`casestudies/amx.md`, `tmm` rows), Hopper `wgmma` reads and writes `rf` at `@a.d` (`casestudies/hopper.md:46-49`), NKI `nc_matmul` with `accumulate=True` adds into the same PSUM tile (nki.isa.nc_matmul page, awsdocs-neuron.readthedocs-hosted.com), and the softmax example (`act-backend/examples/mini_trn/mini_trn.py:96-99`).
2. Hardware aliasing is a property of addresses, so a flag that can name a different buffer (`act-backend/tests/test_extraction.py:507` marks `d0` for a `d2` output) describes nothing that exists under owner decision 17 and must be rejected, not kept.
3. Option D (evaluate at issue) is truest to hardware but costs the backend its static knowledge of which regions share storage, and it is what the front end would need to check at run time, which it cannot (`taidl/instruction.py:163-172` checks nothing).
4. Option C gives the backend a static answer (equal text) and the oracle a defined error (any other overlap), which satisfies owner 16 and keeps the L24 masked-write idiom, whose output is passed as an input with the same text (`language/plan/extensions.md:40`).
5. Partial overlap of an input and an output is a real hardware hazard (DMA engines document src-dst overlap as unsupported), and a fault is the honest defined behavior.

Evidence: `language/spec-draft/instructions.md:49-55`; audit rows R5, RA2; `semantics/rulings.md:8-13`; nki.isa.nc_matmul page on accumulate modes.
Confidence: medium-high.
Changes my mind: a real ISA whose in-place operand is a different register file from the result, or a backend need to alias regions spelled differently at the same address.

### Q2.2

Position: Option C; remove both marks and ship one documented constraint idiom for apart and for whole-or-apart.

1. The marks compare starts by token text (`instructions.md:60-63`, `taidl/instruction.py:216-218`); no accelerator compares spellings, it compares addresses, so the construct has no hardware meaning a writer can check against a datasheet.
2. The two conditions are each one constraint over starts and sizes that a writer already can write (`instructions.md:35`), and under owner 16 a violated constraint is the same defined fault either way.
3. Strides (L19) break the marks' overlap arithmetic (`extensions.md:51`), while a user constraint over strided extents stays writable.
4. The marks force a second state read point (`instructions.md:36-37`, audit R7); removing them leaves one rule, constraints read the pre-state.
5. Tests in three repos and spec pages must migrate (audit R58 list), which is a bounded cost inside the owner 10 batch.

Evidence: `taidl/instruction.py:174-244`; R58; audit R58; minimalist row on the marks.
Confidence: high.
Changes my mind: a generator that needs a mark's intent, not its constraint text, for allocation, with a cited call site.

### Q2.3

Position: an extension enters when two real accelerators with public documents need it, each shown by a sketch that passed the front end, and the idiom fails rather than merely grows; invented ISAs count zero; the panel counts, per construct, in a matrix column.

1. The AMX miscount shows unverified counting is unreliable: `LDTILECFG` is one 64-byte load of palette, 16 colsb bytes, and 8 rows bytes (ldtilecfg page, felixcloutier.com), expressible as one instruction (CP2), yet it was counted as a gap.
2. Public documents exist for Hopper (PTX ISA, docs.nvidia.com), AMX (Intel SDM), Trainium (NKI ISA pages), SME (Arm A64 reference), and Gemmini, while Eyeriss and the CGRA sketches invent their command sets (CP9), so they prove nothing about a shipped ISA.
3. "Describable by idiom" must be split: a verbose idiom is Medium severity by the matrix's own definition (`casestudies/gaps.md:4`), while an idiom that loses information or needs an unbounded input earns syntax; this is Option C's test.
4. A numeric threshold alone admits a construct two toy sketches "need"; the per-extension review (Option D) is the procedure, the threshold and the idiom test are its bar.
5. The rule applies to L17 too (Option E): Blackwell block-scaled `mma` is public in PTX, and the Trainium2 `nc_matmul` page lists fp8 types but no MX type, so the second MX site is XDNA2 or Gaudi3 and the owner should confirm it counts (CP11).

Evidence: CP2, CP5, CP9, CP11; `casestudies/method.md` (no sketch was run); R59; `extensions.md:4`.
Confidence: medium-high.
Changes my mind: a Q2 capacity figure showing per-extension review cannot fit, which would argue for the plain two-site count.

### Q2.4

Position: L17 in v3 (`f4e2m1fn`, `f8e8m0fnu`, `scaled_dot`, no `tf32` storage type); L19 first after v3 as additive syntax; L23 late Q2; L18, L20, L21 idioms; L22 cut.

1. L17: block-scaled types are the current hardware direction (Blackwell `mma` block-scale kinds in PTX; MX formats in OCP), and no idiom builds an e2m1 or e8m0 value from `u8` without bit operators the language lacks (`gaps.md:22`); `tf32` is a 32-bit storage with truncated compute on both Hopper and Trainium (`nc_matmul` lists `tfloat32` beside `float32`), so `reduce_precision` on `f32` is the right model (CP11).
2. L19: strided access is universal: `cuTensorMapEncodeTiled` carries `globalStrides` and `elementStrides` (CUDA driver API, docs.nvidia.com), NKI `dma_copy` supports 3-4 strided dimensions, AIE-ML buffer descriptors have 3-D strides (`casestudies/aie_ml.md:6`), Gemmini `config_ld` sets a stride.
3. L19's read idiom (span plus strided `slice`) is verbose but exact; its write idiom needs the span as an in-place input and a `select`, which is the L24 mask idiom, so it is describable and belongs after v3 by the Q2.3 rule, but it is first in line.
4. L23: indirect DMA's `gather` idiom needs a bounded covering range (CP3), and real indirect DMA has none; the only honest covering input is all of `d0`, whose size is an issue-time parameter (RA10), so the idiom fails and L23 earns syntax.
5. L18 and L21: SME predication and saturating casts are exact under `select` and `clamp` plus `convert`; stochastic rounding is a function of hidden random state that no reference operation defines, so it stays out.
6. L20 and L22: conv engines with public docs (NVDLA) have no study yet, and `dot` plus `pad` and `slice` is exact; bodies are the largest grammar change with no named user.

Evidence: `extensions.md` L17 to L24; `casestudies/trainium2.md:108`; CP3, CP10, CP11; D28; nki.isa.dma_copy page (indirect `vector_offset`, dynamic strides).
Confidence: medium.
Changes my mind: an NVDLA or Ascend study run through the front end showing conv as `dot` loses a mode (dilation, grouped channels) that `pad` and `slice` cannot spell.

### Q2.5

Position: Options C and D; no new state kind in v3, descriptors and indexed state go to one data-dependent addressing study joined with L23, async groups stay plain integer state.

1. A TMA descriptor is a 64-byte-aligned object in memory built by `cuTensorMapEncodeTiled` and named by address in `cp.async.bulk.tensor` (PTX section 5.5.8 tensor-map), and Trainium DMA descriptors are generated into HBM or by DGE hardware at run time (nki.isa.dma_copy `dge_mode`); neither is register state, so a typed state kind mis-models both.
2. The real need is a start read from data (`gaps.md:16`), which is exactly L23's construct; two kinds for one need (CP6) is the wrong shape.
3. `wgmma.commit_group` and `wait_group` are a counter of pending groups, which `@s.pending` as an integer with an update and a constraint already states; a new kind adds nothing functional.
4. Indexed state has real sites (AMX `rows[t]`, `mbarrier` objects, Trainium semaphores, CP5), but each is today a per-register instruction or a flat variable, and the backend rejects every state read anyway (`act-backend/act_backend/spec_checks.py:372-384`), so v3 gains nothing executable from it.
5. Owner 21 keeps `@s.`, so `@s.rows[@c.t]` is a natural later extension once a study fixes how an indexed read interacts with static allocation.

Evidence: R14; audit R14, RA9; CP2, CP5, CP6, CP13; PTX ISA table of contents (tensor-map, mbarrier, wgmma async proxy sections); CUDA driver API tensor-map page.
Confidence: medium-high.
Changes my mind: a backend plan that models state in v3, which would make indexed state worth its grammar cost now.

### Q2.6

Position: Option B; out of v3, with the spec stating that program order is the only order and that a constraint used as a wait faults when its condition is false.

1. Every studied ISA synchronizes explicitly: `mbarrier.try_wait`, `wgmma.wait_group`, AIE-ML locks, Trainium semaphores (`casestudies/hopper.md:41`, `aie_ml.md:33`, `trainium2.md:115`); a correctly synchronized program has the functional meaning of one sequential interleaving, which is what TAIDL gives (`overview.md:10`).
2. A wait whose condition never becomes true is a deadlock on hardware; a fault at issue is the honest defined behavior under owner 16, and the spec should say so (CP14).
3. A core grid as an extra addressed dimension (L24) describes Tensix and AIE-ML storage, but neighbor-tile memory (`aie_ml.md:107`) needs instances sharing a buffer, which is a design study, not a v3 construct without syntax from any source.
4. Spatial-mapping machines (Groq, Cerebras) are describable at their per-PE ISA level and outside at their mapping level; "permanently out" (Option D) overstates it.

Evidence: R13; L24; CP7, CP14; `gaps.md:17`; PTX ISA sections on mbarrier and wgmma async proxy operations.
Confidence: high.
Changes my mind: a second generator (oracle) that must model queue overlap to match hardware results on a real workload.

### Q2.7

Position: Options C, D, B, and E together; fix the three verdicts and recount, count only sketches that passed the front end, add NVDLA instead of Ascend, and add the GPU tensor-core description.

1. The three wrong verdicts all inflate the matrix (CP2, CP3, CP12), and `LDTILECFG` is confirmed one instruction by the Intel SDM page (bytes 16-31 colsb, 48-55 rows, byte 0 palette).
2. Shape bugs (CP15) and `method.md` ("none was run") mean no sketch is evidence yet; R16's front-end run is the minimum.
3. The descriptor-driven engine class has no study (CP7); NVDLA has a public hardware manual with register-level documentation, while Ascend's instruction-level ISA is not public and its site was unreachable (`casestudies/method.md`), so NVDLA is the better second site.
4. Owner 15 places tensor-core instructions in scope and a real `wgmma` and TMA description, checked against the PTX fragment layouts, is the one GPU evidence base the plan lacks (R47, CP13).
5. SME is a CPU extension (D28) but its public reference stays a good check of mode state; SMSTART and SMSTOP zero Z and P when PSTATE.SM changes (Arm SME introduction, developer.arm.com), confirming CP12's fix direction.

Evidence: CP1 to CP3, CP7, CP9, CP12, CP15; R15, R16, R47; audit R15; `casestudies/method.md:5-10`.
Confidence: medium-high.
Changes my mind: an Ascend document at the Cube and Vector instruction level that is public and stable.

### Q2.8

Position: Option A; remove all three and reintroduce at the second real ISA description that needs them.

1. No tensor accelerator in the ten studies or in the public ISAs I know (TPU, Trainium, Hopper, AMX, SME, Gaudi MME, NVDLA, Gemmini) has a Cholesky or triangular-solve instruction; these are library routines built from `dot`.
2. Batch-norm inference on hardware is a scale-and-shift on a vector unit (NKI `activation` takes bias and scale), which `multiply` and `add` state exactly.
3. The three words alone carry the `BOOL`, `TRANSPOSES`, and `NUMBER` forms (`taidl/operations.py:18-23`, `:145-156`), so removing them removes three attribute grammars from every generator.

Evidence: R60; `act-oracle/docs/coverage.md`; nki.isa pages for `nc_matmul` and `activation`.
Confidence: high.
Changes my mind: a shipped accelerator instruction that factors or solves a triangular system, cited from its ISA document.

### Q2.9

Position: remove presumed reshape; make `d0` an ordinary buffer with one addressed dimension and an issue-time size; keep `constraints=` only; defer `ROOT`; adopt R11 for instructions without semantics.

1. A hardware operand has one bit-level layout, so an explicit `reshape` from the region shape is what the datasheet says; an implied reshape hides a layout decision the writer must check.
2. `hbm` is a byte array of unknown static size on every accelerator; `d0` as a `u8` buffer with one access dimension (`declarations.md:29-30`) and RA10's issue-time size is the model, and the `instruction.py:123` special case goes.
3. Configuration instructions with no data flow are the norm (`LDTILECFG`, Gemmini `config_*`, `SMSTART`), so R11's state-only instruction is required, and regions without semantics have no hardware meaning and are rejected.
4. `ROOT` costs nothing a hardware writer notices and keeps HLO pasteable; defer.

Evidence: R9, R11; `taidl/instruction.py:99-100`, `:123`; `casestudies/amx.md`, `gemmini.md:21-27`, `sme.md:19-20`; Intel SDM LDTILECFG page.
Confidence: medium-high.
Changes my mind: a backend cost for an issue-time `d0` size that a declared size would avoid, with a cited allocation path.

### Q2.10

Position: Option A for the grammar and Option E for reserved names; link the `.g4` files, keep token tables, and generate the reserved-name table from the generators' lists.

1. ISA documents keep encoding tables generated from the source of truth; a copied grammar drifted once (S3.2 lost `compare`, `taidl/antlr4/IDLV2Expr.g4:25`), and a link cannot drift.
2. A writer meets the reserved-name error at the first instruction named `copy` or `dot`, before any oracle doc (D17), so the list must be in the spec but never hand-copied.

Evidence: R61; D9, D17; ambiguity row on the grammar page.
Confidence: medium.
Changes my mind: a CI diff test already shipped for the copied rules, which would make Option B as safe as A.

### Q2.11

Position: the execution page (issue model) stays normative in the spec; the object model stays as a contract page limited to the attributes a generator must read (Option C's list), with the rest in the `taidl` README.

1. The issue order (updates, starts, constraints, reads, writes) is what a hardware writer checks against the pipeline description, so S5.6 is language, not API.
2. A generator must read a fixed set of trees and regions (`taidl/instruction.py:131-140`, `:249-252`); naming only those keeps the contract honest without freezing internals.
3. No non-Python generator exists (R42), so a serialized format (Option D) is speculative generality.

Evidence: R42; implementer Q1 to Q10; `object_model.md:1-30`; minimalist row S6.
Confidence: medium-low.
Changes my mind: a second generator in another language, which would make Option D the contract.

### Q2.12

Position: Option E with "ISA description" (R1), and Option B for renames; Python attributes now, the grammar rule `instruction` later.

1. The TAIDL paper calls the user's artifact an ISA description, and every architecture document calls its instruction set an ISA; "spec" stays the language document.
2. Attribute renames (`parameters`, `comp_attr`, `var_type`) land in the owner 10 batch with the B and O visitors; the grammar rule rename needs regenerated parsers (D13) and buys a writer nothing visible.
3. The glossary file must drop `add_tensor_buffer` and `add_control_register`, which owner 19 forbids (audit R2).

Evidence: R1, R2; audit R2; D13; `language/plan/terminology.md`; TAIDL paper, MICRO 2025 (act-compiler.github.io).
Confidence: medium.
Changes my mind: a parser regeneration already scheduled in the batch, which would admit the grammar rename at no extra cost.
