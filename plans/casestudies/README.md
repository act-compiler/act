# Accelerator case studies

What TAIDL can describe today: one sketch per accelerator, a gap matrix, and plan items.
Only public ISAs count as sites; Eyeriss and CGRA are invented and count zero.
The Hopper sketch is the GPU tensor-core description; NVDLA is the queued study.

| file | contents |
| --- | --- |
| [method.md](method.md) | How the studies were made, what counts as a site, and their limits |
| [tpu.md](tpu.md) | Google TPU: MXU, VPU, VMEM |
| [trainium2.md](trainium2.md) | AWS Trainium2 NeuronCore: Tensor, Vector, Scalar, GpSimd engines, SBUF, PSUM |
| [hopper.md](hopper.md) | NVIDIA Hopper tensor-core instructions: `wgmma`, TMA, `mbarrier` |
| [amx.md](amx.md) | Intel AMX: tile registers and tile configuration |
| [sme.md](sme.md) | Arm SME: ZA array, outer products, streaming mode |
| [aie_ml.md](aie_ml.md) | AMD XDNA / AIE-ML: vector core, data memory, DMA, locks |
| [gemmini.md](gemmini.md) | Gemmini: scratchpad, accumulator, configuration instructions |
| [tensix.md](tensix.md) | Tenstorrent Tensix: unpack, math, pack, `Dst` |
| [eyeriss.md](eyeriss.md) | Eyeriss-style row-stationary array |
| [cgra.md](cgra.md) | Generic CGRA with a configuration bitstream |
| [gaps.md](gaps.md) | Gap matrix: missing constructs per site count, with the v3 decision |
| [strengths.md](strengths.md) | What TAIDL already expresses well |
| [plan/README.md](plan/README.md) | Plan items C1 to C13; items that duplicate an L item point to it |
