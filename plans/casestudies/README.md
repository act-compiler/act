# Accelerator case studies

Hard evidence on what TAIDL can describe today: one sketch per accelerator, a gap matrix, and plan items.

| file | contents |
| --- | --- |
| [method.md](method.md) | How the studies were made and their limits |
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
| [gaps.md](gaps.md) | Gap matrix: missing constructs ranked by severity |
| [strengths.md](strengths.md) | What TAIDL already expresses well |
| [plan/README.md](plan/README.md) | Plan items C1 to C13 with their index table |
