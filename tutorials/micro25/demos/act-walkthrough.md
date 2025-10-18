# ACT Tutorial Walkthrough

## The Big Picture: From ISA to End-to-End Compilation

ACT automates the entire compiler development process for tensor accelerators:

```
┌──────────────────────┐
│ 1. ISA Specification │  Python-based TAIDL DSL
│    (Exercise 1)      │  Define: memory, instructions, semantics
└──────────┬───────────┘
           │ generate_oracle()
           ▼
┌──────────────────────┐
│ 2. Kernel Programming│  Generated Python API
│    (Exercise 2)      │  Write: low-level kernels, test correctness
└──────────┬───────────┘
           │ generate_act()
           ▼
┌──────────────────────┐
│ 3. Compiler Backend  │  Automated backend generation
│    (Exercise 3)      │  Compile: HLO → Assembly
└──────────┬───────────┘
           │ JAX/XLA Integration
           ▼
┌──────────────────────┐
│ 4. ML Framework      │  End-to-end compilation
│    (Demo 2)          │  JAX → HLO → Assembly → Execution
└──────────────────────┘
```

**Key Innovation**: Everything after step 1 is **automatically generated** from the ISA specification.

---

## TAIDL (Tensor Accelerator ISA Definition Language)

**TAIDL** is a Python-based DSL for formally specifying tensor accelerator ISAs. Unlike traditional ISA specification languages that focus on scalar operations, TAIDL models coarse-grained tensor operations using XLA-HLO semantics.

In today's tutorial, we will be writing a new tensor accelerator ISA from scratch and observing how the rest of the tooling is automatically generated.

### Accelerator ISAs specified in TAIDL

Currently, ACT supports several popular academic and commercial accelerator designs. We are working on supporting more accelerator designs and are open to collaborations with accelerator architects.

| Accelerator         | Class           | Feature                                 | ISA Specification               |
| ------------------- | --------------- | --------------------------------------- | ------------------------------- |
| Intel AMX           | Commercial      | Tile-based register files               | Intel Instrinsics Guide         |
| NVIDIA Tensor Cores | Commercial      | Warp-level data fragments               | PTX Documentation               |
| AWS Trainium        | Commercial      | Banked/Partitioned scratchpads          | AWS NKI Documentation           |
| Google TPUv1        | Commercial      | Systolic array + Weights FIFO Buffer    | ISCA'17 White paper             |
| Gemmini             | Academic design | Systolic array                          | GitHub Documentation            |
| FEATHER             | Academic design | Reconfigurable and flexible data layout | Collaboration with Architects\* |
| EVA                 | Academic design | CGRA-based design + Evolvable ISA       | Collaboration with Architects\* |

\* = Collaborative effort at ACE, one of the seven centers in JUMP 2.0, a Semiconductor Research Corporation (SRC) program

Let me show you a quick demonstration of how the infrastructure works for one of the popular accelerator designs, Gemmini.
