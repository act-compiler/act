# ACT Tutorial Walkthrough

## The Big Picture: From ISA to End-to-End Compilation

ACT automates the entire compiler development process for AI accelerators:

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
           │ generate_backend()
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

## Tutorial Goals: What You Will See in Each Stage

### Stage 1: ISA Specification (Exercise 1)

You define:

- Data models (`d0`, `d1`, `d2`) and their capacities
- Instruction attributes (`alpha` for compute, `beta` for addressing)
- Instruction semantics using XLA-HLO operators

You run:

```bash
python QKV.py
```

This triggers generation of:

- ISA-specific kernel API + test oracle (`targets/QKV/oracle/`)
- ISA-specific compiler backend (`targets/QKV/backend/` and `backends/QKV`)

### Stage 2: Kernel Programming (Exercise 2)

You write AI accelerator kernels against the generated API:

- `load_rm`, `load_cm`, `gemm`, `softmax`, `mov`, `store_rm`
- Explicit scratchpad address management
- Functional validation against FPGA golden outputs

This stage builds intuition for the generated assembly that you will later get automatically.

### Stage 3: Compiler Backend Generation (Exercise 3)

You provide high-level HLO and compile it using the generated backend:

```bash
./backends/QKV --input attention.hlo --output asm/compiled_qkv.py
```

The backend performs:

- Instruction selection (find ISA-equivalent implementations)
- Memory allocation (solve addressing attributes)
- Code emission (Python assembly kernel)

### Stage 4: Framework Integration (Demonstration 2)

The same backend can be wired into an XLA flow so that JAX programs are lowered into QKV assembly through the existing compilation stack.

### Stage 5: Iteration and Tweaking (Exercise 4)

You modify the ISA specification (for example, add a new instruction or change an attribute) and see how the changes propagate through the entire software stack without manual intervention.

---

## TAIDL (Tensor Accelerator ISA Definition Language)

**TAIDL** is a Python-based DSL for formally specifying AI accelerator ISAs. Unlike traditional ISA specification languages that focus on scalar operations, TAIDL models coarse-grained tensor operations using XLA-HLO semantics.

In today's tutorial, we will be writing a new AI accelerator ISA from scratch and observing how the rest of the tooling is automatically generated.

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
