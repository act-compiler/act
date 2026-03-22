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

- Data models and their capacities
- Instruction mnemonics and attributes (equivalent to a function signature)
- Instruction semantics using XLA-HLO operators (equivalent to a function body)

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

## Next Step: Exercise 1

In Exercise 1, you will specify the ISA for a simple QKV attention accelerator using TAIDL.

Proceed to the [Exercise 1 README](../exercise1/README.md) to get started!
