# Integrating the new Accelerator Backend with XLA Compiler

## From JAX to Accelerator Execution

The complete compilation pipeline connects high-level ML frameworks to custom accelerators:

```
┌─────────────┐
│  JAX Code   │  High-level: ML Framework
└──────┬──────┘
       │ jax.jit (XLA compilation)
       ▼
┌─────────────┐
│  XLA-HLO    │  Mid-level: Tensor operations IR
└──────┬──────┘
       │ ACT Compiler (Session 3)
       ▼
┌─────────────┐
│ QKV Assembly│  Low-level: Executable instructions
└──────┬──────┘
       │ TAIDL-TO Simulator (Session 2)
       ▼
┌─────────────┐
│  Execution  │  Output: Results verified against reference
└─────────────┘
```

---

## Compiling JAX Programs Using XLA

When you annotate JAX code with `jax.jit`, JAX traces Python tensor operations and hands them to XLA for compilation.

At a high level:

1. JAX traces Python function execution into an HLO-compatible graph.
2. XLA applies graph-level optimizations (for example, simplification and layout decisions).
3. XLA lowers optimized computation into tiled kernels.
4. The accelerator-specific backend is invoked on these tiled kernels.
5. Emitted accelerator assembly code is stitched into the surrounding generated program and executed.

In this tutorial, Step 4 is handled by the ACT-generated QKV backend.

---

## Where the ACT Backend Is Invoked

The key boundary is:

- **Before backend invocation**: XLA-level tensor computation (HLO/tiled kernels)
- **At backend invocation**: ACT backend consumes a tiled kernel and emits QKV assembly
- **After backend invocation**: generated assembly is embedded into the final executable flow

This is exactly why Exercise 3 uses HLO as input and emits `asm/compiled_qkv.py`: it mirrors the same compiler boundary used in framework integration.

Think of the flow as two levels of lowering:

- **Framework lowering (JAX -> XLA-HLO)**: ML workload becomes tensor IR
- **Accelerator lowering (XLA-HLO -> QKV assembly)**: tensor IR becomes accelerator instructions

ACT automates the second lowering for new AI accelerators from the TAIDL ISA definition.
