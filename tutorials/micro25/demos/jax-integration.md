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
