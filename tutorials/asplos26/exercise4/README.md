# Hands-on Exercise 4: Tweaking the ISA and Open Discussion

We implemented the ISA specification for the QKV accelerator in Exercise 1, and then manually programmed an attention kernel in Exercise 2. In Exercise 3, we will automatically generate a complete compiler backend from your ISA specification, and compile the same attention HLO to assembly.

In this exercise, we will modify the ISA to explore how architectural changes propagate through the full ACT flow.

## Goal of This Exercise

Starting from `QKV.py`, create a new ISA variant (`QKV_new.py`) with the following changes:

1. Add a new on-chip buffer `d3`, a duplicate of `d1`.
2. Remove column-major load/store instructions, and replace them with row-major variants specialized by destination/source buffer:
	- `load_01` (HBM -> `d1`), `load_03` (HBM -> `d3`)
	- `store_10` (`d1` -> HBM), `store_30` (`d3` -> HBM)
3. Add a transpose-copy instruction `transpose_13` from `d1` to `d3`.
4. Split GEMM into two ISA instructions that both write to `d2`:
	- `gemm_33`: reads both operands from `d3`
	- `gemm_13`: reads operands from `d1` and `d3`
5. Add two move instructions from `d2`:
	- `mov_21`: `d2` -> `d1`
	- `mov_23`: `d2` -> `d3`

These changes let us model a richer memory topology while keeping the same high-level attention computation.

---

## Step 1: Update the Data Model

Add `d3` as a duplicate of `d1`:

```python
qkv.add_data_model("d1", [128], [64], "bf16")
qkv.add_data_model("d3", [128], [64], "bf16")
qkv.add_data_model("d2", [64], [64], "bf16")
```

Interpretation:

- `d1` and `d3` are dual primary scratchpads.
- `d2` remains the compute/output staging buffer.

---

## Step 2: Replace CM Load/Store with Buffer-Specific RM Variants

Instead of one row-major and one column-major path, define just row-major instructions per buffer pair:

- `load_01`, `load_03`
- `store_10`, `store_30`

---

## Step 3: Add Transpose-Copy Path d1 -> d3

Introduce `transpose_13`, an instruction that takes tiles in `d1` and materializes a transposed layout in `d3`.

Why this matters:

- It replaces layout conversion previously hidden in CM loads.
- It exposes layout transforms as first-class ISA operations.
- It gives the backend additional legal lowering paths

You'll see many equivalent ways to implement attention with the new ISA, and the backend will be able to explore them all - stay tuned!

---

## Step 4: Split GEMM by Operand Buffers

Define two GEMM opcodes, both writing to `d2`:

- `gemm_33(addr_1, addr_2, addr_out)`: `d3 x d3 -> d2`
- `gemm_13(addr_1, addr_2, addr_out)`: `d1 x d3 -> d2`

This captures buffer-aware execution choices directly in the ISA.

---

## Step 5: Add another Move Instructions from d2 to d3

Define:

- `mov_21(n, addr_in, addr_out)`: `d2 -> d1` [same as before but renamed for consistency]
- `mov_23(n, addr_in, addr_out)`: `d2 -> d3`

These enable flexible write-back routing after GEMM/softmax.

---

## Expected Instruction Set (QKV_new)

- Loads: `load_01`, `load_03`
- Stores: `store_10`, `store_30`
- Moves: `mov_21`, `mov_23`
- Compute: `gemm_13`, `gemm_33`, `softmax`
- Layout transform: `transpose_13`

---

## Regenerate and Test

After updating `QKV_new.py`, regenerate tooling and validate kernels:

```bash
python QKV_new.py
```

Then verify:

- Oracle generation succeeds for the new instruction set.
- Backend generation succeeds with new buffer/instruction combinations.
- Existing attention workflow can be re-lowered with updated instruction choices.

This exercise demonstrates how ISA-level memory and instruction design choices directly affect generated APIs, generated backend logic, and final assembly structure.
