# Hands-on Exercise 2: Writing Accelerator Kernels

In this hands-on exercise, you will learn to write accelerator kernels using the QKV ISA and programming APIs generated in Exercise 1. We provide three complete example kernels demonstrating different programming patterns. **Your task is to implement the final QKV attention kernel**, bringing together all the concepts.

You will test your implementation against **real data from an FPGA implementation** of the QKV accelerator (designed using Allo, an accelerator design language). This validates that your hand-written kernel produces hardware-correct results.

---

## Understanding the Kernel Programming Model

### What is a Kernel?

An **accelerator kernel** is a program written using the accelerator's ISA instructions. Similar to how you write assembly programs for CPUs, you write kernels for tensor accelerators -- but at a much coarser granularity (entire tensor operations rather than scalar operations).

### Kernel Structure

Kernels in TAIDL-TO API use Python decorators to specify memory addresses and instruction sequences:

```python
@kernel(hbm=<size>,
        input=[...],
        constant=[...],
        output=[...])
def kernel_name():
    api.load_rm(n=64, addr_in=0, addr_out=0)
    api.gemm(addr_1=0, addr_2=64, addr_out=0)
    # ... more instructions
```

**Key Components:**

1. **`@kernel` Decorator**: Defines the memory contract

   - `hbm`: Total off-chip memory size in bytes
   - `input`: Input tensors with HBM addresses, shapes, and dtypes
   - `constant`: Constant tensors (e.g., weights, identity matrices) with HBM addresses, shapes, dtypes and constant values.
   - `output`: Output tensors with HBM addresses, shapes, and dtypes

2. **Instruction Sequence**: Calls to generated API functions

   - Each API function corresponds to an ISA instruction from Exercise 1
   - Arguments match the instruction attributes defined in Exercise 1

3. **Memory Management**: You explicitly control:

   - **Data movement**: When to load/store between HBM and scratchpads
   - **Scratchpad allocation**: Which rows of d1/d2 to use
   - **Address calculation**: Ensuring no overwrites or conflicts

### Memory Address Conventions

Recall the QKV memory hierarchy:

- **HBM (d0)**: Off-chip memory

  - Addressed in **bytes**
  - Input/output tensors reside here
  - Example: `addr=0` means byte offset 0

- **Scratchpad d1**: Primary on-chip buffer

  - 128 rows × 64 columns of BF16
  - Addressed by row number (0-127)
  - Used for input/output data staging

- **Scratchpad d2**: Computation buffer
  - 64 rows × 64 columns of BF16
  - Addressed by row number (0-63)
  - GEMM and softmax outputs go here

**Important**: Each scratchpad row holds 64 BF16 elements = 128 bytes

---

## Step 1: Setting Up Your Development Environment

Let's begin by preparing the skeleton for the attention kernel.

From your host machine in the `tutorials/micro25/` directory, copy the boilerplate for Exercise 2.

```bash
./copy.sh exercise2
```

The `copy.sh` script will copy example kernels and the attention kernel skeleton to the `act/asm/` directory, along with FPGA test data.

### File Structure

Your `act/` directory should now contain:

```
act/
├── asm/
│   ├── identity.py        # Example: Simple load/store
│   ├── matmul.py          # Example: Matrix multiplication
│   ├── softmax.py         # Example: Softmax computation
│   └── attention.py       # TODO: Your task - implement QKV attention
├── data/
│   ├── Q.dat              # FPGA input: Query matrix
│   ├── K.dat              # FPGA input: Key matrix
│   ├── V.dat              # FPGA input: Value matrix
│   └── attention.dat      # FPGA golden output
└── test_qkv.py            # Test script for validation
```

**FPGA Data Context:**

The test data comes from a real hardware implementation of the QKV accelerator, designed using **Allo** (Accelerator Design Language from Cornell: https://github.com/cornell-zhang/allo). The Allo framework allows us to design accelerators and synthesize them to FPGAs. We ran the QKV attention computation on the FPGA and saved the inputs/outputs for this tutorial, since running FPGA synthesis at the tutorial is infeasible.

You can now edit the kernel files in your preferred editor on your host machine.

---

## Step 2: Example Kernel Walkthroughs

Before implementing the attention kernel, let's study three complete examples that demonstrate different programming patterns. **You don't need to write these—they're already complete in the boilerplate.**

### Example 1: Identity Kernel

**Purpose**: The simplest possible kernel -- just copy data from input to output.

**File**: `asm/identity.py` (already complete in boilerplate)

```python
import jax.numpy as jnp


def identity(kernel, api):
    @kernel(
        hbm=16384,  # 16 KB: 8 KB input + 8 KB output
        input=[
            {'addr': 0, 'shape': (64, 64), 'dtype': jnp.bfloat16},
        ],
        constant=[],
        output=[
            {'addr': 8192, 'shape': (64, 64), 'dtype': jnp.bfloat16},
        ]
    )
    def identity_():
        # Load 64 rows from HBM address 0 to scratchpad d1 address 0
        api.load_rm(n=64, addr_in=0, addr_out=0)

        # Store 64 rows from scratchpad d1 address 0 to HBM address 8192
        api.store_rm(n=64, addr_in=0, addr_out=8192)

    return identity_
```

**Walkthrough:**

1. **Memory Layout**:

   - Input: 64×64 BF16 = 8192 bytes at HBM address 0
   - Output: 64×64 BF16 = 8192 bytes at HBM address 8192
   - Output starts at 8192 to ensure no overlap with input

2. **Instruction Sequence**:

   - `load_rm(n=64, addr_in=0, addr_out=0)`: Load 64 rows from HBM[0:8192] to d1[0:63]
   - `store_rm(n=64, addr_in=0, addr_out=8192)`: Store 64 rows from d1[0:63] to HBM[8192:16384]

**Key Insight**: The identity kernel demonstrates the basic load-compute-store pattern, though "compute" is trivial here -- just data movement.

---

### Example 2: Matrix Multiplication Kernel

**Purpose**: Compute C = A × B using the hardware GEMM instruction.

**File**: `asm/matmul.py` (already complete in boilerplate)

```python
import jax.numpy as jnp


def matmul(kernel, api):
    @kernel(
        hbm=24576,  # 24 KB: 3 matrices × 8 KB
        input=[
            {'addr': 0, 'shape': (64, 64), 'dtype': jnp.bfloat16},      # Matrix A
            {'addr': 8192, 'shape': (64, 64), 'dtype': jnp.bfloat16},   # Matrix B
        ],
        constant=[],
        output=[
            {'addr': 16384, 'shape': (64, 64), 'dtype': jnp.bfloat16},  # Matrix C = A × B
        ]
    )
    def matmul_():
        # Load matrix A into d1[0:63]
        api.load_rm(n=64, addr_in=0, addr_out=0)

        # Load matrix B into d1[64:127]
        api.load_rm(n=64, addr_in=8192, addr_out=64)

        # Compute C = A × B, result goes to d2[0:63]
        api.gemm(addr_1=0, addr_2=64, addr_out=0)

        # Move result from d2[0:63] to d1[0:63]
        api.mov(n=64, addr_in=0, addr_out=0)

        # Store result back to HBM
        api.store_rm(n=64, addr_in=0, addr_out=16384)

    return matmul_
```

**Walkthrough:**

1. **Memory Layout**:

   - Input A: 8192 bytes at HBM[0:8191]
   - Input B: 8192 bytes at HBM[8192:16383]
   - Output C: 8192 bytes at HBM[16384:24575]

2. **Scratchpad Allocation**:

   - d1[0:63]: Holds matrix A
   - d1[64:127]: Holds matrix B
   - d2[0:63]: Holds result C (output of GEMM)

3. **Instruction Sequence**:

   - Load both inputs into different regions of d1
   - GEMM computes into d2 (as specified in ISA semantics)
   - Move result from d2 back to d1 (needed for store)
   - Store result back to HBM

4. **Key Insight**: `gemm` outputs to d2, but `store_rm` reads from d1. The `mov` instruction copies data between scratchpads.

---

### Example 3: Softmax Kernel

**Purpose**: Apply softmax normalization across each row of a matrix.

**File**: `asm/softmax.py` (already complete in boilerplate)

```python
import jax.numpy as jnp


def softmax(kernel, api):
    @kernel(
        hbm=24576,
        input=[
            {'addr': 0, 'shape': (64, 64), 'dtype': jnp.bfloat16},
        ],
        constant=[
            # Identity matrix I
            {'addr': 8192, 'shape': (64, 64), 'dtype': jnp.bfloat16,
             'value': jnp.eye(64, dtype=jnp.bfloat16)},
        ],
        output=[
            {'addr': 16384, 'shape': (64, 64), 'dtype': jnp.bfloat16},
        ]
    )
    def softmax_():
        # Load input matrix A
        api.load_rm(n=64, addr_in=0, addr_out=0)

        # Load identity matrix I (constant)
        api.load_cm(n=64, addr_in=8192, addr_out=64)

        # Compute A × I = A
        api.gemm(addr_1=0, addr_2=64, addr_out=0)

        # Apply softmax to A (in-place in d2)
        api.softmax(n=64, addr=0)

        # Move softmax(A) from d2 to d1
        api.mov(n=64, addr_in=0, addr_out=0)

        # Store softmax(A) from d1 to HBM
        api.store_rm(n=64, addr_in=0, addr_out=16384)

    return softmax_
```

**Walkthrough:**

1. **Hardware Constraint**: Softmax only operates on data in d2, but data loads into d1. Therefore, we use perform a "dummy" GEMM using an identity matrix to move data from d1 to d2.

2. **Constant Tensors**:

   - Defined in `@kernel` decorator with `'value'` field
   - Automatically loaded into HBM before kernel execution
   - Here: Identity matrix I at HBM address 8192

3. **Why load_cm for constants?**  
   Using `load_cm` (column-major) for the identity matrix demonstrates flexible data layouts. Though I = I^T, this pattern is crucial for loading K^T in attention.

4. **In-Place Softmax**:

   - `softmax(n=64, addr=0)`: Operates on d2[0:63], overwrites input with output
   - Recall from Exercise 1: softmax computes `exp(x) / sum(exp(x))` row-wise

**Key Insight**: Constants (weights, biases, matrices) are declared separately and automatically managed by the runtime.

---

## Step 3: Your Task - Implement QKV Attention Kernel

Now it's your turn! You'll implement the complete attention mechanism:

```
Attention(Q, K, V) = softmax(Q × K^T) × V
```

This is the most complex kernel in this tutorial. We'll build it incrementally, starting from the computation breakdown, then planning memory layout, and finally implementing each stage step-by-step.

### 3.1: Understanding the Computation

**Attention consists of two matrix multiplications with softmax in between:**

1. **Score Computation**: `S = Q × K^T` (64×64 @ 64×64 → 64×64)
2. **Normalization**: `P = softmax(S)` (row-wise)
3. **Weighted Sum**: `O = P × V` (64×64 @ 64×64 → 64×64)

**Mathematical Breakdown:**

```
Given: Q, K, V ∈ ℝ^(64×64)

Step 1: S = Q × K^T
        For each row i: S[i,:] = Σ_k Q[i,k] * K[:,k]

Step 2: P = softmax(S)
        For each row i: P[i,j] = exp(S[i,j]) / Σ_j exp(S[i,j])

Step 3: O = P × V
        For each row i: O[i,:] = Σ_j P[i,j] * V[j,:]
```

**Why This Pattern?** This is the canonical scaled dot-product attention from "Attention is All You Need" (Vaswani et al., 2017), fundamental to transformers.

---

### 3.2: Planning Memory Layout

Before writing any code, we need a clear memory allocation strategy.

**HBM Layout (Off-Chip):**

| Address Range | Content | Size  | Description               |
| ------------- | ------- | ----- | ------------------------- |
| 0 - 8191      | Q       | 8192B | Query matrix (64×64 BF16) |
| 8192 - 16383  | K       | 8192B | Key matrix (64×64 BF16)   |
| 16384 - 24575 | V       | 8192B | Value matrix (64×64 BF16) |
| 24576 - 32767 | Output  | 8192B | Result O (64×64 BF16)     |

**Scratchpad Layout (On-Chip):**

| Buffer | Rows   | Stage 1 (Q×K^T)  | Stage 2 (P×V)     |
| ------ | ------ | ---------------- | ----------------- |
| d1     | 0-63   | Q matrix         | P matrix (reused) |
| d1     | 64-127 | K^T matrix       | V matrix (reused) |
| d2     | 0-63   | S → P (in-place) | O result          |

**Key Design Decisions:**

1. **Why `load_cm` for K?** Loads K as K^T directly, avoiding explicit transpose
2. **Why reuse d1[64:127]?** After computing P, we no longer need K^T; reuse that space for V
3. **Why `mov` between stages?** GEMM reads from d1, writes to d2; we need to move data back

---

### 3.3: Setting Up the Kernel Decorator

Let's start by defining the kernel structure with proper memory declarations.

**File**: `asm/attention.py` (Part 1: Decorator setup)

```python
import jax.numpy as jnp


def qkv(kernel, api):
    @kernel(
        hbm=32768,  # 32 KB: enough for 3 inputs + 1 output
        input=[
            {'addr': 0, 'shape': (64, 64), 'dtype': jnp.bfloat16},      # Q
            {'addr': 8192, 'shape': (64, 64), 'dtype': jnp.bfloat16},   # K
            {'addr': 16384, 'shape': (64, 64), 'dtype': jnp.bfloat16},  # V
        ],
        constant=[],  # No constants needed
        output=[
            {'addr': 24576, 'shape': (64, 64), 'dtype': jnp.bfloat16},  # O
        ]
    )
    def qkv_():
        # Kernel implementation goes here
        pass  # We'll fill this in progressively

        # ===== STAGE 1: Compute Attention Scores (Q × K^T) =====

        # ===== STAGE 2: Normalize Scores (softmax) =====

        # ===== STAGE 3: Prepare for Second MatMul =====

        # ===== STAGE 4: Compute Final Output (P × V) =====

        # ===== STAGE 5: Store Result =====

    return qkv_
```

**What We've Done:**

- Declared 3 input matrices at non-overlapping HBM addresses
- Allocated output space starting at 24576 (3 × 8192)
- Set HBM size to 32768 (4 × 8192) to accommodate all data

---

### 3.4: Stage 1 - Computing Attention Scores (Q × K^T)

Now let's implement the first computation stage. This is similar to the matmul kernel, but with a transpose.

**Recall from MatMul Example:**

```python
# Standard matmul pattern (from Step 2, Example 2)
api.load_rm(n=64, addr_in=0, addr_out=0)        # Load A
api.load_rm(n=64, addr_in=8192, addr_out=64)    # Load B
api.gemm(addr_1=0, addr_2=64, addr_out=0)       # A × B
```

**For Attention Scores (Q × K^T):**

```python
# Load Q in row-major (standard)
api.load_rm(n=64, addr_in=0, addr_out=0)

# Load K in column-major (gives us K^T automatically!)
api.load_cm(n=64, addr_in=8192, addr_out=64)

# Compute S = Q × K^T
api.gemm(addr_1=0, addr_2=64, addr_out=0)
```

**Key Difference:** `load_cm` instead of `load_rm` for K. Recall from Session 1 that `load_cm` includes a transpose operation.

**Current State After Stage 1:**

- d1[0:63]: Q matrix (input for GEMM)
- d1[64:127]: K^T matrix (input for GEMM)
- d2[0:63]: S = Q × K^T (attention scores)

---

### 3.5: Stage 2 - Normalizing Scores (softmax)

After computing attention scores S, we need to normalize them row-wise using softmax.

**Recall from Softmax Example:**

```python
# From Step 2, Example 3 (simplified)
api.softmax(n=64, addr=0)  # Operates in-place on d2[0:63]
```

**For Our QKV Kernel:**

```python
# Apply softmax to S, converting it to P (in-place in d2)
api.softmax(n=64, addr=0)
```

**In-Place Operation:** The softmax instruction overwrites S with P in d2[0:63]. No additional memory needed!

**Current State After Stage 2:**

- d1[0:63]: Q matrix (no longer needed)
- d1[64:127]: K^T matrix (no longer needed)
- d2[0:63]: P = softmax(S) (attention probabilities)

**Important Observation:** We can now reuse d1 for the second matmul stage!

---

### 3.6: Stage 3 - Preparing for Second MatMul (P × V)

Before computing P × V, we need to:

1. Move P from d2 to d1 (GEMM reads inputs from d1)
2. Load V into d1 (reusing the space where K^T was)

**Data Movement:**

```python
# Move P from d2[0:63] to d1[0:63]
# Recall: mov copies between scratchpads
api.mov(n=64, addr_in=0, addr_out=0)
```

**Loading V:**

```python
# Load V into d1[64:127] (reusing K^T's space)
api.load_rm(n=64, addr_in=16384, addr_out=64)
```

**Current State After Preparation:**

- d1[0:63]: P matrix (first input for GEMM)
- d1[64:127]: V matrix (second input for GEMM)
- d2[0:63]: Old P data (will be overwritten)

**Memory Efficiency:** We've reused scratchpad space without needing additional buffers!

---

### 3.7: Stage 4 - Computing Final Output (P × V)

Now we're ready for the second matrix multiplication, which produces the final attention output.

**Recall the MatMul Pattern:**

```python
# From Stage 1
api.gemm(addr_1=0, addr_2=64, addr_out=0)
```

**For Final Output:**

```python
# Compute O = P × V, result in d2[0:63]
api.gemm(addr_1=0, addr_2=64, addr_out=0)
```

**Current State After Computation:**

- d1[0:63]: P matrix (no longer needed)
- d1[64:127]: V matrix (no longer needed)
- d2[0:63]: O = P × V (final output!)

---

### 3.8: Stage 5 - Storing the Result

Finally, we need to move the result from d2 back to d1, then store it to HBM.

**Recall from Identity Example:**

```python
# Store pattern from Step 2, Example 1
api.store_rm(n=64, addr_in=0, addr_out=8192)
```

**For Our Output:**

```python
# Move O from d2[0:63] to d1[0:63]
api.mov(n=64, addr_in=0, addr_out=0)

# Store O to HBM at address 24576
api.store_rm(n=64, addr_in=0, addr_out=24576)
```

**Why Move Before Store?** The `store_rm` instruction reads from d1, not d2. We must move the data first.

---

### 3.9: Complete Implementation

Now let's put it all together! Here's the complete QKV attention kernel with all stages integrated.

**File**: `asm/attention.py` (Complete)

```python
import jax.numpy as jnp


def qkv(kernel, api):
    @kernel(
        hbm=32768,  # 32 KB: enough for 3 inputs + 1 output
        input=[
            {'addr': 0, 'shape': (64, 64), 'dtype': jnp.bfloat16},      # Q
            {'addr': 8192, 'shape': (64, 64), 'dtype': jnp.bfloat16},   # K
            {'addr': 16384, 'shape': (64, 64), 'dtype': jnp.bfloat16},  # V
        ],
        constant=[],  # No constants needed
        output=[
            # O = softmax(Q × K^T) × V
            {'addr': 24576, 'shape': (64, 64), 'dtype': jnp.bfloat16},
        ]
    )
    def qkv_():
        # Kernel implementation goes here

        # ===== STAGE 1: Compute Attention Scores (Q × K^T) =====
        # Load Q in row-major (standard)
        api.load_rm(n=64, addr_in=0, addr_out=0)

        # Load K in column-major (gives us K^T automatically!)
        api.load_cm(n=64, addr_in=8192, addr_out=64)

        # Compute S = Q × K^T
        api.gemm(addr_1=0, addr_2=64, addr_out=0)

        # ===== STAGE 2: Normalize Scores (softmax) =====
        # Apply softmax to S, converting it to P (in-place in d2)
        api.softmax(n=64, addr=0)

        # ===== STAGE 3: Prepare for Second MatMul =====
        # Move P from d2[0:63] to d1[0:63]
        # Recall: mov copies between scratchpads
        api.mov(n=64, addr_in=0, addr_out=0)

        # Load V into d1[64:127] (reusing K^T's space)
        api.load_rm(n=64, addr_in=16384, addr_out=64)

        # ===== STAGE 4: Compute Final Output (P × V) =====
        # Compute O = P × V, result in d2[0:63]
        api.gemm(addr_1=0, addr_2=64, addr_out=0)

        # ===== STAGE 5: Store Result =====
        # Move O from d2[0:63] to d1[0:63]
        api.mov(n=64, addr_in=0, addr_out=0)

        # Store O to HBM at address 24576
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    return qkv_
```

---

### 3.10: Detailed Walkthrough with Data Flow

Let's trace the complete data flow through the kernel:

**Stage 1 (Lines 21-29): Score Computation**

- **Input State**: Q, K, V in HBM; scratchpads empty
- **After load_rm**: d1[0:63] ← Q
- **After load_cm**: d1[64:127] ← K^T (transposed during load)
- **After gemm**: d2[0:63] ← Q × K^T = S
- **Key Insight**: K^T loaded directly, no explicit transpose needed

**Stage 2 (Lines 31-33): Normalization**

- **Input State**: d2[0:63] contains S
- **After softmax**: d2[0:63] ← softmax(S) = P (in-place)
- **Key Insight**: In-place operation saves memory

**Stage 3 (Lines 35-41): Preparation**

- **Input State**: d2[0:63] contains P; d1 contains old Q and K^T
- **After mov**: d1[0:63] ← P (copied from d2)
- **After load_rm**: d1[64:127] ← V (overwrites K^T)
- **Key Insight**: Scratchpad reuse -- K^T no longer needed

**Stage 4 (Lines 43-45): Final Computation**

- **Input State**: d1[0:63] contains P; d1[64:127] contains V
- **After gemm**: d2[0:63] ← P × V = O
- **Key Insight**: Same GEMM pattern as Stage 1

**Stage 5 (Lines 47-52): Output**

- **Input State**: d2[0:63] contains O
- **After mov**: d1[0:63] ← O (copied from d2)
- **After store_rm**: HBM[24576:32767] ← O
- **Final State**: Result written to HBM

---

### 3.11: Design Insights and Optimization

**Why This Implementation is Efficient:**

1. **Memory Reuse**: Only uses d1 and d2, no additional buffers

   - d1[0:63] reused: Q → P → O
   - d1[64:127] reused: K^T → V

2. **Minimal Data Movement**: Only necessary `mov` operations

   - Move P: Required because GEMM reads from d1
   - Move O: Required because store reads from d1

3. **In-Place Operations**: Softmax overwrites input

   - Saves memory bandwidth
   - Reduces scratchpad pressure

4. **Leveraging Hardware Features**:
   - `load_cm` provides free transpose
   - GEMM operates on fixed 64×64 tiles

---

### 3.12: Key Takeaways from QKV Implementation

1. **Resource Management**: Efficient scratchpad utilization

   - Temporal reuse (Q → P → O in same location)
   - Spatial reuse (K^T → V in same location)
   - In-place operations where possible

2. **Hardware-Software Co-Design**: Leveraging ISA features

   - Transposing loads reduce instruction count
   - In-place operations reduce memory traffic
   - Fixed-size operations simplify control logic

**Congratulations!** You've implemented a complete, efficient attention kernel for our new accelerator. This pattern extends to larger models by tiling across multiple 64×64 blocks.

### Implementation Hints

1. **Use `load_cm` for K**: Loading K in column-major gives you K^T automatically

   ```python
   api.load_cm(n=64, addr_in=8192, addr_out=64)  # Loads K^T into d1[64:127]
   ```

2. **Remember GEMM output location**: Always goes to d2, regardless of inputs

3. **Softmax is in-place**: Operates on and modifies d2 directly

   ```python
   api.softmax(n=64, addr=0)  # Applies to d2[0:63]
   ```

4. **Reuse scratchpad space**: After computing S, you can overwrite K^T's space with V

5. **Count your instructions**: The solution uses exactly 9 instructions total

---

## Step 4: Testing Your Implementation

Once you've implemented the attention kernel, it's time to validate it against real hardware data.

### Understanding the Test Script

The test script `test_qkv.py` does the following:

1. **Imports the kernel programming infrastructure** (generated in Exercise 1)
2. **Imports your attention kernel** from `asm/attention.py`
3. **Loads FPGA test data** (Q, K, V inputs and golden output)
4. **Compiles the kernel** for functional simulation
5. **Runs the simulation** with the FPGA inputs
6. **Compares outputs** with the FPGA golden reference

### Running the Test

Now that you've implemented the attention kernel, let's test it against real FPGA data using the generated test oracle.
**This step requires running inside the Docker container.**

From your host machine in the `tutorials/micro25/` directory, launch Docker:

```bash
./docker.sh --sim
```

Inside the Docker container, you'll be at the `/workspace` which maps to the `act/` repository.

```bash
# Run the test script
python test_qkv.py
```

### Expected Output (Successful Implementation)

```
Simulation ready in 523ms
Loaded data/Q.dat, data/K.dat, data/V.dat (raw bfloat16 bits)
Simulation ran in 847ms
Inputs:
  Q: (64, 64), bfloat16
  K: (64, 64), bfloat16
  V: (64, 64), bfloat16
Outputs:
  Output: (64, 64), bfloat16
Loaded data/attention.dat (raw bfloat16 bits) as golden output
Max absolute difference between simulation and golden: 0.0
Output matches golden exactly!
Great! Your hand-written attention kernel is perfect!
```

### Understanding the Test Data

**Where does this data come from?**

The test data (`Q.dat`, `K.dat`, `V.dat`, `attention.dat`) comes from a **real FPGA implementation** of the QKV accelerator. Here's the process:

1. **Design**: We specified the QKV accelerator using **Allo** (Accelerator Design Language)

   - Allo: https://github.com/cornell-zhang/allo
   - Developed at Cornell, allows high-level accelerator design

2. **Synthesis**: Allo generated Vitis HLS and synthesized it to an FPGA

3. **Execution**: We ran the attention computation on the physical FPGA

4. **Data Capture**: We saved the inputs and outputs as raw BF16 binary files

5. **Tutorial Use**: Since running FPGA synthesis at the tutorial is infeasible (takes hours), we provide the saved data for validation

**Why is this important?**

- Your simulation must match real hardware behavior
- Validates that TAIDL semantics correctly model the accelerator we discussed
- Validates that the hand-written kernel correctly represents the attention computation

### Debugging Failed Tests

If your output doesn't match the golden reference:

```
Max absolute difference between simulation and golden: 1.25
Output does not match golden.
Please debug your hand-written attention kernel.
```

**Common Issues:**

1. **Wrong load order**: Did you load K^T with `load_cm`?
2. **Incorrect addresses**: Check HBM addresses (0, 8192, 16384, 24576)
3. **Scratchpad conflicts**: Ensure d1 row ranges don't overlap within a stage
4. **Missing mov**: Did you move results from d2 to d1 before storing?
5. **Wrong GEMM operands**: Check `addr_1` and `addr_2` point to correct d1 rows

**Need help?** Check the solution in `exercise2/solution/asm/attention.py` (but try to implement it yourself first!)

---

## Key Takeaways

### What You've Learned

1. **Kernel Programming Model**: How to write accelerator programs using generated ISA APIs

2. **Memory Management**: Explicit control over:

   - HBM address allocation
   - Scratchpad resource allocation
   - Data movement orchestration

3. **Hardware Constraints**: Working within architectural limitations:

   - GEMM outputs only to d2
   - Softmax operates only on d2
   - Data must be in d1 for load/store

4. **Real Hardware Validation**: Testing against FPGA-generated golden outputs

### The Manual Kernel Programming Challenge

By completing this exercise, you've experienced firsthand the **tedious and error-prone nature** of manual kernel programming:

- **Low-level details**: Explicit address management for every tensor
- **Hardware quirks**: Workarounds for architectural constraints (e.g., dummy GEMMs)
- **Repetitive patterns**: Same load-compute-store patterns for every operation
- **Debugging difficulty**: Hard to isolate issues without fine-grained tracing

**The key question**: What if we could **automate** this entire process?

### Motivation for Exercise 3

In the upcoming talk (see tutorial agenda), you'll learn about **automatic compiler backend generation**. The ACT ecosystem automatically generates compiler backends that:

- Takes a high-level tensor program (e.g., attention in XLA-HLO)
- Automatically generate the instruction sequence you just wrote by hand
- Handle memory allocation, address calculation, and constraint satisfaction

**Exercise 3** will demonstrate this: you'll generate a complete compiler backend from the QKV ISA specification, then watch it automatically compile high-level operations into low-level kernels -- no manual programming required!

---

## Next Steps: Exercise 3

Now that you've hand-written accelerator kernels using the previously generated kernel programming APIs and validated them using the auto-generated test oracle, you're ready to:

1. **Learn about automatic compiler backend generation** in Akash's talk after the break
2. **Generate a complete compiler backend** from your QKV ISA specification
3. **Write high-level HLO programs** instead of low-level assembly
4. **Compile and test** automatically generated kernels against the same FPGA data

The manual kernel you wrote represents what compilers must generate automatically. Proceed to [Exercise 3: Generating a Compiler Backend](../exercise3/README.md) where you'll see how ACT eliminates this manual effort entirely!

---

## Additional Resources

- **TAIDL Paper**: Section 6-7 for test oracle implementation and evaluation
- **Example Kernels**: `taidl-artifact-micro25/accelerators/*/kernels.py`
