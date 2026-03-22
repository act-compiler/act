"""QKV Accelerator ISA Definition"""

from taidl import Accelerator

qkv = Accelerator("QKV")


# Define Data Models

# d1: 128 rows x 64 columns of bf16
# d2: 64 rows x 64 columns of bf16


# Define Instruction semantics

# (1) load_rm: Loads data from HBM (d0) in row-major format to d1

# (2) load_cm: Loads data from HBM (d0) in column-major format to d1 (with transpose)

# (3) store_rm: Stores data from d1 to HBM (d0) in row-major format

# (4) store_cm: Stores data from d1 to HBM (d0) in column-major format (with transpose)

# (5) mov: Copies data from d2 to d1

# (6) gemm: Matrix multiplication between two d1 tensors, output to d2

# (7) softmax: Applies softmax along dimension 1 (rows) on d2


# Generate programming APIs and test oracle (functional simulator)
qkv.generate_oracle()
