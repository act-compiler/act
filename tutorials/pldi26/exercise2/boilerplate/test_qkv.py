try:
    import sys
    sys.path.insert(0, '/workspace/targets/QKV')
    from oracle.decorator import kernel
    import oracle.api as api

    from oracle.decorator import set_simulation_backend
    # Run the simulations on CPU
    set_simulation_backend('CPU')
except Exception as e:
    print(f"Error setting path for QKV Oracle: {e}")
    print("Make sure you generated the Oracle using the generator in Hands-on Exercise 1.")
    exit(1)

try:
    import sys
    sys.path.insert(0, '/workspace/asm')
    from attention import qkv
except Exception as e:
    print(f"Error importing attention kernel: {e}")
    print("Make sure you implemented the attention kernel in asm/attention.py.")
    exit(1)


import os
import numpy as np
import jax
import jax.numpy as jnp


DATA_DIR = "/workspace/data/"


def load_bf16_matrix(path, shape):
    """
    Load a raw 8-bit file and reinterpret as jax bfloat16 matrix with given shape.
    """
    np_uint8 = np.fromfile(path, dtype=np.uint8)
    if np_uint8.size != (shape[0] * shape[1] * 2):
        raise ValueError(f"Data in {path} has size {np_uint8.size}, expected {shape[0]*shape[1]}")
    np_uint8 = np_uint8.reshape(shape[0], shape[1], 2)
    j_uint8 = jnp.array(np_uint8, dtype=jnp.uint8)
    mat = jax.lax.bitcast_convert_type(j_uint8, jnp.bfloat16)
    return mat


if __name__ == "__main__":
    # Compile the simulation for the attention kernel
    qkv_kernel = qkv(kernel, api)
    inputs, compile_time = qkv_kernel('fsim-compile')()
    print(f"Simulation ready in {compile_time}ms")

    # Load input data
    Q = load_bf16_matrix(os.path.join(DATA_DIR, "Q.dat"), (64, 64))
    K = load_bf16_matrix(os.path.join(DATA_DIR, "K.dat"), (64, 64))
    V = load_bf16_matrix(os.path.join(DATA_DIR, "V.dat"), (64, 64))
    print("Loaded data/Q.dat, data/K.dat, data/V.dat (raw bfloat16 bits)")

    # Run the simulation
    outputs, elapsed = qkv_kernel('fsim')(Q, K, V)
    qkv_output = outputs[0]
    print(f"Simulation ran in {elapsed}ms")

    # Print input and output shapes and dtypes
    print(f"Inputs:")
    print(f"  Q: {Q.shape}, {Q.dtype}")
    print(f"  K: {K.shape}, {K.dtype}")
    print(f"  V: {V.shape}, {V.dtype}")
    print(f"Outputs:")
    print(f"  Output: {qkv_output.shape}, {qkv_output.dtype}")

    # Load golden output (from FPGA implementation of QKV accelerator)
    golden = load_bf16_matrix(os.path.join(DATA_DIR, "attention.dat"), (64, 64))
    print("Loaded data/attention.dat (raw bfloat16 bits) as golden output")

    # Compare simulation output of attention kernel with golden
    max_diff = jnp.max(jnp.abs(qkv_output - golden))
    print(f"Max absolute difference between simulation and golden: {max_diff}")
    if max_diff == 0:
        print("Output matches golden exactly!")
        print("Great! Your hand-written attention kernel is perfect!")
    else:
        print("Output does not match golden.")
        print("Please debug your hand-written attention kernel.")
