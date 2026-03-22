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
