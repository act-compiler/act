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
