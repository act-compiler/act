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
