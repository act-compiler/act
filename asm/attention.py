import jax.numpy as jnp

def qkv(kernel, api):
    @kernel(
        hbm=32768,    # 32 KB: enough for 3 inputs + 1 output
        input=[
            {'addr': 0, 'shape': (64,64), 'dtype': jnp.bfloat16},      #Q
            {'addr': 8192, 'shape': (64, 64), 'dtype': jnp.bfloat16},  #K
            {'addr': 16384, 'shape': (64, 64), 'dtype': jnp.bfloat16}, #V
        ],     # allocate the input addresses here
        constant=[],  # if needed, we can add constants here (none in this case)
        output=[
            {'addr': 24576, 'shape': (64,64), 'dtype': jnp.bfloat16}, #O
        ]     # allocate the output address here
    )
    def qkv_():
        # Kernel implementation goes here

        # Load Q in row-major (standard)
        api.load_rm(n=64, addr_in=0, addr_out=0)

        # Load K in column-major (gives us K^T automatically!)
        api.load_cm(n=64, addr_in=8192, addr_out=64)

        # Compute S = Q × K^T
        api.gemm(addr_1=0, addr_2=64, addr_out=0)

        # Apply softmax to S, converting it to P (in-place in d2)
        api.softmax(n=64, addr=0)

        # Move P from d2[0:63] to d1[0:63]
        # Recall: mov copies between scratchpads
        api.mov(n=64, addr_in=0, addr_out=0)

        # Load V into d1[64:127] (reusing K^T's space)
        api.load_rm(n=64, addr_in=16384, addr_out=64)

        # Compute O = P × V, result in d2[0:63]
        api.gemm(addr_1=0, addr_2=64, addr_out=0)

        # Move O from d2[0:63] to d1[0:63]
        api.mov(n=64, addr_in=0, addr_out=0)

        # Store O to HBM at address 24576
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    return qkv_
