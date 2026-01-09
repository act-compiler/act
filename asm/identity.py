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
