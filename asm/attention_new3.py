# Input file: log.hlo
# Kernel name: qkv
# PII number: 0
# Do not edit!

import jax.numpy as jnp


def qkv(kernel, api):
    @kernel(hbm=32768,
            input=[
                {'addr': 0, 'shape': (64, 64), 'dtype': jnp.bfloat16},
                {'addr': 8192, 'shape': (64, 64), 'dtype': jnp.bfloat16},
                {'addr': 16384, 'shape': (64, 64), 'dtype': jnp.bfloat16},
            ],
            constant=[],
            output=[
                {'addr': 24576, 'shape': (64, 64), 'dtype': jnp.bfloat16},
            ]
            )
    def qkv_():
        api.load1_rm(n = 64, addr_in = 0, addr_out = 0)
        api.load1_rm(n = 64, addr_in = 8192, addr_out = 64)
        api.mov_cm(n = 64, addr_in = 64, addr_out = 0)
        api.gemm13(addr_1 = 0, addr_2 = 0, addr_out = 0)
        api.softmax(n = 64, addr = 0)
        api.mov1(n = 64, addr_in = 0, addr_out = 0)
        api.load3_rm(n = 64, addr_in = 16384, addr_out = 0)
        api.gemm13(addr_1 = 0, addr_2 = 0, addr_out = 0)
        api.mov1(n = 64, addr_in = 0, addr_out = 0)
        api.store1_rm(n = 64, addr_in = 0, addr_out = 24576)

    return qkv_
