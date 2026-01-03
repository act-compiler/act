import jax.numpy as jnp

def qkv(kernel, api):
    @kernel(
        hbm=32768,
        input=[
        {'addr': 0, 'shape':(64,64), 'dtype': jnp.bfloat16}, #Q
        {'addr':8192, 'shape':(64,64), 'dtype': jnp.bfloat16}, #K
        {'addr':16384, 'shape':(64,64), 'dtype': jnp.bfloat16}, #V
        ],
        constant=[],
        output=[
            {'addr':24576, 'shape':(64,64), 'dtype': jnp.bfloat16},
        ]
    )

    def qkv_():
        api.load1_rm(n=64, addr_in = 0, addr_out = 0) #Load all of Q (64 rows) into d1 at addr 0.
        api.load1_rm(n=64, addr_in=8192, addr_out=64) #Load K into the 2nd set of 64 rows in d1 to transpose later
        api.mov_cm(n=64, addr_in=64, addr_out=0) #transpose K from d1[64] to d3[0]
        api.gemm13(addr_1=0, addr_2=0, addr_out=0) #multiply Q in d1[0] with K^T in d3[0]. deposit result in d2[0]
        api.softmax(n=64, addr=0) #softmax (Q x K^T) already located in d2[0] in place
        api.mov3(n=64, addr_in=0, addr_out=0) #move softmax(QxK^T) from d2[0] to d3[0]. overwrites K^T
        api.load3_rm(n=64, addr_in=16384, addr_out=64) #load in v to d3[64].
        api.gemm33(addr_1=0, addr_2=64, addr_out=0) #perform final matrix multiplication. 1st input from d3[0], 2nd input from d3[64]. output in d2[0]
        api.mov1(n=64, addr_in=0, addr_out=0) #moving our final result to d1 so I can output it
        api.store1_rm(n=64, addr_in=0, addr_out=24576) #yay we did it we computed attention
    return qkv_