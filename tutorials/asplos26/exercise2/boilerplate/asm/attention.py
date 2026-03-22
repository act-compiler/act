import jax.numpy as jnp


def qkv(kernel, api):
    @kernel(
        hbm=32768,    # 32 KB: enough for 3 inputs + 1 output
        input=[],     # allocate the input addresses here
        constant=[],  # if needed, we can add constants here
        output=[]     # allocate the output address here
    )
    def qkv_():
        # Kernel implementation goes here
        pass

    return qkv_
