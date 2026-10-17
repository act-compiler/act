"""ACT Generators - each generator, imported on call"""


def generate_oracle(*args, **kwargs):
    # imported on call, so the oracle needs no backend
    import taidl_to

    return taidl_to.generate_oracle(*args, **kwargs)


def generate_backend(*args, **kwargs):
    # imported on call, so the backend needs no oracle
    import act_backend

    return act_backend.generate_backend(*args, **kwargs)


__all__ = ['generate_backend', 'generate_oracle']
