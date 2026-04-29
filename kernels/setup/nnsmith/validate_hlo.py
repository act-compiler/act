import re
import numpy as np
from jaxlib import xla_client

DTYPE_MAP = {
    "pred": np.bool_,
    "s8": np.int8,
    "s16": np.int16,
    "s32": np.int32,
    "s64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
}


def parse_input_specs(hlo_txt):
    """Parse input tensor specs from HLO module header."""
    header = hlo_txt.split('\n')[0]
    layout_match = re.search(r'entry_computation_layout=\{(.*)\}', header)
    if not layout_match:
        raise ValueError(f"Could not parse entry_computation_layout from header: {header}")

    layout = layout_match.group(1)
    inputs_str = layout.split('->')[0]

    # Extract individual tensor specs like "f32[128,128]" or "pred[]"
    specs = re.findall(r'(\w+)\[([^\]]*)\]', inputs_str)
    result = []
    for dtype_str, dims_str in specs:
        np_dtype = DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype in HLO header: {dtype_str}")
        if dims_str.strip() == '':
            shape = ()
        else:
            shape = tuple(int(d.strip()) for d in dims_str.split(',') if d.strip())
        result.append((np_dtype, shape))
    return result


def make_random_inputs(specs):
    """Generate random numpy arrays for each input spec."""
    inputs = []
    for np_dtype, shape in specs:
        if np_dtype == np.bool_:
            inputs.append(np.random.choice([True, False], size=shape))
        elif np.issubdtype(np_dtype, np.integer):
            inputs.append(np.random.randint(0, 8, size=shape).astype(np_dtype))
        else:
            inputs.append(np.random.randn(*shape).astype(np_dtype) if shape else np.array(np.random.randn(), dtype=np_dtype))
    return inputs


def validate_hlo(hlo_file, execute=True, verbose=False):
    """
    Validate that an XLA HLO file can be compiled and executed.

    Args:
        hlo_file: Path to the HLO file to validate
        execute: If True, also run the compiled executable with random inputs
        verbose: If True, print messages

    Returns:
        tuple: (success: bool, executable or None, error or None)
    """
    try:
        with open(hlo_file, 'r') as f:
            hlo_txt = f.read()

        hlo_module = xla_client._xla.hlo_module_from_text(hlo_txt)

        backend = xla_client.make_cpu_client()
        options = xla_client.CompileOptions()
        hlo_proto = hlo_module.as_serialized_hlo_module_proto()
        computation = xla_client.XlaComputation(hlo_proto)
        mlir = xla_client._xla.mlir.xla_computation_to_mlir_module(computation)
        executable = backend.compile(mlir, compile_options=options)

        if execute:
            specs = parse_input_specs(hlo_txt)
            inputs = make_random_inputs(specs)
            result = xla_client.execute_with_python_values(executable, inputs, backend)
            if verbose:
                print(f"Compiled and executed successfully ({hlo_file})")
        else:
            if verbose:
                print(f"Compiled successfully ({hlo_file})")

        return True, executable, None
    except Exception as e:
        if verbose:
            print(f"Error: {hlo_file}: {e}")
        return False, None, e
