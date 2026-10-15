"""One spec through both generators, then a kernel compiled and simulated"""

import importlib
import importlib.util
import subprocess
import sys

import jax.numpy as jnp
import numpy as np

from act.generators import generate_backend, generate_oracle
from act.taidl import Accelerator

# The kernel: the product of two 4x4 matrices
KERNEL = """\
HloModule matmul, entry_computation_layout={(bf16[4,4]{1,0}, bf16[4,4]{1,0})->bf16[4,4]{1,0}}

ENTRY main {
  %a = bf16[4,4] parameter(0)
  %b = bf16[4,4] parameter(1)
  ROOT %c = bf16[4,4] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
"""


def accelerator(base_dir):
    """Return MM: load, gemm and store over one buffer of 4-column rows"""
    acc = Accelerator('MM', base_dir)
    acc.add_data_model('d1', [12], [4], 'bf16')

    # load: n rows from HBM to d1
    instr = acc.add_instruction('load', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['@c.n * 8']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])
    instr.add_semantics("""
ENTRY load {
    %In1 = u8[`@c.n * 8`] parameter(0);
    %a = u8[`@c.n`,4,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,4] bitcast_convert(%a);
}
""")

    # gemm: the product of two 4x4 tiles of d1
    instr = acc.add_instruction('gemm', [], ['addr_1', 'addr_2', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_1'], ['4']], ['d1', ['@a.addr_2'], ['4']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['4']]])
    instr.add_semantics("""
ENTRY gemm {
    %In1 = bf16[4,4] parameter(0);
    %In2 = bf16[4,4] parameter(1);
    ROOT %Out0 = bf16[4,4] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

    # store: n rows from d1 to HBM
    instr = acc.add_instruction('store', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['@c.n']]])
    instr.set_outputs([['d0', ['@a.addr_out'], ['@c.n * 8']]])
    instr.add_semantics("""
ENTRY store {
    %In1 = bf16[`@c.n`,4] parameter(0);
    %a = u8[`@c.n`,4,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n * 8`] reshape(%a);
}
""")
    return acc


def test_end_to_end(tmp_path, pytestconfig):
    acc = accelerator(str(tmp_path))
    generate_oracle(acc)
    generate_backend(acc)

    # The backend compiles the kernel to assembly
    backend, log = tmp_path / 'backends' / 'MM', tmp_path / 'log'
    hlo, assembly = tmp_path / 'matmul.hlo', tmp_path / 'matmul.py'
    hlo.write_text(KERNEL)
    subprocess.run([backend, '--input', hlo, '--output', assembly, '--log', log], check=True)

    # The oracle, imported from its generated package
    oracle_dir = tmp_path / 'targets' / 'MM' / 'oracle'
    spec = importlib.util.spec_from_file_location(
        'oracle', oracle_dir / '__init__.py', submodule_search_locations=[str(oracle_dir)]
    )
    sys.modules['oracle'] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules['oracle'])
    decorator = importlib.import_module('oracle.decorator')
    decorator.set_simulation_backend(pytestconfig.getoption('platform'))

    # The assembly, simulated on two matrices of small integers
    spec = importlib.util.spec_from_file_location('matmul', assembly)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.matmul(decorator.kernel, importlib.import_module('oracle.api'))
    kernel('fsim-compile')()
    a = np.arange(16).reshape(4, 4) % 5
    b = np.arange(16).reshape(4, 4) % 3
    outputs, _ = kernel('fsim')(jnp.array(a, jnp.bfloat16), jnp.array(b, jnp.bfloat16))

    # Small integer products and sums are exact in bf16
    assert np.array_equal(np.asarray(outputs[0], np.float64), a @ b)
