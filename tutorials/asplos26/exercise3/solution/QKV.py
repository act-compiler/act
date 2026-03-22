"""QKV Accelerator ISA Definition"""

from taidl import Accelerator

qkv = Accelerator("QKV")


# Define Data Models

# d1: 128 rows x 64 columns of bf16
qkv.add_data_model("d1", [128], [64], "bf16")

# d2: 64 rows x 64 columns of bf16
qkv.add_data_model("d2", [64], [64], "bf16")


# Define Instruction semantics

# (1) load_rm: Loads data from HBM (d0) in row-major format to d1
instr = qkv.add_instruction("load_rm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY load_rm {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

# (2) load_cm: Loads data from HBM (d0) in column-major format to d1 (with transpose)
instr = qkv.add_instruction("load_cm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY load_cm {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    %b = bf16[`@c.n`,64] bitcast_convert(%a);
    ROOT %Out0 = bf16[64,`@c.n`] transpose(%b), dimensions={1,0};
}
""")

# (3) store_rm: Stores data from d1 to HBM (d0) in row-major format
instr = qkv.add_instruction("store_rm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY store_rm {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")

# (4) store_cm: Stores data from d1 to HBM (d0) in column-major format (with transpose)
instr = qkv.add_instruction("store_cm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY store_cm {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[64,`@c.n`] transpose(%In1), dimensions={1,0};
    %b = u8[64,`@c.n`,2] bitcast_convert(%a);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%b);
}
""")

# (5) mov: Copies data from d2 to d1
instr = qkv.add_instruction("mov", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d2", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY mov {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
""")

# (6) gemm: Matrix multiplication between two d1 tensors, output to d2
instr = qkv.add_instruction("gemm", [], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_1"], ["64"]], ["d1", ["@a.addr_2"], ["64"]]])
instr.set_outputs([["d2", ["@a.addr_out"], ["64"]]])  # bf16[64, 64]
instr.add_semantics("""
ENTRY gemm {
    %In1 = bf16[64,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

# (7) softmax: Applies softmax along dimension 1 (rows) on d2
instr = qkv.add_instruction("softmax", ["n"], ["addr"])
instr.set_inputs([["d2", ["@a.addr"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d2", ["@a.addr"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY softmax {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[`@c.n`,64] exponential(%In1);
    %reduced = bf16[`@c.n`] reduce_add(%a), dimensions={1};
    %b = bf16[`@c.n`,64] broadcast(%reduced), dimensions={0};
    ROOT %Out0 = bf16[`@c.n`,64] divide(%a, %b);
}
""")


# Generate programming APIs and test oracle (functional simulator)
qkv.generate_oracle()

# Generate compiler backend
qkv.generate_backend()
