"""QKV Accelerator ISA Definition"""

from taidl import Accelerator

qkv = Accelerator("QKV")


# Define Data Models
# d1: 128 rows x 64 columns of bf16
qkv.add_data_model("d1", [128], [64], "bf16")

# d3: 128 rows x 64 columns of bf16
qkv.add_data_model("d3", [128], [64], "bf16")

# d2: 64 rows x 64 columns of bf16
qkv.add_data_model("d2", [64], [64], "bf16")


# Define Instruction semantics

# (1) load_rm: Loads data from HBM (d0) in row-major format to d1
instr = qkv.add_instruction("load_rm_01", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY load_rm_01 {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")


# (2) load_rm: Loads data from HBM (d0) in row-major format to d3
instr = qkv.add_instruction("load_rm_03", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.set_outputs([["d3", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY load_rm_03 {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")


# (3) store_rm: Stores data from d1 to HBM (d0) in row-major format
instr = qkv.add_instruction("store_rm_10", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY store_rm_10 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")

# (4) store_rm: Stores data from d3 to HBM (d0) in row-major format
instr = qkv.add_instruction("store_rm_30", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d3", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d0", ["@a.addr_out"], ["@c.n * 128"]]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY store_rm_30 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")

# (5) store_cm: Moves data from d1 to d3 in row-major format (with transpose)
instr = qkv.add_instruction("transpose_13",[], ["addr_in", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_in"], ["64"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d3", ["@a.addr_out"], ["64"]]])  # bf16[64, @c.n]
instr.add_semantics("""
ENTRY transpose_13 {
    %In1 = bf16[64,64] parameter(0);
    %a = bf16[64,64] transpose(%In1), dimensions={1,0};
    ROOT %Out0 = bf16[64, 64] copy(%a);
}
""")


# (6) mov: Copies data from d2 to d1
instr = qkv.add_instruction("mov_21", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d2", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY mov_21 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
""")

# (7) mov: Copies data from d2 to d3
# Couldn't figure out how to move 64 columns into a 128 column buffer.
instr = qkv.add_instruction("mov_23", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([["d2", ["@a.addr_in"], ["@c.n"]]])  # bf16[@c.n, 64]
instr.set_outputs([["d3", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64] (ignore last 64 columns)
instr.add_semantics("""
ENTRY mov_23 {
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
""")

# (8) gemm: Matrix multiplication between two d3 tensors, output to d2
instr = qkv.add_instruction("gemm_3", [], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["d3", ["@a.addr_1"], ["64"]], ["d3", ["@a.addr_2"], ["64"]]])
instr.set_outputs([["d2", ["@a.addr_out"], ["64"]]])  # bf16[64, 64]
instr.add_semantics("""
ENTRY gemm_3 {
    %In1 = bf16[64,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

# (9) gemm: Matrix multiplication between one d1 tensor and one d3 tensor, output to d2
instr = qkv.add_instruction("gemm_13", [], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([["d1", ["@a.addr_1"], ["64"]], ["d3", ["@a.addr_2"], ["64"]]])
instr.set_outputs([["d2", ["@a.addr_out"], ["64"]]])  # bf16[64, 64]
instr.add_semantics("""
ENTRY gemm_13 {
    %In1 = bf16[64,64] parameter(0);
    %In2 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

# (10) softmax: Applies softmax along dimension 1 (rows) on d2
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




#Unused:
# # (2) load_cm: Loads data from HBM (d0) in column-major format to d3 (with transpose)
# instr = qkv.add_instruction("load_cm", ["n"], ["addr_in", "addr_out"])
# instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
# instr.set_outputs([["d3", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
# instr.add_semantics("""
# ENTRY load_cm {
#     %In1 = u8[`@c.n * 128`] parameter(0);
#     %a = u8[`@c.n`,64,2] reshape(%In1);
#     %b = bf16[`@c.n`,64] bitcast_convert(%a);
#     ROOT %Out0 = bf16[64,`@c.n`] transpose(%b), dimensions={1,0};
# }
# """)

# # (2) load_cm: Loads data from HBM (d0) in column-major format to d1 (with transpose)
# instr = qkv.add_instruction("load_cm", ["n"], ["addr_in", "addr_out"])
# instr.set_inputs([["d0", ["@a.addr_in"], ["@c.n * 128"]]])  # u8[@c.n * 128]
# instr.set_outputs([["d1", ["@a.addr_out"], ["@c.n"]]])  # bf16[@c.n, 64]
# instr.add_semantics("""
# ENTRY load_cm {
#     %In1 = u8[`@c.n * 128`] parameter(0);
#     %a = u8[`@c.n`,64,2] reshape(%In1);
#     %b = bf16[`@c.n`,64] bitcast_convert(%a);
#     ROOT %Out0 = bf16[64,`@c.n`] transpose(%b), dimensions={1,0};
# }
# """)



# Generate programming APIs and test oracle (functional simulator)
qkv.generate_oracle()

# Generate compiler backend
# qkv.generate_backend()
