"""Gemmini Accelerator ISA Definition"""

from act.taidl import Accelerator
from act.generator import generate_backend

gemmini = Accelerator("Gemmini")

# Define Data Models
gemmini.add_data_model("spad", [1024 * 16, 16], [], "s8")
gemmini.add_data_model("acc", [64 * 16, 16], [], "s32")

# --- SPAD load/store ---

instr = gemmini.add_instruction("mvin_spad", ["rows", "cols", "scale"], ["hbm_addr", "sp_addr"])
instr.set_inputs([["d0", ["@a.hbm_addr"], ["@c.rows * @c.cols"]]])
instr.set_outputs([["spad", ["@a.sp_addr", 0], ["@c.rows", "@c.cols"]]])
instr.add_semantics("""
ENTRY mvin_spad{
    %In1 = u8[`@c.rows*@c.cols`] parameter(0);
    %data = u8[`@c.rows`,`@c.cols`] reshape(%In1);
    %val = s8[`@c.rows`,`@c.cols`] bitcast_convert(%data);
    %scale = s8[1] constant(`@c.scale`);
    %scale_bc = s8[`@c.rows`,`@c.cols`] broadcast(%scale), dimensions={};
    ROOT %Out0 = s8[`@c.rows`,`@c.cols`] multiply(%val, %scale_bc);
}
""")

instr = gemmini.add_instruction("mvout_spad", ["rows", "cols"], ["hbm_addr", "sp_addr"])
instr.set_inputs([["spad", ["@a.sp_addr", 0], ["@c.rows", "@c.cols"]]])
instr.set_outputs([["d0", ["@a.hbm_addr"], ["@c.rows * @c.cols"]]])
instr.add_semantics("""
ENTRY mvout_spad{
    %In1 = s8[`@c.rows`, `@c.cols`] parameter(0);
    %data = u8[`@c.rows`,`@c.cols`] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.rows*@c.cols`] reshape(%data);
}
""")

instr = gemmini.add_instruction("mvout_spad_relu", ["rows", "cols"], ["hbm_addr", "sp_addr"])
instr.set_inputs([["spad", ["@a.sp_addr", 0], ["@c.rows", "@c.cols"]]])
instr.set_outputs([["d0", ["@a.hbm_addr"], ["@c.rows * @c.cols"]]])
instr.add_semantics("""
ENTRY mvout_spad_relu{
    %In1 = s8[`@c.rows`, `@c.cols`] parameter(0);
    %zero = s8[1] constant(0);
    %zeros = s8[`@c.rows`,`@c.cols`] broadcast(%zero), dimensions={};
    %relu = s8[`@c.rows`,`@c.cols`] maximum(%In1, %zeros);
    %data = u8[`@c.rows`,`@c.cols`] bitcast_convert(%relu);
    ROOT %Out0 = u8[`@c.rows*@c.cols`] reshape(%data);
}
""")

# --- ACC load/store (s32) ---

instr = gemmini.add_instruction("mvin_acc", ["rows", "cols", "scale"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["d0", ["@a.hbm_addr"], ["@c.rows * @c.cols * 4"]]])
instr.set_outputs([["acc", ["@a.acc_addr", 0], ["@c.rows", "@c.cols"]]])
instr.add_semantics("""
ENTRY mvin_acc{
    %In1 = u8[`@c.rows * @c.cols * 4`] parameter(0);
    %r = u8[`@c.rows`,`@c.cols`,4] reshape(%In1);
    %val = s32[`@c.rows`, `@c.cols`] bitcast_convert(%r);
    %scale = s32[1] constant(`@c.scale`);
    %scale_bc = s32[`@c.rows`,`@c.cols`] broadcast(%scale), dimensions={};
    ROOT %Out0 = s32[`@c.rows`,`@c.cols`] multiply(%val, %scale_bc);
}
""")

instr = gemmini.add_instruction("mvout_acc", ["rows", "cols"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["acc", ["@a.acc_addr", 0], ["@c.rows", "@c.cols"]]])
instr.set_outputs([["d0", ["@a.hbm_addr"], ["@c.rows * @c.cols * 4"]]])
instr.add_semantics("""
ENTRY mvout_acc{
    %In1 = s32[`@c.rows`, `@c.cols`] parameter(0);
    %data = u8[`@c.rows`,`@c.cols`,4] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.rows*@c.cols*4`] reshape(%data);
}
""")

instr = gemmini.add_instruction("mvout_acc_relu", ["rows", "cols"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["acc", ["@a.acc_addr", 0], ["@c.rows", "@c.cols"]]])
instr.set_outputs([["d0", ["@a.hbm_addr"], ["@c.rows * @c.cols * 4"]]])
instr.add_semantics("""
ENTRY mvout_acc_relu{
    %In1 = s32[`@c.rows`, `@c.cols`] parameter(0);
    %zero = s32[1] constant(0);
    %zeros = s32[`@c.rows`,`@c.cols`] broadcast(%zero), dimensions={};
    %relu = s32[`@c.rows`,`@c.cols`] maximum(%In1, %zeros);
    %data = u8[`@c.rows`,`@c.cols`,4] bitcast_convert(%relu);
    ROOT %Out0 = u8[`@c.rows*@c.cols*4`] reshape(%data);
}
""")

# --- ACC load/store (s8 <-> s32 with convert) ---

instr = gemmini.add_instruction("mvin_acc_low", ["rows", "cols", "scale"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["d0", ["@a.hbm_addr"], ["@c.rows * @c.cols"]]])
instr.set_outputs([["acc", ["@a.acc_addr", 0], ["@c.rows", "@c.cols"]]])
instr.add_semantics("""
ENTRY mvin_acc_low{
    %In1 = u8[`@c.rows * @c.cols`] parameter(0);
    %a = u8[`@c.rows`,`@c.cols`] reshape(%In1);
    %b = s8[`@c.rows`,`@c.cols`] bitcast_convert(%a);
    %scale = s8[1] constant(`@c.scale`);
    %scale_bc = s8[`@c.rows`,`@c.cols`] broadcast(%scale), dimensions={};
    %c = s8[`@c.rows`,`@c.cols`] multiply(%b, %scale_bc);
    ROOT %Out0 = s32[`@c.rows`,`@c.cols`] convert(%c);
}
""")

instr = gemmini.add_instruction("mvout_acc_low", ["rows", "cols"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["acc", ["@a.acc_addr", 0], ["@c.rows", "@c.cols"]]])
instr.set_outputs([["d0", ["@a.hbm_addr"], ["@c.rows * @c.cols"]]])
instr.add_semantics("""
ENTRY mvout_acc_low{
    %In1 = s32[`@c.rows`, `@c.cols`] parameter(0);
    %a = s8[`@c.rows`,`@c.cols`] convert(%In1);
    %b = u8[`@c.rows`,`@c.cols`] bitcast_convert(%a);
    ROOT %Out0 = u8[`@c.rows * @c.cols`] reshape(%b);
}
""")

instr = gemmini.add_instruction("mvout_acc_low_relu", ["rows", "cols"], ["hbm_addr", "acc_addr"])
instr.set_inputs([["acc", ["@a.acc_addr", 0], ["@c.rows", "@c.cols"]]])
instr.set_outputs([["d0", ["@a.hbm_addr"], ["@c.rows * @c.cols"]]])
instr.add_semantics("""
ENTRY mvout_acc_low_relu{
    %In1 = s32[`@c.rows`, `@c.cols`] parameter(0);
    %zero = s32[1] constant(0);
    %zeros = s32[`@c.rows`,`@c.cols`] broadcast(%zero), dimensions={};
    %relu = s32[`@c.rows`,`@c.cols`] maximum(%In1, %zeros);
    %a = s8[`@c.rows`,`@c.cols`] convert(%relu);
    %b = u8[`@c.rows`,`@c.cols`] bitcast_convert(%a);
    ROOT %Out0 = u8[`@c.rows * @c.cols`] reshape(%b);
}
""")

# --- Matmul / MAC ---

instr = gemmini.add_instruction("matmul8", ["DIM_I", "DIM_J", "DIM_K"], ["C_dst", "A_src", "B_src"])
instr.set_inputs([["spad", ["@a.A_src", 0], ["@c.DIM_I", "@c.DIM_K"]],
                  ["spad", ["@a.B_src", 0], ["@c.DIM_K", "@c.DIM_J"]],
                  ])
instr.set_outputs([["spad", ["@a.C_dst", 0], ["@c.DIM_I", "@c.DIM_J"]]])
instr.add_semantics("""
ENTRY matmul_8{
    %In1 = s8[`@c.DIM_I`, `@c.DIM_K`] parameter(0);
    %In2 = s8[`@c.DIM_K`, `@c.DIM_J`] parameter(1);
    ROOT %Out0 = s8[`@c.DIM_I`,`@c.DIM_J`] dot(%In1, %In2), lhs_batch_dims={}, lhs_contracting_dims={1}, rhs_batch_dims={}, rhs_contracting_dims={0};
}
""")

instr = gemmini.add_instruction("matmul32", ["DIM_I", "DIM_J", "DIM_K"], ["C_dst", "A_src", "B_src"])
instr.set_inputs([["spad", ["@a.A_src", 0], ["@c.DIM_I", "@c.DIM_K"]],
                  ["spad", ["@a.B_src", 0], ["@c.DIM_K", "@c.DIM_J"]],
                  ])
instr.set_outputs([["acc", ["@a.C_dst", 0], ["@c.DIM_I", "@c.DIM_J"]]])
instr.add_semantics("""
ENTRY matmul_32{
    %In1 = s8[`@c.DIM_I`, `@c.DIM_K`] parameter(0);
    %In2 = s8[`@c.DIM_K`, `@c.DIM_J`] parameter(1);
    %a = s32[`@c.DIM_I`, `@c.DIM_K`] convert(%In1);
    %b = s32[`@c.DIM_K`, `@c.DIM_J`] convert(%In2);
    ROOT %Out0 = s32[`@c.DIM_I`,`@c.DIM_J`] dot(%a, %b), lhs_batch_dims={}, lhs_contracting_dims={1}, rhs_batch_dims={}, rhs_contracting_dims={0};
}
""")

instr = gemmini.add_instruction("mac8", ["DIM_I", "DIM_J", "DIM_K"], ["C_dst", "A_src", "B_src", "D_src"])
instr.set_inputs([["spad", ["@a.A_src", 0], ["@c.DIM_I", "@c.DIM_K"]],
                  ["spad", ["@a.B_src", 0], ["@c.DIM_K", "@c.DIM_J"]],
                  ["spad", ["@a.D_src", 0], ["@c.DIM_I", "@c.DIM_J"]]
                  ])
instr.set_outputs([["spad", ["@a.C_dst", 0], ["@c.DIM_I", "@c.DIM_J"]]])
instr.add_semantics("""
ENTRY mac_8{
    %In1 = s8[`@c.DIM_I`, `@c.DIM_K`] parameter(0);
    %In2 = s8[`@c.DIM_K`, `@c.DIM_J`] parameter(1);
    %In3 = s8[`@c.DIM_I`, `@c.DIM_J`] parameter(2);
    %dot = s8[`@c.DIM_I`, `@c.DIM_J`] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = s8[`@c.DIM_I`, `@c.DIM_J`] add(%dot, %In3);
}
""")

instr = gemmini.add_instruction("mac32", ["DIM_I", "DIM_J", "DIM_K"], ["C_dst", "A_src", "B_src", "D_src"])
instr.set_inputs([["spad", ["@a.A_src", 0], ["@c.DIM_I", "@c.DIM_K"]],
                  ["spad", ["@a.B_src", 0], ["@c.DIM_K", "@c.DIM_J"]],
                  ["acc", ["@a.D_src", 0], ["@c.DIM_I", "@c.DIM_J"]],
                  ])
instr.set_outputs([["acc", ["@a.C_dst", 0], ["@c.DIM_I", "@c.DIM_J"]]])
instr.add_semantics("""
ENTRY mac_32{
    %In1 = s8[`@c.DIM_I`, `@c.DIM_K`] parameter(0);
    %In2 = s8[`@c.DIM_K`, `@c.DIM_J`] parameter(1);
    %In3 = s32[`@c.DIM_I`, `@c.DIM_J`] parameter(2);
    %a = s32[`@c.DIM_I`,`@c.DIM_K`] convert(%In1);
    %b = s32[`@c.DIM_K`,`@c.DIM_J`] convert(%In2);
    %dot = s32[`@c.DIM_I`,`@c.DIM_J`] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = s32[`@c.DIM_I`,`@c.DIM_J`] add(%dot, %In3);
}
""")

generate_backend(gemmini)
