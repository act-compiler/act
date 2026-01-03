"""QKV Accelerator ISA definition, new exercise"""

from taidl import Accelerator

qkv = Accelerator("QKV")

#data models

#d1 is I/O but only 64/64 because we need to be able to transpose it into d3 and we don't want d3 to be 128 by 128
qkv.add_data_model("d1", [128], [64], "bf16")
#d2 is intermediate value buffer. we're leaving it as 64 by 64 because that's all we need
qkv.add_data_model("d2", [64], [64], "bf16")
#d3 has to be 128 rows so that we can gemm (matrix multiplication) from just d3 into d2
qkv.add_data_model("d3", [128], [64], "bf16")

#instruction semantics
#notes: @c means computational attributes, @a means addressing attributes.
#d0 is implicit off-chip HBM/DRAM memory. Its elements are stored in a flat byte-addressed array, whereas all the scratch pads actually have rows and columns


instr = qkv.add_instruction("load1_rm", ["n"], ["addr_in", "addr_out"]) #(instruction_name, [list_of_computational_attributes], [list_of_addressing_attributes])
instr.set_inputs([[ "d0", ["@a.addr_in"], ["@c.n * 128"] ]]) #([[ input_buffer, [addressing_attribute], [size_of_input] ]]). here we do c.n * 128 because c.n is the number of rows and for each row there are 64 bf16s which are each 2 bytes a piece (1 byte = 8 bits), so 128 bytes in total.
instr.set_outputs([[ "d1", ["@a.addr_out"], ["@c.n"] ]]) #([[ output_buffer, [addressing_attribute], [size_of_input] ]]). here we don't multiply by 128 because d1 already has row-size built-in
instr.add_semantics("""
ENTRY load1_rm {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

instr = qkv.add_instruction("load3_rm", ["n"], ["addr_in", "addr_out"]) #(instruction_name, [list_of_computational_attributes], [list_of_addressing_attributes])
instr.set_inputs([[ "d0", ["@a.addr_in"], ["@c.n * 128"] ]]) #([[ input_buffer, [addressing_attribute], [size_of_input] ]]). here we do c.n * 128 because c.n is the number of rows and for each row there are 64 bf16s which are each 2 bytes a piece (1 byte = 8 bits), so 128 bytes in total.
instr.set_outputs([[ "d3", ["@a.addr_out"], ["@c.n"] ]]) #([[ output_buffer, [addressing_attribute], [size_of_input] ]]). here we don't multiply by 128 because d1 already has row-size built-in
instr.add_semantics("""
ENTRY load3_rm {
    %In1 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

instr = qkv.add_instruction("store1_rm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([[ "d1", ["@a.addr_in"], ["@c.n"] ]])
instr.set_outputs([[ "d0", ["@a.addr_out"], ["@c.n * 128"] ]])
instr.add_semantics("""
ENTRY store1_rm {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")

instr = qkv.add_instruction("store3_rm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([[ "d3", ["@a.addr_in"], ["@c.n"] ]])
instr.set_outputs([[ "d0", ["@a.addr_out"], ["@c.n * 128"] ]])
instr.add_semantics("""
ENTRY store3_rm {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In1);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")

instr = qkv.add_instruction("mov1", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([[ "d2", ["@a.addr_in"], ["@c.n"] ]])
instr.set_outputs([[ "d1", ["@a.addr_out"], ["@c.n"] ]])
instr.add_semantics("""
ENTRY mov1 {
    %In1 = bf16[`@c.n`,64] parameter(0); 
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
""")

instr = qkv.add_instruction("mov3", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([[ "d2", ["@a.addr_in"], ["@c.n"] ]])
instr.set_outputs([[ "d3", ["@a.addr_out"], ["@c.n"] ]])
instr.add_semantics("""
ENTRY mov3 {
    %In1 = bf16[`@c.n`,64] parameter(0); 
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In1);
}
""")

instr = qkv.add_instruction("mov_cm", ["n"], ["addr_in", "addr_out"])
instr.set_inputs([[ "d1", ["@a.addr_in"], ["@c.n"] ]])
instr.set_outputs([[ "d3", ["@a.addr_out"], ["@c.n"] ]])
instr.add_semantics("""
ENTRY mov_cm { 
    %In1 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[64, `@c.n`] transpose(%In1), dimensions={1,0};
}
""")

#the other 2 errors are in gem13 (forgot = between ..._dims={1} and ..._dims{0} in both this and gemm33

instr = qkv.add_instruction("gemm13", [], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([ ["d1", ["@a.addr_1"], ["64"]], ["d3", ["@a.addr_2"], ["64"]] ])
instr.set_outputs([ ["d2", ["@a.addr_out"], ["64"]] ])
instr.add_semantics("""
ENTRY gemm13 {
    %In1 = bf16[64, 64] parameter(0);
    %In2 = bf16[64, 64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

#2 of the errors are in gemm33

instr = qkv.add_instruction("gemm33", [], ["addr_1", "addr_2", "addr_out"])
instr.set_inputs([ ["d3", ["@a.addr_1"], ["64"]], ["d3", ["@a.addr_2"], ["64"]] ])
instr.set_outputs([ ["d2", ["@a.addr_out"], ["64"]] ])
instr.add_semantics("""
ENTRY gemm33 {
    %In1 = bf16[64, 64] parameter(0);
    %In2 = bf16[64, 64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In1, %In2), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

instr = qkv.add_instruction("softmax", ["n"], ["addr"])
instr.set_inputs([["d2", ["@a.addr"], ["@c.n"]]])
instr.set_outputs([["d2", ["@a.addr"], ["@c.n"]]])
instr.add_semantics("""
ENTRY softmax {
    %In1 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[`@c.n`,64] exponential(%In1);
    %reduced = bf16[`@c.n`] reduce_add(%a), dimensions={1};
    %b = bf16[`@c.n`,64] broadcast(%reduced), dimensions={0};
    ROOT %Out0 = bf16[`@c.n`,64] divide(%a, %b);
}
""")

qkv.generate_oracle()

qkv.generate_backend()