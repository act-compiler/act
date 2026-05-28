import os
from typing import List
from dataclasses import dataclass
from .template_loader import get_backend_template_loader


@dataclass
class InstructionInfo:
    name: str
    arity: int
    has_metadata: bool
    inplace: List[bool]


def get_rust_variant_name(instr_name: str) -> str:
    words = instr_name.replace('-', '_').split('_')
    return ''.join(word.capitalize() for word in words)


def extract_instruction_info(instructions) -> List[InstructionInfo]:
    info_list = []

    for instruction in instructions:
        name = instruction.instruction
        arity = len(instruction.instr_inputs)
        has_metadata = len(instruction.comp_attr) > 0
        inplace = instruction.inplace

        info_list.append(InstructionInfo(name, arity, has_metadata, inplace))

    return info_list


def generate_instruction_variants(instructions_info: List[InstructionInfo], templates):
    """Generate instruction variants using templates"""
    enum_variants = []
    num_children_arms = []
    set_metadata_arms = []
    children_arms = []
    children_mut_arms = []
    from_op_arms = []
    display_arms = []
    inplace_arms = []

    for info in instructions_info:
        variant_name = get_rust_variant_name(info.name)
        kebab_name = info.name.replace('_', '-')

        # Enum variant
        if info.has_metadata:
            signature = f"(String, [Id; {info.arity}])"
        else:
            signature = f"([Id; {info.arity}])"
        enum_variants.append(templates.render("egraph.rs.ISA_ENUM_VARIANTS.variant.txt",
                                             variant_name=variant_name,
                                             signature=signature))

        # Num children arm
        num_children_arms.append(templates.render("egraph.rs.ISA_NUM_CHILDREN_MATCH_ARMS.arm.txt",
                                                  variant_name=variant_name,
                                                  arity=str(info.arity)))

        # Set metadata arm (only for instructions with metadata)
        if info.has_metadata:
            set_metadata_arms.append(templates.render("egraph.rs.ISA_SET_METADATA_MATCH_ARMS.arm.txt",
                                                      variant_name=variant_name))

        # Children arms
        data_pattern = "(_, ids)" if info.has_metadata else "(ids)"
        children_arms.append(templates.render("egraph.rs.ISA_CHILDREN_MATCH_ARMS.arm.txt",
                                             variant_name=variant_name,
                                             data_pattern=data_pattern))
        children_mut_arms.append(templates.render("egraph.rs.ISA_CHILDREN_MUT_MATCH_ARMS.arm.txt",
                                                  variant_name=variant_name,
                                                  data_pattern=data_pattern))

        # From op arms
        if info.has_metadata:
            from_op_arms.append(templates.render("egraph.rs.ISA_FROM_OP_MATCH_ARMS.arm_with_metadata.txt",
                                                 kebab_name=kebab_name,
                                                 arity=str(info.arity),
                                                 variant_name=variant_name))
        else:
            from_op_arms.append(templates.render("egraph.rs.ISA_FROM_OP_MATCH_ARMS.arm_no_metadata.txt",
                                                 kebab_name=kebab_name,
                                                 arity=str(info.arity),
                                                 variant_name=variant_name))

        # Display arms
        if info.has_metadata:
            display_arms.append(templates.render("egraph.rs.ISA_DISPLAY_MATCH_ARMS.arm_with_metadata.txt",
                                                 variant_name=variant_name,
                                                 instruction_name=info.name))
        else:
            display_arms.append(templates.render("egraph.rs.ISA_DISPLAY_MATCH_ARMS.arm_no_metadata.txt",
                                                 variant_name=variant_name,
                                                 instruction_name=info.name))

        # Inplace arm: vec of bool for each instruction
        inplace_vec = ", ".join("true" if v else "false" for v in info.inplace)
        inplace_arms.append(templates.render(
            "egraph.rs.ISA_INPLACE_MATCH_ARMS.arm.txt",
            variant_name=variant_name,
            inplace_vec=inplace_vec))

    return {
        'enum_variants': enum_variants,
        'num_children_arms': num_children_arms,
        'set_metadata_arms': set_metadata_arms,
        'children_arms': children_arms,
        'children_mut_arms': children_mut_arms,
        'from_op_arms': from_op_arms,
        'display_arms': display_arms,
        'inplace_arms': inplace_arms,
    }


def generate_egraph_rs_file(act_dest_dir, instructions):
    """Template egraph.rs from generic file"""
    templates = get_backend_template_loader()
    egraph_file = os.path.join(act_dest_dir, 'src', 'ir', 'egraph.rs')

    with open(egraph_file, 'r') as f:
        content = f.read()

    instructions_info = extract_instruction_info(instructions)
    instruction_parts = generate_instruction_variants(instructions_info, templates)

    replacements = {
        '{{ISA_ENUM_VARIANTS}}': '\n'.join(instruction_parts['enum_variants']),
        '{{ISA_NUM_CHILDREN_MATCH_ARMS}}': '\n'.join(instruction_parts['num_children_arms']),
        '{{ISA_SET_METADATA_MATCH_ARMS}}': '\n'.join(instruction_parts['set_metadata_arms']),
        '{{ISA_CHILDREN_MATCH_ARMS}}': '\n'.join(instruction_parts['children_arms']),
        '{{ISA_CHILDREN_MUT_MATCH_ARMS}}': '\n'.join(instruction_parts['children_mut_arms']),
        '{{ISA_FROM_OP_MATCH_ARMS}}': '\n'.join(instruction_parts['from_op_arms']),
        '{{ISA_DISPLAY_MATCH_ARMS}}': '\n'.join(instruction_parts['display_arms']),
        '{{ISA_INPLACE_MATCH_ARMS}}': '\n'.join(instruction_parts['inplace_arms']),
    }

    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    with open(egraph_file, 'w') as f:
        f.write(content)
