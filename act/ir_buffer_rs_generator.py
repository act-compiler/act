import os
from typing import List
from dataclasses import dataclass
from .template_loader import get_backend_template_loader


@dataclass
class InstructionInfo:
    name: str
    arity: int
    has_metadata: bool
    buffer_assignment: str


def get_rust_variant_name(instr_name: str) -> str:
    words = instr_name.replace('-', '_').split('_')
    return ''.join(word.capitalize() for word in words)


def get_buffer_name(data_model_name: str) -> str:
    return "HBM" if data_model_name == "d0" else data_model_name.upper()


def extract_instruction_info(instructions) -> List[InstructionInfo]:
    info_list = []

    for instruction in instructions:
        name = instruction.instruction
        arity = len(instruction.instr_inputs)
        has_metadata = len(instruction.comp_attr) > 0

        buffers = []
        for output in instruction.instr_outputs:
            buffer_name = get_buffer_name(output[0])
            buffers.append(f"Buffer::{buffer_name}")

        for input_item in instruction.instr_inputs:
            buffer_name = get_buffer_name(input_item[0])
            buffers.append(f"Buffer::{buffer_name}")

        buffer_assignment = f"Some(vec![{', '.join(buffers)}])" if buffers else "None"

        info_list.append(InstructionInfo(name, arity, has_metadata, buffer_assignment))

    return info_list


def generate_buffer_variants(data_models, templates):
    """Generate buffer variants using templates"""
    variants = []
    display_arms = []

    for data_model in data_models:
        if data_model.var_name != "d0":
            buffer_name = data_model.var_name.upper()
            variants.append(templates.render("buffer.rs.BUFFER_VARIANTS.variant.txt", 
                                            buffer_name=buffer_name))
            display_arms.append(templates.render("buffer.rs.BUFFER_DISPLAY_MATCH_ARMS.arm.txt",
                                                 buffer_name=buffer_name))

    return variants, display_arms


def generate_buffer_assignment_arms(instructions_info: List[InstructionInfo], templates):
    """Generate buffer assignment arms using templates"""
    buffer_assignment_arms = []

    for info in instructions_info:
        variant_name = get_rust_variant_name(info.name)
        pattern = "(_, _)" if info.has_metadata else "(_)"
        buffer_assignment_arms.append(
            templates.render("buffer.rs.ISA_BUFFER_ASSIGNMENT_MATCH_ARMS.arm.txt",
                           variant_name=variant_name,
                           pattern=pattern,
                           buffer_assignment=info.buffer_assignment))

    return buffer_assignment_arms


def generate_buffer_rs_file(act_dest_dir, instructions, data_models):
    """Template buffer.rs from generic file"""
    templates = get_backend_template_loader()
    buffer_file = os.path.join(act_dest_dir, 'src', 'ir', 'buffer.rs')

    with open(buffer_file, 'r') as f:
        content = f.read()

    buffer_variants, buffer_display_arms = generate_buffer_variants(data_models, templates)
    instructions_info = extract_instruction_info(instructions)
    buffer_assignment_arms = generate_buffer_assignment_arms(instructions_info, templates)

    replacements = {
        '{{BUFFER_VARIANTS}}': '\n'.join(buffer_variants),
        '{{BUFFER_DISPLAY_MATCH_ARMS}}': '\n'.join(buffer_display_arms),
        '{{ISA_BUFFER_ASSIGNMENT_MATCH_ARMS}}': '\n'.join(buffer_assignment_arms)
    }

    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    with open(buffer_file, 'w') as f:
        f.write(content)
