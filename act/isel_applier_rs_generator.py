import os


def generate_function_mappings(instructions):
    """Generate the match arms for precond, metadata, and set_shapes functions"""
    mappings = {'precond': [], 'metadata': [], 'set_shapes': []}

    for instruction in instructions:
        name = instruction.instruction
        kebab_name = name.replace('_', '-')

        mappings['precond'].append(f'        "{kebab_name}" => precond_{name},')
        mappings['metadata'].append(f'        "{kebab_name}" => metadata_{name},')
        mappings['set_shapes'].append(f'        "{kebab_name}" => set_shapes_{name},')

    return mappings['precond'], mappings['metadata'], mappings['set_shapes']


def generate_applier_file(act_dest_dir: str, instructions) -> None:
    """Template the applier.rs file with ISA-specific function mappings"""
    applier_file = os.path.join(act_dest_dir, 'src', 'isel', 'rewrites', 'applier.rs')

    with open(applier_file, 'r') as f:
        content = f.read()

    precond_arms, metadata_arms, set_shapes_arms = generate_function_mappings(instructions)

    # Replace template placeholders
    replacements = {
        '{{ISA_PRECOND_MATCH_ARMS}}': '\n'.join(precond_arms),
        '{{ISA_METADATA_MATCH_ARMS}}': '\n'.join(metadata_arms),
        '{{ISA_SET_SHAPES_MATCH_ARMS}}': '\n'.join(set_shapes_arms)
    }

    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    with open(applier_file, 'w') as f:
        f.write(content)
