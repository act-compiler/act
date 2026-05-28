"""Generator for instructions.h - templates the generic file with instruction classes"""

import os
import re
from .template_loader import get_backend_template_loader


def to_camel_case(snake_str):
    return ''.join(word.capitalize() for word in snake_str.split('_'))


def extract_addr_var(expr):
    match = re.search(r'@a\.(\w+)', expr)
    return match.group(1) if match else None


def extract_size_expr(expr):
    if isinstance(expr, int):
        return str(expr)
    result = re.sub(r'@c\.', '', str(expr))
    return result.strip()


def generate_get_h_method(index, addr_vars, size_exprs, templates):
    range_pairs = []
    for addr_var, size_expr in zip(addr_vars, size_exprs):
        # Literal integer addresses (e.g. 0) need to be wrapped as known constants
        if addr_var.isdigit() or (addr_var.startswith('-') and addr_var[1:].isdigit()):
            addr_expr = f"MAKE_INT_KNOWN({addr_var})"
            range_pairs.append(f"{{{addr_expr}, MAKE_INT_KNOWN({addr_var} + {size_expr})}}")
        else:
            range_pairs.append(f"{{{addr_var}, MAKE_SUM({addr_var}, {size_expr})}}")
    ranges = ", ".join(range_pairs)
    return templates.render("instructions.h.INSTRUCTION_CLASSES.get_h_method.txt",
                            index=index,
                            ranges=ranges)


def generate_instruction_class(instruction, templates):
    """Generate C++ instruction class from instruction definition"""
    class_name = to_camel_case(instruction.instruction)
    op_name = instruction.instruction
    comp_attrs = instruction.comp_attr
    params = instruction.parameters

    # Generate members
    comp_attr_members = ""
    for attr in comp_attrs:
        comp_attr_members += f"  int64 {attr};\n"

    addr_var_members = "\n"
    for param in params:
        addr_var_members += f"  IntVar *{param} = MAKE_INT_VAR(INT64_MIN, INT64_MAX);\n"

    # Generate get_h methods: output first (get_h0), then inputs (get_h1, get_h2, ...)
    get_h_methods = ""
    get_h_cases = ""
    h_index = 0

    # Process outputs
    for output in instruction.instr_outputs:
        _, addr_expr_list, size_expr_list = output
        addr_vars = [extract_addr_var(a) if isinstance(a, str) and '@a.' in str(a) else str(a)
                     for a in addr_expr_list]
        size_exprs = [extract_size_expr(s) for s in size_expr_list]

        if any(a for a in addr_vars if a and a != '0'):
            get_h_methods += generate_get_h_method(h_index, addr_vars, size_exprs, templates)
            get_h_cases += templates.render(
                "instructions.h.INSTRUCTION_CLASSES.get_h_case.txt", index=h_index)
            h_index += 1

    # Process inputs
    for inp in instruction.instr_inputs:
        _, addr_expr_list, size_expr_list = inp
        addr_vars = [extract_addr_var(a) if isinstance(a, str) and '@a.' in str(a) else str(a)
                     for a in addr_expr_list]
        size_exprs = [extract_size_expr(s) for s in size_expr_list]

        if any(a for a in addr_vars if a and a != '0'):
            get_h_methods += generate_get_h_method(h_index, addr_vars, size_exprs, templates)
            get_h_cases += templates.render(
                "instructions.h.INSTRUCTION_CLASSES.get_h_case.txt", index=h_index)
            h_index += 1

    # Generate constructor parameters
    comp_attr_params = ""
    if comp_attrs:
        comp_attr_params = ", " + ", ".join(f"int64 {attr}" for attr in comp_attrs)

    # Generate constructor initialization
    comp_attr_init = ""
    if comp_attrs:
        comp_attr_init = ", " + ", ".join(f"{attr}({attr})" for attr in comp_attrs)

    # No validations needed
    validations = ""
    str_bound_checks = ""

    str_format = ""
    for attr in comp_attrs:
        if str_format:
            str_format += ", "
        str_format += f"{attr} = \" + std::to_string({attr}) + \""
    for param in params:
        if str_format:
            str_format += ", "
        str_format += f"{param} = \" + std::to_string({param}->Value()) + \""

    # Generate get_int_var list
    get_int_var_list = ", ".join(params)

    # Generate inplace override for all instructions
    inplace_vec = ", ".join("true" if v else "false" for v in instruction.inplace)
    inplace_override = templates.render("instructions.h.INSTRUCTION_CLASSES.inplace_override.txt",
                                        INPLACE_VEC=inplace_vec)

    # Generate complete class using template
    return templates.render("instructions.h.INSTRUCTION_CLASSES.class.txt",
                            CLASS_NAME=class_name,
                            OP_NAME=op_name,
                            COMP_ATTR_MEMBERS=comp_attr_members,
                            ADDR_VAR_MEMBERS=addr_var_members,
                            GET_H_METHODS=get_h_methods,
                            COMP_ATTR_PARAMS=comp_attr_params,
                            COMP_ATTR_INIT=comp_attr_init,
                            CONSTRUCTOR_VALIDATIONS=validations,
                            STR_BOUND_CHECKS=str_bound_checks,
                            STR_FORMAT=str_format,
                            GET_H_CASES=get_h_cases,
                            GET_INT_VAR_LIST=get_int_var_list,
                            INPLACE_OVERRIDE=inplace_override)


def generate_instructions_file(backend_gen_dir, instructions, data_models):
    """Template instructions.h from generic file"""
    templates = get_backend_template_loader()

    # Read generic template file
    generic_file = os.path.join(backend_gen_dir, 'cpp', 'malloc', 'include', 'instructions.h')
    with open(generic_file, 'r') as f:
        content = f.read()

    # Generate all instruction classes
    classes = ""
    for instruction in instructions:
        if hasattr(instruction, 'instr_inputs') and hasattr(instruction, 'instr_outputs'):
            classes += generate_instruction_class(instruction, templates) + "\n"

    # Replace placeholder
    content = content.replace('{{INSTRUCTION_CLASSES}}', classes)

    # Write back
    with open(generic_file, 'w') as f:
        f.write(content)
