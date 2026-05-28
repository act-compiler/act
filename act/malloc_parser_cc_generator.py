import os
from .template_loader import get_backend_template_loader


def to_camel_case(snake_str):
    return ''.join(word.capitalize() for word in snake_str.split('_'))


def generate_parser_instruction(instruction, templates):
    op_name = instruction.instruction
    class_name = to_camel_case(op_name)
    comp_attrs = instruction.comp_attr

    if len(comp_attrs) == 0:
        return templates.render("parser.cc.INSTRUCTION_CASES.case_no_attr.txt",
                                op_name=op_name,
                                ClassName=class_name
                                )
    elif len(comp_attrs) == 1:
        return templates.render("parser.cc.INSTRUCTION_CASES.case_with_attr.txt",
                                op_name=op_name,
                                ClassName=class_name,
                                attr_name=comp_attrs[0]
                                )
    else:
        attr_decls = '\n'.join(
            templates.render("parser.cc.INSTRUCTION_CASES.multi_attr_decl.txt", attr_name=a)
            for a in comp_attrs
        )
        attr_parses = '\n'.join(
            templates.render("parser.cc.INSTRUCTION_CASES.multi_attr_parse.txt", attr_name=a)
            for a in comp_attrs
        )
        attr_args = ", ".join(comp_attrs)
        return templates.render("parser.cc.INSTRUCTION_CASES.case_with_multi_attr.txt",
                                op_name=op_name,
                                ClassName=class_name,
                                attr_decls=attr_decls,
                                attr_parses=attr_parses,
                                attr_args=attr_args
                                )


def generate_parser_instructions(instructions, templates):
    cases = []
    for instruction in instructions:
        cases.append(generate_parser_instruction(instruction, templates))

    return ''.join(cases)


def generate_parser_file(act_dest_dir, instructions):
    parser_file = os.path.join(act_dest_dir, 'cpp', 'malloc', 'src', 'parser.cc')

    templates = get_backend_template_loader()

    with open(parser_file, 'r') as f:
        content = f.read()

    instruction_cases = generate_parser_instructions(instructions, templates)

    content = content.replace('{{INSTRUCTION_CASES}}', instruction_cases)

    with open(parser_file, 'w') as f:
        f.write(content)
