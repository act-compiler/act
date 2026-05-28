import os
from .template_loader import get_backend_template_loader


def generate_buffer_names(data_models, templates):
    """Generate buffer names using templates"""
    buffer_names = []
    for dm in data_models:
        if dm.var_name != "d0":
            buffer_name = dm.var_name.upper()
            buffer_names.append(templates.render("act_malloc.cc.BUFFER_NAMES.name.txt",
                                                 buffer_name=f'"{buffer_name}"'))
    return ', '.join(buffer_names)


def generate_act_malloc_file(act_dest_dir, data_models):
    """Template act_malloc.cc from generic file"""
    templates = get_backend_template_loader()
    file_path = os.path.join(act_dest_dir, 'cpp', 'malloc', 'src', 'act_malloc.cc')

    with open(file_path, 'r') as f:
        content = f.read()

    buffer_names = generate_buffer_names(data_models, templates)
    content = content.replace('{{BUFFER_NAMES}}', buffer_names)

    with open(file_path, 'w') as f:
        f.write(content)
