import os
import re
from .taidl.antlr4 import IDLV2Parser, IDLV2Visitor
from .template_loader import get_backend_template_loader


class RustGeneratorVisitor(IDLV2Visitor):
    def __init__(self):
        super().__init__()
        self.assignments = []
        self.root_assignment = None
        self.comp_attributes = set()
        self.variable_definitions = {}
        self.variable_usage = set()

    def visitModule(self, ctx):
        self.visitChildren(ctx)

        duplicates = [a for a in self.assignments if a['lhs'].endswith('_dup')]
        non_duplicates = [a for a in self.assignments if not a['lhs'].endswith('_dup')]
        self.assignments = duplicates + non_duplicates

        if self.root_assignment:
            for assignment in self.assignments:
                if assignment['lhs'] == self.root_assignment['lhs']:
                    self.root_assignment = assignment
                    break

        return {
            'assignments': self.assignments,
            'root_assignment': self.root_assignment,
            'comp_attributes': list(self.comp_attributes)
        }

    def visitInstruction(self, ctx):
        lhs_name = ctx.IDENTIFIER().getText()
        result_type_info = self.visit(ctx.result_type())
        lhs_type, lhs_shape = result_type_info

        is_root = ctx.ROOT() is not None

        self._extract_comp_attrs_from_text(lhs_shape)

        operand_names = []
        operand_texts = []
        if ctx.operands():
            operands = self.visit(ctx.operands())
            for operand_text, operand_type in operands:
                self._extract_comp_attrs_from_text(operand_text)
                operand_texts.append(operand_text)
                if operand_type == 'IDENTIFIER':
                    operand_names.append(operand_text)

        for operand_text in operand_names:
            if operand_text in self.variable_definitions:
                if operand_text in self.variable_usage:
                    self._duplicate_variable_recursive(operand_text)
                else:
                    self.variable_usage.add(operand_text)

        if ctx.attributes():
            attributes = self.visit(ctx.attributes())
            for _, attr_value, attr_type in attributes:
                if attr_type in ['EXPRESSION', 'BRACELIST']:
                    self._extract_comp_attrs_from_text(str(attr_value))

        assignment = {
            'lhs': lhs_name,
            'dtype': lhs_type,
            'shape': lhs_shape,
            'operands': operand_names,
            'operand_texts': operand_texts,
        }

        self.variable_definitions[lhs_name] = len(self.assignments)
        self.assignments.append(assignment)

        if is_root:
            self.root_assignment = assignment

        return assignment

    def _duplicate_variable_recursive(self, var_name):
        dup_name = f"{var_name}_dup"

        if dup_name in self.variable_definitions:
            return

        original = self.assignments[self.variable_definitions[var_name]]

        for operand in original['operands']:
            if operand in self.variable_definitions:
                self._duplicate_variable_recursive(operand)

        duplicate = {
            'lhs': dup_name,
            'dtype': original['dtype'],
            'shape': original['shape'],
            'operands': original['operands']
        }
        self.variable_definitions[dup_name] = len(self.assignments)
        self.assignments.append(duplicate)

    def _extract_comp_attrs_from_text(self, text):
        pattern = r'@c\.(\w+)'
        matches = re.findall(pattern, str(text))
        self.comp_attributes.update(matches)

    def visitResult_type(self, ctx):
        shape_info = self.visit(ctx.shape())
        shape_dims, shape_type = shape_info
        return (shape_type, shape_dims)

    def visitShape(self, ctx):
        dims = ctx.getText()
        typ = ctx.parentCtx.TYPE().getText()
        return (dims, typ)

    def visitOperands(self, ctx):
        operand_list = []
        for operand_ctx in ctx.operand():
            operand_info = self.visit(operand_ctx)
            operand_list.append(operand_info)
        return operand_list

    def visitOperand(self, ctx):
        value_ctx = ctx.value()
        value_text = value_ctx.getText()

        if value_ctx.INT():
            operand_type = 'INT'
        elif value_ctx.IDENTIFIER():
            operand_type = 'IDENTIFIER'
        elif value_ctx.EXPRESSION():
            operand_type = 'EXPRESSION'
        else:
            operand_type = 'UNKNOWN'

        return (value_text, operand_type)

    def visitAttributes(self, ctx):
        attributes_list = []
        for attribute_ctx in ctx.attribute():
            attr_info = self.visit(attribute_ctx)
            attributes_list.append(attr_info)
        return attributes_list

    def visitAttribute(self, ctx):
        attr_name = ctx.IDENTIFIER().getText()
        attr_value_info = self.visit(ctx.attributeValue())
        attr_value, attr_type = attr_value_info
        return (attr_name, attr_value, attr_type)

    def visitAttributeValue(self, ctx):
        if ctx.braceList():
            brace_info = self.visit(ctx.braceList())
            return (brace_info, 'BRACELIST')
        elif ctx.value():
            value_ctx = ctx.value()
            value_text = value_ctx.getText()

            if value_ctx.INT():
                value_type = 'INT'
            elif value_ctx.IDENTIFIER():
                value_type = 'IDENTIFIER'
            elif value_ctx.EXPRESSION():
                value_type = 'EXPRESSION'
            else:
                value_type = 'UNKNOWN'

            return (value_text, value_type)
        else:
            return ("", "UNKNOWN")


def map_dtype_to_rust(dtype):
    dtype_map = {
        'u8': 'Dtype::U8',
        's8': 'Dtype::I8',
        'u16': 'Dtype::U16',
        's16': 'Dtype::I16',
        'u32': 'Dtype::U32',
        's32': 'Dtype::I32',
        'u64': 'Dtype::U64',
        's64': 'Dtype::I64',
        'f16': 'Dtype::FP16',
        'f32': 'Dtype::FP32',
        'f64': 'Dtype::FP64',
        'bf16': 'Dtype::BF16',
        'pred': 'Dtype::Pred',
    }

    rust_type = dtype_map.get(dtype)
    assert (rust_type != None)
    return rust_type


def extract_comp_attr_from_shape(shape_text, comp_attrs):
    # Find the computational attribute that appears in the shape
    for attr in comp_attrs:
        if f'@c.{attr}' in shape_text:
            return attr
    return comp_attrs[0] if comp_attrs else None


def parse_comp_attr_expression(shape_expr, comp_attr):
    # Only works with @c.attr or @c.attr * constant
    # Very fragile but should work
    if not comp_attr or f'@c.{comp_attr}' not in shape_expr:
        return comp_attr, []

    expr = shape_expr.replace('`', '').replace(' ', '')
    pattern = f'@c.{comp_attr}'

    if f'{pattern}*' in expr:
        start = expr.index(pattern) + len(pattern) + 1
        factor_str = ''
        for char in expr[start:]:
            if char.isdigit():
                factor_str += char
            else:
                break
        if factor_str:
            return comp_attr, [int(factor_str)]

    return comp_attr, []


def generate_all_comp_attr_extractions(root_assignment, comp_attrs, all_assignments,
                                        need_check, templates):
    if not comp_attrs or not root_assignment:
        return ""

    num_assignments = len(all_assignments)
    parts = []

    for attr in comp_attrs:
        found = False
        # Case 1: attr appears alone in a shape dimension (e.g. `@c.rows` in `@c.rows,@c.cols`)
        for i, assignment in enumerate(all_assignments):
            shape = assignment['shape']
            if f'@c.{attr}' in shape:
                dim_index = _find_dim_index_for_attr(shape, attr)
                if dim_index < 0:
                    continue
                _, factors = parse_comp_attr_expression(shape, attr)
                division = ""
                if factors:
                    division_factor = 1
                    for f in factors:
                        division_factor *= f
                    division = f" / {division_factor}"
                if need_check:
                    shape_template = "ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.attr_from_shape_checked.txt"
                else:
                    shape_template = "ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.attr_from_shape.txt"
                parts.append(templates.render(shape_template,
                    attr=attr, index=str(i), dim_index=str(dim_index), division=division))
                found = True
                break

        # Case 2: attr appears in an operand text (e.g. constant(`@c.scale`))
        if not found:
            for i, assignment in enumerate(all_assignments):
                for t in assignment.get('operand_texts', []):
                    if f'@c.{attr}' in t:
                        parts.append(templates.render(
                            "ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.attr_from_constant.txt",
                            attr=attr, index=str(i)))
                        found = True
                        break
                if found:
                    break

        if not found:
            root_index = num_assignments - 1
            shape_template = "ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.attr_from_shape_checked.txt" if checked else "ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.attr_from_shape.txt"
            parts.append(templates.render(shape_template,
                attr=attr, index=str(root_index), dim_index="0", division=""))

    return '\n'.join(parts) + '\n' if parts else ""


def _find_dim_index_for_attr(shape_text, attr):
    """Find the dimension index where @c.attr appears alone (not mixed with other @c. refs)."""
    shape = shape_text.replace('`', '').replace(' ', '')
    dims = shape.split(',')
    for i, dim in enumerate(dims):
        if f'@c.{attr}' in dim:
            # Check this dimension only references this one attr
            all_refs = re.findall(r'@c\.\w+', dim)
            if len(all_refs) == 1:
                return i
    return -1


def generate_precond_function(instruction_name, semantics_data, rhs_size, templates):
    if not semantics_data:
        return ""

    assignments = semantics_data['assignments']
    root_assignment = semantics_data['root_assignment']
    comp_attrs = semantics_data['comp_attributes']

    num_assignments = len(assignments)
    # Generate comp attribute extraction
    comp_attr_code = generate_all_comp_attr_extractions(root_assignment, comp_attrs, assignments, True, templates)

    # Generate shape/dtype checks for all assignments
    shape_checks = []
    for i, assignment in enumerate(assignments):
        dtype_rust = map_dtype_to_rust(assignment['dtype'])
        shape_expr = assignment['shape']

        # Parse shape expression to extract dimensions
        shape_expr = shape_expr.replace('`', '').replace(' ', '')

        # Replace all @c.attr patterns with variable names
        for attr in comp_attrs:
            shape_expr = shape_expr.replace(f'@c.{attr}', attr)
        shape_expr = re.sub(r'@c\.(\w+)', r'\1', shape_expr)

        if ',' in shape_expr:
            dims = [d.strip() for d in shape_expr.split(',')]
        else:
            dims = [shape_expr]

        shape_vec = f"vec![{', '.join(dims)}]"

        shape_check = templates.render("ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.shape_check.txt",
                                       index=str(i),
                                       shape_vec=shape_vec,
                                       dtype=dtype_rust
                                       )
        shape_checks.append(shape_check)

    shape_checks_code = '\n'.join(shape_checks)

    return templates.render("ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.precond.txt",
                            instruction_name=instruction_name,
                            num_assignments=str(num_assignments),
                            comp_attr_code=comp_attr_code,
                            shape_checks=shape_checks_code
                            )


def generate_metadata_function(instruction_name, semantics_data, rhs_size, templates):
    if not semantics_data:
        return ""

    assignments = semantics_data['assignments']
    comp_attrs = semantics_data['comp_attributes']
    root_assignment = semantics_data['root_assignment']

    num_assignments = len(assignments)

    if comp_attrs:
        comp_attr_extraction = generate_all_comp_attr_extractions(
            root_assignment, comp_attrs, assignments, False, templates)

        if len(comp_attrs) == 1:
            metadata_expr = f'{comp_attrs[0]}'
        else:
            fmt_str = ','.join(['{}'] * len(comp_attrs))
            fmt_args = ', '.join(comp_attrs)
            metadata_expr = f'format!("{fmt_str}", {fmt_args})'

        return templates.render("ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.metadata_with_attr.txt",
                                instruction_name=instruction_name,
                                num_assignments=str(num_assignments),
                                comp_attr=metadata_expr,
                                comp_attr_extraction=comp_attr_extraction.strip(),
                                rhs_size=str(rhs_size)
                                )
    else:
        return templates.render("ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.metadata_no_attr.txt",
                                instruction_name=instruction_name,
                                num_assignments=str(num_assignments),
                                rhs_size=str(rhs_size)
                                )


def generate_set_shapes_function(instruction_name, semantics_data, rhs_size, templates):
    if not semantics_data:
        return ""

    assignments = semantics_data['assignments']
    root_assignment = semantics_data['root_assignment']
    comp_attrs = semantics_data['comp_attributes']

    num_assignments = len(assignments)

    # Generate metadata extraction - use unified logic same as other functions
    if comp_attrs:
        metadata_code = templates.render(
            "ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.lhs_metadata.txt"
        ) + "\n" + generate_all_comp_attr_extractions(
            root_assignment, comp_attrs, assignments, False, templates)
    else:
        metadata_code = templates.render(
            "ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.lhs_metadata_unused.txt"
        ) + "\n"

    # Generate shape setting
    shape_setting = ""
    if root_assignment:
        dtype_rust = map_dtype_to_rust(root_assignment['dtype'])
        shape_expr = root_assignment['shape']

        # Parse shape expression
        shape_expr = shape_expr.replace('`', '').replace(' ', '')

        # Replace all @c.attr patterns with variable names
        for attr in comp_attrs:
            shape_expr = shape_expr.replace(f'@c.{attr}', attr)
        shape_expr = re.sub(r'@c\.(\w+)', r'\1', shape_expr)

        if ',' in shape_expr:
            dims = [d.strip() for d in shape_expr.split(',')]
        else:
            dims = [shape_expr]

        shape_vec = f"vec![{', '.join(dims)}]"

        shape_setting = templates.render("ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.shape_setting.txt",
                                         shape_vec=shape_vec,
                                         dtype=dtype_rust
                                         )

    return templates.render("ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.set_shapes.txt",
                            instruction_name=instruction_name,
                            num_assignments=str(num_assignments),
                            rhs_size=str(rhs_size),
                            metadata_code=metadata_code,
                            shape_setting=shape_setting
                            )


def generate_ir2isa_rust_functions(metadata, templates):
    visitor = RustGeneratorVisitor()
    semantics_data = visitor.visit(metadata.semantics_ast)
    if not semantics_data:
        return f"// Failed to parse semantics for {metadata.name}\n"

    if metadata.comp_attrs:
        semantics_data['comp_attributes'] = metadata.comp_attrs

    precond = generate_precond_function(metadata.name, semantics_data, metadata.rhs_size, templates)
    metadata_func = generate_metadata_function(
        metadata.name, semantics_data, metadata.rhs_size, templates)
    set_shapes = generate_set_shapes_function(
        metadata.name, semantics_data, metadata.rhs_size, templates)

    return f"{precond}\n\n{metadata_func}\n\n{set_shapes}\n\n"


def generate_ir2isa_rewrites_rs_file(backend_gen_dir: str, instruction_metadata_list):
    """Template ir2isa_rewrites.rs from generic file"""
    templates = get_backend_template_loader()

    # Read generic file
    rust_file = os.path.join(backend_gen_dir, 'src', 'isel', 'rewrites', 'ir2isa_rewrites.rs')
    with open(rust_file, 'r') as f:
        content = f.read()

    # Generate all rewrite functions
    functions = []
    for metadata in instruction_metadata_list:
        func_code = generate_ir2isa_rust_functions(metadata, templates)
        functions.append(func_code)

    # Join functions with newlines
    all_functions = ''.join(functions)

    # Replace placeholder
    content = content.replace('{{IR2ISA_REWRITE_FUNCTIONS}}', all_functions)

    # Write back
    with open(rust_file, 'w') as f:
        f.write(content)
