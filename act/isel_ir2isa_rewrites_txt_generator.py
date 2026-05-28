import os
from .taidl.antlr4 import IDLV2Parser, IDLV2Visitor
from .template_loader import get_backend_template_loader


class RewriteRuleVisitor(IDLV2Visitor):
    OP_MAP = {
        'reshape': 'reshape_?', 'convert': 'convert_?', 'slice': 'slice_?',
        'transpose': 'transpose_?', 'broadcast': 'broadcast_?', 'concat': 'concat_?',
        'constant': 'constant_?', 'eye': 'eye_?', 'compare': 'compare_?',
        'bitcast_convert': 'bitcvt',
        'reduce_add': 'reduce_?',
        'shift_left': 'shift-left', 'shift_right_logical': 'shift-right-logical',
        'maximum': 'maximum', 'minimum': 'minimum', 'negate': 'negate',
        'subtract': 'subtract', 'multiply': 'multiply', 'clamp': 'clamp',
        'reverse': 'reverse_?',
    }

    def __init__(self):
        super().__init__()
        self.pattern_stack = []
        self.variable_counter = 0
        self.input_variables = {}
        self.parameter_variables = {}
        self.variable_definitions = {}
        self.root_variable = None
        self.variable_usage_count = set()

    def get_next_variable(self):
        var_name = f"?{chr(ord('a') + self.variable_counter)}"
        self.variable_counter += 1
        return var_name

    def substitute_params(self, pattern, substitutions):
        result = pattern
        for old_var, new_var in substitutions.items():
            result = result.replace(old_var, new_var)
        return result

    def visitModule(self, ctx: IDLV2Parser.ModuleContext):
        self.visitChildren(ctx)
        if self.root_variable and self.root_variable in self.variable_definitions:
            return self.variable_definitions[self.root_variable]
        elif self.pattern_stack:
            return self.pattern_stack[-1]
        return None

    def get_parameter_variables(self):
        sorted_params = sorted(self.parameter_variables.items(), key=lambda x: x[0])
        return [var for param_num, var in sorted_params]

    def visitInstruction(self, ctx: IDLV2Parser.InstructionContext):
        lhs_name = ctx.IDENTIFIER().getText()
        op_name = ctx.OPERATION().getText()

        is_root = ctx.ROOT() is not None
        if is_root:
            self.root_variable = lhs_name

        if op_name == 'parameter':
            if ctx.operands():
                operand_info = self.visit(ctx.operands())
                if operand_info:
                    param_num = int(operand_info[0][0])
                    var = self.get_next_variable()
                    self.parameter_variables[param_num] = var
                    self.variable_definitions[lhs_name] = var
                    return var

        # Check if this instruction has any IDENTIFIER operands (references to other SSA values).
        # Leaf ops like constant(`@c.scale`) have no identifier operands -- they should
        # become variables in the pattern to avoid egg's 0-arity metadata matching issue.
        has_identifier_operand = False
        operand_info = []
        if ctx.operands():
            operand_info = self.visit(ctx.operands())
            for operand_value, operand_type in operand_info:
                if operand_type == 'IDENTIFIER' and operand_value in self.variable_definitions:
                    has_identifier_operand = True
                    break

        if not has_identifier_operand and not is_root:
            var = self.get_next_variable()
            self.variable_definitions[lhs_name] = var
            self.pattern_stack.append(var)
            return var

        operand_patterns = []
        for operand_value, operand_type in operand_info:
            if operand_type == 'IDENTIFIER':
                if operand_value in self.variable_definitions:
                    pattern = self.variable_definitions[operand_value]

                    if operand_value in self.variable_usage_count:
                        substitutions = {}
                        for param_var in self.parameter_variables.values():
                            if param_var in pattern:
                                fresh_var = self.get_next_variable()
                                substitutions[param_var] = fresh_var
                        if substitutions:
                            pattern = self.substitute_params(pattern, substitutions)
                    else:
                        self.variable_usage_count.add(operand_value)

                    operand_patterns.append(pattern)
                else:
                    var = self.get_next_variable()
                    operand_patterns.append(var)

        rewrite_op_name = self.OP_MAP.get(op_name, op_name)
        pattern = f"({rewrite_op_name} {' '.join(operand_patterns)})" if operand_patterns else f"({rewrite_op_name})"

        self.variable_definitions[lhs_name] = pattern
        self.pattern_stack.append(pattern)

        return pattern

    def visitOperands(self, ctx: IDLV2Parser.OperandsContext):
        operand_list = []
        for operand_ctx in ctx.operand():
            operand_info = self.visit(operand_ctx)
            operand_list.append(operand_info)
        return operand_list

    def visitOperand(self, ctx: IDLV2Parser.OperandContext):
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

    def visitAttributes(self, ctx: IDLV2Parser.AttributesContext):
        attributes_list = []
        for attribute_ctx in ctx.attribute():
            attr_info = self.visit(attribute_ctx)
            attributes_list.append(attr_info)
        return attributes_list

    def visitAttribute(self, ctx: IDLV2Parser.AttributeContext):
        attr_name = ctx.IDENTIFIER().getText()
        attr_value_info = self.visit(ctx.attributeValue())
        attr_value, attr_type = attr_value_info
        return (attr_name, attr_value, attr_type)

    def visitAttributeValue(self, ctx: IDLV2Parser.AttributeValueContext):
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


def generate_rewrite_rule(metadata, templates):
    """Generate a single rewrite rule using template"""
    visitor = RewriteRuleVisitor()
    lhs_pattern = visitor.visit(metadata.semantics_ast)
    parameter_vars = visitor.get_parameter_variables()
    kebab_name = metadata.name.replace('_', '-')

    # Fallback for unparseable semantics
    if not (lhs_pattern and parameter_vars):
        # Use fixed padding for fallback
        padding = ' ' * (19 - len(kebab_name))
        return templates.render(
            "ir2isa_rewrites.txt.IR2ISA_REWRITE_RULES.rule.txt",
            kebab_name=kebab_name,
            padding=padding,
            lhs_pattern="(?fallback)",
            rhs=f"({kebab_name} ?args)"
        )

    # Generate RHS with or without _? suffix
    rhs = f"({kebab_name}_? {' '.join(parameter_vars)})" if metadata.has_comp_attrs else f"({kebab_name} {' '.join(parameter_vars)})"

    # Calculate padding to align the '=>' at column 20
    padding = ' ' * (19 - len(kebab_name))

    return templates.render(
        "ir2isa_rewrites.txt.IR2ISA_REWRITE_RULES.rule.txt",
        kebab_name=kebab_name,
        padding=padding,
        lhs_pattern=lhs_pattern,
        rhs=rhs
    )


def generate_ir2isa_rewrites_txt_file(backend_gen_dir: str, instruction_metadata_list):
    """Template ir2isa_rewrites.txt from generic file"""
    templates = get_backend_template_loader()

    # Read generic file
    rewrites_file = os.path.join(backend_gen_dir, 'src', 'isel', 'rewrites', 'ir2isa_rewrites.txt')
    with open(rewrites_file, 'r') as f:
        content = f.read()

    # Generate all rewrite rules
    rules = []
    for metadata in instruction_metadata_list:
        rule = generate_rewrite_rule(metadata, templates)
        rules.append(rule)

    # Join rules with newlines
    all_rules = '\n'.join(rules)

    # Replace placeholder
    content = content.replace('{{IR2ISA_REWRITE_RULES}}', all_rules)

    # Write back
    with open(rewrites_file, 'w') as f:
        f.write(content)
