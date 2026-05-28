# Generated from IDLV2.g4 by ANTLR 4.9.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .IDLV2Parser import IDLV2Parser
else:
    from IDLV2Parser import IDLV2Parser

# This class defines a complete generic visitor for a parse tree produced by IDLV2Parser.

class IDLV2Visitor(ParseTreeVisitor):

    # Visit a parse tree produced by IDLV2Parser#module.
    def visitModule(self, ctx:IDLV2Parser.ModuleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#instruction.
    def visitInstruction(self, ctx:IDLV2Parser.InstructionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#result_type.
    def visitResult_type(self, ctx:IDLV2Parser.Result_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#operands.
    def visitOperands(self, ctx:IDLV2Parser.OperandsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#operand.
    def visitOperand(self, ctx:IDLV2Parser.OperandContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#attributes.
    def visitAttributes(self, ctx:IDLV2Parser.AttributesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#attribute.
    def visitAttribute(self, ctx:IDLV2Parser.AttributeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#attributeValue.
    def visitAttributeValue(self, ctx:IDLV2Parser.AttributeValueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#braceList.
    def visitBraceList(self, ctx:IDLV2Parser.BraceListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#sliceList.
    def visitSliceList(self, ctx:IDLV2Parser.SliceListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#shape.
    def visitShape(self, ctx:IDLV2Parser.ShapeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by IDLV2Parser#value.
    def visitValue(self, ctx:IDLV2Parser.ValueContext):
        return self.visitChildren(ctx)



del IDLV2Parser