# Generated from IDLV2.g4 by ANTLR 4.9.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .IDLV2Parser import IDLV2Parser
else:
    from IDLV2Parser import IDLV2Parser

# This class defines a complete listener for a parse tree produced by IDLV2Parser.
class IDLV2Listener(ParseTreeListener):

    # Enter a parse tree produced by IDLV2Parser#module.
    def enterModule(self, ctx:IDLV2Parser.ModuleContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#module.
    def exitModule(self, ctx:IDLV2Parser.ModuleContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#instruction.
    def enterInstruction(self, ctx:IDLV2Parser.InstructionContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#instruction.
    def exitInstruction(self, ctx:IDLV2Parser.InstructionContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#result_type.
    def enterResult_type(self, ctx:IDLV2Parser.Result_typeContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#result_type.
    def exitResult_type(self, ctx:IDLV2Parser.Result_typeContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#operands.
    def enterOperands(self, ctx:IDLV2Parser.OperandsContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#operands.
    def exitOperands(self, ctx:IDLV2Parser.OperandsContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#operand.
    def enterOperand(self, ctx:IDLV2Parser.OperandContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#operand.
    def exitOperand(self, ctx:IDLV2Parser.OperandContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#attributes.
    def enterAttributes(self, ctx:IDLV2Parser.AttributesContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#attributes.
    def exitAttributes(self, ctx:IDLV2Parser.AttributesContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#attribute.
    def enterAttribute(self, ctx:IDLV2Parser.AttributeContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#attribute.
    def exitAttribute(self, ctx:IDLV2Parser.AttributeContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#attributeValue.
    def enterAttributeValue(self, ctx:IDLV2Parser.AttributeValueContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#attributeValue.
    def exitAttributeValue(self, ctx:IDLV2Parser.AttributeValueContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#braceList.
    def enterBraceList(self, ctx:IDLV2Parser.BraceListContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#braceList.
    def exitBraceList(self, ctx:IDLV2Parser.BraceListContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#sliceList.
    def enterSliceList(self, ctx:IDLV2Parser.SliceListContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#sliceList.
    def exitSliceList(self, ctx:IDLV2Parser.SliceListContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#shape.
    def enterShape(self, ctx:IDLV2Parser.ShapeContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#shape.
    def exitShape(self, ctx:IDLV2Parser.ShapeContext):
        pass


    # Enter a parse tree produced by IDLV2Parser#value.
    def enterValue(self, ctx:IDLV2Parser.ValueContext):
        pass

    # Exit a parse tree produced by IDLV2Parser#value.
    def exitValue(self, ctx:IDLV2Parser.ValueContext):
        pass



del IDLV2Parser