"""Instruction set definitions for TAIDLv2"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from antlr4 import CommonTokenStream, InputStream

from .data_model import DataModel
from .antlr4 import IDLV2Lexer, IDLV2Parser


@dataclass
class Instruction:
    instruction: str
    comp_attr: List[str]
    parameters: List[str]
    constraints: List[str]
    data_models: List[DataModel]
    cost: str
    update: List[str]

    instr_inputs: List[Tuple[str, List[str], List[str]]]
    instr_outputs: List[Tuple[str, List[str], List[str]]]
    instr_semantics_str: str
    instr_semantics_ast: Optional[IDLV2Parser.ModuleContext]

    def __init__(self, instruction: str, comp_attr: List[str], addr_attr: List[str], constraints: List[str], cost: str, update: List[str], data_models: List[DataModel]):
        self.instruction = instruction.replace(" ", "_")
        self.comp_attr = comp_attr
        self.parameters = addr_attr
        self.constraints = constraints
        self.cost = cost
        self.update = update
        self.data_models = data_models
        self.instr_inputs = []
        self.instr_outputs = []
        self.instr_semantics_str = ""
        self.instr_semantics_ast = None
        self.inplace = []

    def find_data_model(self, name: str) -> DataModel:
        for model in self.data_models:
            if (model.var_name == name):
                break
        else:
            model = None
        assert (model != None), f"Data buffer '{name}' not found"
        return model

    def set_inputs(self, inputs: List[Tuple[str, List[str], List[str]]]):
        self.instr_inputs = inputs
        if not self.inplace:
            self.inplace = [False] * len(inputs)

    def set_outputs(self, outputs: List[Tuple[str, List[str], List[str]]]):
        self.instr_outputs = outputs

    def set_inplace(self, inplace_operands: List[bool]):
        """Mark which RHS operands are inplace (overwritten by the instruction).
        Length must match the number of inputs."""
        self.inplace = inplace_operands

    def add_semantics(self, semantics: str):
        self.instr_semantics_str = semantics
        self.instr_semantics_ast = None
        if semantics and semantics.strip():
            self.instr_semantics_ast = self._parse_semantics(semantics)

    def _parse_semantics(self, semantics: str):
        """Parse semantics into AST eagerly to catch syntax errors early"""
        input_stream = InputStream(semantics)
        lexer = IDLV2Lexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = IDLV2Parser(stream)
        tree = parser.module()
        if parser.getNumberOfSyntaxErrors() > 0:
            raise SyntaxError(f"Failed to parse semantics for instruction '{self.instruction}'")
        return tree

    def add_constraints(self, constraints: str):
        lines = constraints.strip().split("\n")
        self.constraints = [line for line in lines if line.strip()]
