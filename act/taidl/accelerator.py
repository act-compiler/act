"""Main Accelerator class for TAIDLv2"""

import os
import inspect
from dataclasses import dataclass, field
from typing import List

from .data_model import Constant, DataModel
from .instruction import Instruction


@dataclass
class Accelerator:
    name: str
    constants: List[Constant]
    data_model: List[DataModel]
    state: List[Constant]
    instructions: List[Instruction] = field(init=False)
    base_dir: str = field(init=False)

    def __init__(self, name: str, base_dir: str = None):
        self.name = name.replace(" ", "_")
        self.constants = []
        self.instructions = []
        self.state = []
        self.data_model = []
        if base_dir is not None:
            self.base_dir = base_dir
        else:
            caller_file = inspect.stack()[1].filename
            if os.path.exists(caller_file):
                self.base_dir = os.path.dirname(os.path.abspath(caller_file))
            else:
                self.base_dir = os.getcwd()
        self.add_data_model("d0", [], [], "u8")

    def add_data_model(self, model_name: str, dimensions: List[int], unit_dim: List[int], var_type: str) -> None:
        self.data_model.append(DataModel(model_name, dimensions, unit_dim, var_type))

    def add_instruction(self, instruction: str, computation_attr: List[str], addressing_attr: List[str], cost: str = "0", update: List[str] = [], constraints: List[str] = []) -> Instruction:
        new_instruction = Instruction(instruction, computation_attr,
                                      addressing_attr, constraints, cost, update, self.data_model)
        self.instructions.append(new_instruction)
        return new_instruction
