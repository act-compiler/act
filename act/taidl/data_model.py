"""Data model definitions for TAIDLv2"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Constant:
    const_name: str
    value: int


@dataclass
class DataModel:
    var_name: str
    var_dim: List[int]
    access_dim: List[int]
    unit_dim: List[int]
    var_type: str
    array_dim_str: str = field(init=False)
    num_dim_str: str = field(init=False)

    def __init__(self, var_name: str, access_dim: List[int], unit_dim: List[int], var_type: str = "s8"):
        self.var_name = var_name
        self.var_dim = access_dim + unit_dim
        self.access_dim = access_dim
        self.unit_dim = unit_dim
        self.var_type = var_type
        self.array_dim_str = self.create_arr_str(self.var_dim)
        self.num_dim_str = self.create_num_str()

    def create_arr_str(self, dims) -> str:
        output = '['
        for dim in dims:
            output += f"'{dim}',"
        output = output[:-1]
        output += ']'
        return output

    def create_num_str(self) -> str:
        output = '{'
        for i in range(len(self.var_dim) - 1, -1, -1):
            if (i == 0):
                output += str(i)
            else:
                output += str(i) + ","
        output += "}"
        return output
