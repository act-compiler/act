"""TAIDL - Tensor Accelerator ISA Definition Language"""

__version__ = "2.0.0"

from .accelerator import Accelerator
from .instruction import Instruction
from .data_model import DataModel, Constant

__all__ = ['Accelerator', 'Instruction', 'DataModel', 'Constant']
