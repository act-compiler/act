"""Backend code generator for ACT compiler

This module provides the main interface for generating backend code.
It coordinates all the individual file generators to produce a complete
ACT backend implementation.
"""

import os
from pathlib import Path
import shutil
from typing import List
import subprocess
from dataclasses import dataclass

from .taidl import Accelerator

from .ir_buffer_rs_generator import generate_buffer_rs_file
from .ir_egraph_rs_generator import generate_egraph_rs_file
from .malloc_globals_cc_generator import generate_globals_file
from .malloc_instructions_h_generator import generate_instructions_file
from .malloc_parser_cc_generator import generate_parser_file
from .isel_applier_rs_generator import generate_applier_file
from .malloc_act_malloc_cc_generator import generate_act_malloc_file
from .isel_ir2isa_rewrites_txt_generator import generate_ir2isa_rewrites_txt_file
from .isel_ir2isa_rewrites_rs_generator import generate_ir2isa_rewrites_rs_file


@dataclass
class InstructionMetadata:
    name: str
    semantics_ast: object
    has_comp_attrs: bool
    comp_attrs: List[str]
    rhs_size: int


def _get_instruction_metadata(accelerator: Accelerator) -> List[InstructionMetadata]:
    """Extract instruction metadata needed for rewrite rule generation"""
    metadata_list = []
    for instruction in accelerator.instructions:
        if instruction.instr_semantics_ast:
            metadata = InstructionMetadata(
                name=instruction.instruction,
                semantics_ast=instruction.instr_semantics_ast,
                has_comp_attrs=len(instruction.comp_attr) > 0,
                comp_attrs=list(instruction.comp_attr),
                rhs_size=len(instruction.instr_inputs) + 1
            )
            metadata_list.append(metadata)
    return metadata_list


def generate_backend(accelerator: Accelerator, base_dir: str = None) -> None:
    """
    Generate complete ACT backend code for an accelerator.

    Args:
        accelerator: TAIDL Accelerator object
        base_dir: Base directory of the project. Defaults to accelerator.base_dir.
    """
    if base_dir is None:
        base_dir = accelerator.base_dir

    # Setup directories
    target_dir = os.path.join(base_dir, 'targets', accelerator.name)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    generic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generic')
    backend_gen_dir = os.path.join(target_dir, 'backend')

    # Copy generic backend structure
    if os.path.exists(backend_gen_dir):
        shutil.rmtree(backend_gen_dir)
    shutil.copytree(generic_dir, backend_gen_dir)

    print(f"Copied generic backend structure to {backend_gen_dir}")

    instruction_metadata = _get_instruction_metadata(accelerator)

    # Generate rewrite rules
    generate_ir2isa_rewrites_txt_file(backend_gen_dir, instruction_metadata)
    print(f"Generated ir2isa_rewrites.txt")

    # Generate Rust files
    generate_ir2isa_rewrites_rs_file(backend_gen_dir, instruction_metadata)
    print(f"Generated ir2isa_rewrites.rs")

    generate_buffer_rs_file(backend_gen_dir, accelerator.instructions, accelerator.data_model)
    print(f"Generated buffer.rs")

    generate_egraph_rs_file(backend_gen_dir, accelerator.instructions)
    print(f"Generated egraph.rs")

    generate_applier_file(backend_gen_dir, accelerator.instructions)
    print(f"Generated applier.rs")

    # Generate C++ malloc files
    generate_globals_file(backend_gen_dir, accelerator.data_model)
    print(f"Generated globals.cc")

    generate_instructions_file(backend_gen_dir, accelerator.instructions, accelerator.data_model)
    print(f"Generated instructions.h")

    generate_parser_file(backend_gen_dir, accelerator.instructions)
    print(f"Generated parser.cc")

    generate_act_malloc_file(backend_gen_dir, accelerator.data_model)
    print(f"Generated act_malloc.cc")

    print(f"Backend generation complete for {accelerator.name}")

    # Build the backend
    print(f"Building backend for {accelerator.name}")
    cargo_build_dir = os.path.join(backend_gen_dir, 'target')
    if os.path.exists(cargo_build_dir):
        shutil.rmtree(cargo_build_dir)
    subprocess.run(["cargo", "build", "--release"], cwd=backend_gen_dir,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"Backend build complete for {accelerator.name}")

    # Copy built backend to final destination
    build_backend_path = os.path.join(cargo_build_dir, 'release', 'backend')
    final_dest_path = os.path.join(base_dir, 'backends', accelerator.name)
    if os.path.exists(build_backend_path):
        Path(final_dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(build_backend_path, final_dest_path)
        print(f"Final backend binary located at {final_dest_path}")
    else:
        raise RuntimeError("Backend build failed. Please check the build logs.")
