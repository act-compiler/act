"""
NNSmith model generator wrapper.

Calls nnsmith.model_gen to generate random TensorFlow computation graphs.
The output is a GIR (Graph IR) pickle file.

Configuration:
  nnsmith.model_gen model.type=tensorflow backend.type=xla
    mgen.seed=<seed> mgen.max_nodes=<max_nodes>
    mgen.include="[tensorflow.TFMatMul, core.Abs, core.Add, core.Max, core.Min, core.Neg, core.Clip, core.Sub]"

NNSmith op -> XLA-HLO op mapping:
  TFMatMul -> dot
  Abs      -> reverse
  Add      -> add
  Max      -> maximum
  Min      -> minimum
  Neg      -> negate
  Clip     -> clamp
  Sub      -> subtract
"""

import os
import subprocess
import shutil
from pathlib import Path

INCLUDE = "[tensorflow.TFMatMul, core.Abs, core.Add, core.Max, core.Min, core.Neg, core.Clip, core.Sub]"


def run_nnsmith(max_nodes, seed, output_path):
    """
    Run nnsmith.model_gen to generate a random computation graph.

    Args:
        max_nodes: Maximum number of operators in the graph
        seed: Random seed for model generation
        output_path: Directory path for nnsmith output (contains model/gir.pkl)

    Raises:
        RuntimeError: If nnsmith fails or GIR file is not produced
    """
    output_path = Path(output_path).resolve()
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnsmith.model_gen",
        "model.type=tensorflow",
        "backend.type=xla",
        f"mgen.seed={seed}",
        f"mgen.max_nodes={max_nodes}",
        f"mgen.include={INCLUDE}",
        f"model.path={output_path}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nnsmith.model_gen failed (seed={seed}): {result.stderr}")

    gir_path = output_path / "model" / "gir.pkl"
    if not gir_path.exists():
        gir_path = output_path / "gir.pkl"
    if not gir_path.exists():
        raise RuntimeError(f"GIR file not found after nnsmith run in {output_path}")

    return gir_path
