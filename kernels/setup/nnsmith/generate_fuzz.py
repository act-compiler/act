"""
Generate fuzz HLO files for Gemmini using NNSmith.

Pipeline: NNSmith -> GIR pickle -> GIRGraph -> XLA-HLO IR
"""

import os
import re
import argparse
import shutil
import random

from nnsmith_runner import run_nnsmith
from gir_graph import GIRGraph
from hlo_emitter import generate_hlo
from validate_hlo import validate_hlo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def count_hlo_nodes(hlo_path):
    """Count total HLO nodes (all lines inside ENTRY except reduce_init).

    Every HLO file has the same structure: 10 header lines (HloModule, reduce_add region,
    ENTRY, reduce_init) and 1 closing brace. Total nodes = total lines - HEADER_LINES.
    """
    HEADER_LINES = 11  # HloModule + blank + reduce_add{3} + blank + ENTRY + reduce_init + closing }
    with open(hlo_path) as f:
        total_lines = len(f.readlines())
    return total_lines - HEADER_LINES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fuzz HLO files using NNSmith")
    parser.add_argument("--output_dir", required=True, help="Output directory for .gen.hlo files")
    args = parser.parse_args()

    # Ensure we run from the script directory so NNSmith outputs land here
    os.chdir(SCRIPT_DIR)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    nnsmith_tmp = os.path.join(SCRIPT_DIR, "nnsmith_output")

    random.seed(0)  # For reproducibility of random seeds
    # 100 configurations: (nnsmith_nodes, nnsmith_seed, random_seed)
    CONFIGS = []
    for i in range(100):
        nnsmith_nodes = 5 + (i // 2)  # linearly from 5 to 55
        CONFIGS.append((nnsmith_nodes, random.randint(0, 10000), random.randint(0, 10000)))

    generated = 0
    failed = 0

    for idx, (nnsmith_nodes, nnsmith_seed, random_seed) in enumerate(CONFIGS):
        output_path = os.path.join(args.output_dir, f"fuzz_{idx:03d}.gen.hlo")

        try:
            gir_path = run_nnsmith(nnsmith_nodes, nnsmith_seed, nnsmith_tmp)
            gir_graph = GIRGraph(gir_path, seed=random_seed)

            if generate_hlo(gir_graph, output_path):
                ok, _, err = validate_hlo(output_path)
                if ok:
                    generated += 1
                    nodes = count_hlo_nodes(output_path)
                    print(
                        f"[{idx+1:03d}/{len(CONFIGS):03d}] nnsmith_nodes={nnsmith_nodes:2d}, nnsmith_seed={nnsmith_seed:4d}, random_seed={random_seed:4d} -> {nodes:3d} nodes")
                else:
                    os.remove(output_path)
                    failed += 1
                    print(f"[{idx+1:03d}/{len(CONFIGS):03d}] Validation failed: {err}")
            else:
                failed += 1
                print(f"[{idx+1:03d}/{len(CONFIGS):03d}] HLO generation failed")
        except Exception as e:
            failed += 1
            print(f"[{idx+1:03d}/{len(CONFIGS):03d}] Error: {e}")

    if os.path.exists(nnsmith_tmp):
        shutil.rmtree(nnsmith_tmp)

    print(f"\nFuzz Summary:")
    print(f"  Generated HLO: {generated}")
    print(f"  Failed: {failed}")
    print(f"  Total configs: {len(CONFIGS)}")
