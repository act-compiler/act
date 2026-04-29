"""
Summarize generated fuzz HLO files into a CSV.

Reads all .gen.hlo files from a directory, counts per-op occurrences
and total nodes, and writes summary.csv.
"""

import os
import re
import csv
import argparse
from collections import Counter

HEADER_LINES = 11  # HloModule + blank + reduce_add{3} + blank + ENTRY + reduce_init + closing }

ALL_OPS = ["parameter", "broadcast", "constant", "dot", "minimum", "maximum",
           "clamp", "reverse", "negate", "add", "subtract", "reduce"]


def analyze_hlo(hlo_path):
    """Count total nodes and per-op breakdown for one HLO file."""
    with open(hlo_path) as f:
        lines = f.readlines()

    total_nodes = len(lines) - HEADER_LINES
    op_counts = Counter()

    for line in lines[HEADER_LINES-1:-1]:
        line = line.strip()
        if line.startswith("ROOT"):
            line = line[4:].strip()
        m = re.match(r'\S+ = \S+ (\S+)[\(]', line)
        if m:
            op_counts[m.group(1)] += 1

    return total_nodes, op_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize fuzz HLO files")
    parser.add_argument("--input_dir", required=True, help="Directory with .gen.hlo files")
    parser.add_argument("--output_csv", default="summary.csv", help="Output CSV path")
    args = parser.parse_args()

    hlo_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.gen.hlo')])
    if not hlo_files:
        print(f"No .gen.hlo files found in {args.input_dir}")
        exit(1)

    # Analyze each file
    rows = []
    global_ops = Counter()

    for hlo_file in hlo_files:
        hlo_path = os.path.join(args.input_dir, hlo_file)
        total_nodes, op_counts = analyze_hlo(hlo_path)
        global_ops += op_counts

        row = {"filename": hlo_file, "total_nodes": total_nodes}
        for op in ALL_OPS:
            row[op] = op_counts.get(op, 0)
        rows.append(row)

    # Write CSV
    fieldnames = ["filename", "total_nodes"] + ALL_OPS
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"Analyzed {len(hlo_files)} HLO files -> {args.output_csv}")
    print(f"\nPer-op totals across all files:")
    for op in ALL_OPS:
        count = global_ops.get(op, 0)
        files_with = sum(1 for r in rows if r[op] > 0)
        print(f"  {op:12s}: {count:5d} instances in {files_with:3d}/{len(hlo_files)} files")

    total_nodes = sum(r["total_nodes"] for r in rows)
    print(f"\nTotal nodes across all files: {total_nodes}")
    print(f"Average nodes per file: {total_nodes / len(hlo_files):.1f}")
