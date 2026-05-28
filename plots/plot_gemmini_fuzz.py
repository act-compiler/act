"""Plot compilation time breakdown for Gemmini fuzz kernels.

Reads stats from a log file produced by run-gemmini.sh, joins with
kernels/gemmini/summary.csv for HLO node counts, groups kernels by
node count, averages per-module times, and produces a stacked area plot.

Usage:
    python plots/plot_gemmini_fuzz.py --stats compiled_asm/gemmini/fuzz/stats.log
"""

import argparse
import csv
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def parse_stats_log(stats_path):
    """Parse stats log: blocks separated by '=== fuzz_NNN ===' headers."""
    results = {}
    current = None
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^=== (fuzz_\d+) ===$', line)
            if m:
                current = m.group(1)
                results[current] = {}
                continue
            m = re.match(r'\[STATS\] (M[0-9X]_\w+)=([\d.]+)ms', line)
            if m and current:
                results[current][m.group(1)] = float(m.group(2))
    return results


def load_summary(summary_path):
    """Load summary.csv: filename -> total_nodes."""
    node_counts = {}
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['filename'].replace('.gen.hlo', '')
            node_counts[name] = int(row['total_nodes'])
    return node_counts


def aggregate(stats, node_counts, bin_size=10):
    """Bin kernels by node count ranges, average each module per bin."""
    groups = {}
    for name, modules in stats.items():
        n = node_counts.get(name)
        if n is None:
            continue
        bucket = (n // bin_size) * bin_size
        if bucket not in groups:
            groups[bucket] = []
        groups[bucket].append(modules)

    module_keys = ['M2_Rewriter', 'M3_Extractor', 'M5_CSPGen', 'MX_Solver']
    misc_keys = ['M1_Initializer', 'M4_TopoSort', 'M6_Emitter']

    sorted_buckets = sorted(groups.keys())
    out = {k: [] for k in module_keys}
    out['Misc'] = []
    x_nodes = []

    for bucket in sorted_buckets:
        entries = groups[bucket]
        x_nodes.append(bucket + bin_size // 2)
        for k in module_keys:
            vals = [e.get(k, 0) for e in entries]
            out[k].append(np.mean(vals))
        misc_vals = []
        for e in entries:
            m = sum(e.get(k, 0) for k in misc_keys)
            misc_vals.append(m)
        out['Misc'].append(np.mean(misc_vals))

    return x_nodes, out


def plot(x_nodes, data, output_path):
    palette = "tab10"
    colors = sns.color_palette(palette)
    bar_colors = [colors[0], colors[1], colors[2], colors[6], colors[4]]

    labels = {
        'M2_Rewriter': 'Rewrite Applier',
        'M3_Extractor': 'Graph Extractor',
        'M5_CSPGen': 'CSP Generation',
        'MX_Solver': 'CP-SAT Solver',
        'Misc': 'Misc',
    }
    order = ['M2_Rewriter', 'M3_Extractor', 'M5_CSPGen', 'MX_Solver', 'Misc']

    fig, ax = plt.subplots(figsize=(12, 5))

    arrays = [np.array(data[k]) for k in order]
    ax.stackplot(x_nodes, *arrays,
                 labels=[labels[k] for k in order],
                 alpha=0.85, colors=bar_colors)

    ax.set_xlabel('Kernel size (#HLO nodes)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Avg. compilation time (ms)', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(axis='y', zorder=0, alpha=0.4)
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.tick_params(axis='y', which='minor', length=4)
    ax.grid(which='minor', axis='y', linestyle=':', color='gray', alpha=0.3)
    ax.set_xlim(min(x_nodes), max(x_nodes))

    handles, plot_labels = ax.get_legend_handles_labels()
    ax.legend(handles, plot_labels, loc='upper left', fontsize=13,
              frameon=True, framealpha=0.6)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', required=True, help='Path to stats log file')
    parser.add_argument('--summary', default=None, help='Path to summary.csv')
    parser.add_argument('--output', default=None, help='Output PDF path')
    args = parser.parse_args()

    if args.summary is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.summary = os.path.join(root, 'kernels', 'gemmini', 'summary.csv')
    if args.output is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output = os.path.join(root, 'plots', 'gemmini_fuzz_time.pdf')

    stats = parse_stats_log(args.stats)
    print(f"Parsed {len(stats)} kernel stats")

    node_counts = load_summary(args.summary)
    x_nodes, data = aggregate(stats, node_counts)

    print(f"Node count buckets: {x_nodes}")
    for k, v in data.items():
        print(f"  {k}: {[f'{x:.2f}' for x in v]}")

    plot(x_nodes, data, args.output)
