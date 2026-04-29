#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../../gemmini/fuzz"

echo "=== NNSmith Fuzz Kernel Setup ==="
echo "Output dir: $OUTPUT_DIR"
echo

# Generate fuzz HLO files using NNSmith
# Pipeline: NNSmith -> GIR pickle -> GIRGraph -> XLA-HLO IR (s8[16,16] tiles)

rm -rf "$SCRIPT_DIR/outputs"
rm -f "$OUTPUT_DIR"/*.gen.hlo
rm -f "$OUTPUT_DIR/../summary.csv"

mkdir -p "$OUTPUT_DIR"

python3 "$SCRIPT_DIR/generate_fuzz.py" --output_dir "$OUTPUT_DIR"
python3 "$SCRIPT_DIR/summary.py" --input_dir "$OUTPUT_DIR" --output_csv "$OUTPUT_DIR/../summary.csv"

rm -rf "$SCRIPT_DIR/nnsmith_output"
rm -rf "$SCRIPT_DIR/outputs"

echo
echo "=== NNSmith Setup Complete ==="
echo "Generated HLO files in: $OUTPUT_DIR"
echo "Total: $(ls "$OUTPUT_DIR"/*.gen.hlo 2>/dev/null | wc -l) files"
