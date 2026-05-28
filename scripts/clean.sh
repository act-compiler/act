#!/bin/bash

set -e

cd "$(dirname "$0")/../"

echo "Cleaning generated files..."

# Remove Python build artifacts
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove Rust build artifacts
find ./backends -type d -name "target" -exec rm -rf {} + 2>/dev/null || true

# Remove C++ build artifacts
find ./backends -type d -name "build" -exec rm -rf {} + 2>/dev/null || true

# Remove generated logs
rm -rf log/ 2>/dev/null || true
find . -name "*.log" -exec rm -f {} + 2>/dev/null || true

# Remove generated assembly files and plots
find . -name "*.asm.py" -exec rm -f {} + 2>/dev/null || true
find . -name "*.pdf" -exec rm -f {} + 2>/dev/null || true

# Remove NNSmith temp data
rm -rf kernels/setup/nnsmith/nnsmith_output 2>/dev/null || true
rm -rf kernels/setup/nnsmith/outputs 2>/dev/null || true

echo "Done."
