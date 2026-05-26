#!/bin/bash

set -e

cd "$(dirname "$0")"/../..

./docker/chown.sh > /dev/null 2>&1 || true

find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name "build" -exec rm -rf {} +

rm -rf QKV.py
rm -rf targets/
rm -rf backends/

rm -rf attention.hlo
rm -rf test_qkv.py
rm -rf data/
rm -rf asm/

echo "Cleaned up the repository for a fresh start."
