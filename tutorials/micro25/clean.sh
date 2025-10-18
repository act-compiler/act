#!/bin/bash

set -e

cd "$(dirname "$0")"/../..

rm -rf **/__pycache__/
rm -rf **/build/

rm -rf QKV.py
rm -rf targets/
rm -rf backends/

rm -rf attention.hlo
rm -rf test_qkv.py
rm -rf data/
rm -rf asm/

echo "Cleaned up the repository for a fresh start."
