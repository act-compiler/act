#!/bin/bash
trap "exit" INT TERM
trap "kill 0" EXIT

set -e

export PYTHONUNBUFFERED=1

echo "########## Accelerator: QKV ##########"
echo

pip install -e /act --quiet --root-user-action=ignore

echo "(1) Generating QKV backend from ISA spec..."

cd /act/accelerators/
python qkv.py

echo
echo "(2) Compiling QKV attention kernel..."

rm -rf /act/compiled_asm/qkv
mkdir -p /act/compiled_asm/qkv
/act/accelerators/backends/QKV \
    --input /act/kernels/qkv/qkv.hlo \
    --output /act/compiled_asm/qkv/qkv.asm.py \
    --stats

echo
echo "########## QKV Complete ##########"
