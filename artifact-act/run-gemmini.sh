#!/bin/bash
trap "exit" INT TERM
trap "kill 0" EXIT

set -e

export PYTHONUNBUFFERED=1

echo "########## Accelerator: Gemmini ##########"
echo

pip install -e /act --quiet --root-user-action=ignore

echo "(1) Generating Gemmini backend from ISA spec..."

cd /act/accelerators/
python gemmini.py

echo
echo "(2) Compiling Gemmini attention kernel..."

rm -rf /act/compiled_asm/gemmini
mkdir -p /act/compiled_asm/gemmini/swlib /act/compiled_asm/gemmini/fuzz

for file in /act/kernels/gemmini/swlib/*.hlo; do
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    /act/accelerators/backends/Gemmini \
        --input $file \
        --output /act/compiled_asm/gemmini/swlib/${filename}.asm.py \
        --stats
done

FUZZ_STATS=/act/compiled_asm/gemmini/fuzz/stats.log
> $FUZZ_STATS

for file in /act/kernels/gemmini/fuzz/*.hlo; do
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    echo "=== ${filename%.gen} ===" >> $FUZZ_STATS
    /act/accelerators/backends/Gemmini \
        --input $file \
        --output /act/compiled_asm/gemmini/fuzz/${filename}.asm.py \
        --stats \
        --stop-at-first 2>&1 | tee -a $FUZZ_STATS
done

echo
echo "(3) Generating compilation time plot..."
python /act/plots/plot_gemmini_fuzz.py --stats $FUZZ_STATS

echo
echo "########## Gemmini Complete ##########"
