#!/bin/bash
set -e

BIN_DIR="bin"
OUT_SUFFIX=".out"
REF_SUFFIX=".ref"

mkdir -p $BIN_DIR

echo "🔁 Running reproducibility tests..."

for prog in accumulation_order x87_precision math_reordering openmp_sum random_seed; do
    bin="./$BIN_DIR/$prog"
    out_file="$prog$OUT_SUFFIX"
    ref_file="$prog$REF_SUFFIX"

    echo "▶️  Running $prog..."

    $bin > "$out_file"

    if [ -f "$ref_file" ]; then
        if cmp -s "$out_file" "$ref_file"; then
            echo "✅ $prog: bitwise identical"
        else
            echo "❌ $prog: output differs"
        fi
    else
        cp "$out_file" "$ref_file"
        echo "🆕 $prog: reference saved"
    fi
done
