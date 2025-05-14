#!/bin/bash
set -e

# Directories
mkdir -p ref
OUTDIR="./outputs"
mkdir -p $OUTDIR

# Set environment (allow BLIS_ARCH_TYPE/BLIS_MODEL_TYPE override externally)
if [ -z "$BLIS_ARCH_TYPE" ]; then
    echo "Using default BLIS_ARCH_TYPE (no override)."
else
    echo "Using BLIS_ARCH_TYPE=$BLIS_ARCH_TYPE"
fi
if [ -z "$BLIS_MODEL_TYPE" ]; then
    echo "Using default BLIS_MODEL_TYPE (no override)."
else
    echo "Using BLIS_MODEL_TYPE=$BLIS_MODEL_TYPE"
fi

# Helper function to run a test and compare
run_test() {
    mode=$1     # "single" or "multi"
    prog=$2     # program name (without _st or _mt suffix)
    bin="$3"    # binary to run
    suffix=$4   # suffix for reference file ("single" or "multi")

    echo "Running $prog ($mode-thread)..."
    if [ "$mode" = "multi" ]; then
        export OMP_NUM_THREADS=4   # set multi-thread count
    else
        export OMP_NUM_THREADS=1
    fi

    outfile="$OUTDIR/${prog}.out.bin"
    ./"$bin" "$outfile"

    ref="./ref/${prog}_${suffix}.ref"
    if [ ! -f "$ref" ]; then
        # First-time reference creation
        echo "Creating reference for $prog (${mode}-thread)."
        cp "$outfile" "$ref"
    else
        # Compare output to reference
        if cmp -s "$outfile" "$ref"; then
            echo "$prog (${mode}-thread): PASS"
        else
            echo "$prog (${mode}-thread): FAIL"
            cmp "$outfile" "$ref" || true
        fi
    fi
    echo
}

# Build binaries if not built
if [ ! -x ./test_dgemm_st ]; then
    echo "Building test binaries..."
    make
fi

# Single-threaded tests
run_test "single" "dgemm" "test_dgemm_st" "single"
run_test "single" "daxpy" "test_daxpy_st" "single"
run_test "single" "dot"   "test_dot_st"   "single"

# Multi-threaded tests
run_test "multi" "dgemm" "test_dgemm_mt" "multi"
run_test "multi" "daxpy" "test_daxpy_mt" "multi"
run_test "multi" "dot"   "test_dot_mt"   "multi"

echo "All tests completed."
